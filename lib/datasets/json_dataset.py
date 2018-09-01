# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import os

import numpy as np
import scipy.sparse
import pickle
from tqdm import tqdm


# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu

envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
import utils.segms as segm_utils
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX
from .dataloader_wad_cvpr2018 import WAD_CVPR2018
from .dataloader_apolloscape import ApolloScape
from .dataloader_3d_car import Car3D
from utilities.utils import euler_angles_to_rotation_matrix, euler_angles_to_quaternions

from PIL import Image
import json
import cv2

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name, dataset_dir=None):
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.debug_timer = Timer()
        self.dataset_name = name
        if self.dataset_name == 'ApolloScape':
            self.ApolloScape = ApolloScape()
            self.classes = ['__background__'] + [c for c in self.ApolloScape.eval_cat]
            self.num_classes = len(self.ApolloScape.eval_cat) + 1
            self.keypoints = None
            self.eval_class = self.ApolloScape.eval_class

        elif self.dataset_name == 'Car3D':
            self.Car3D = Car3D(dataset_dir)
            self.car_models = self.Car3D.load_car_models()
            self.classes = ['__background__'] + [c for c in self.Car3D.eval_cat]
            self.num_classes = len(self.Car3D.unique_car_models)
            self.keypoints = None
            self.eval_class = self.Car3D.eval_class

        elif self.dataset_name == 'wad':
            self.WAD_CVPR2018 = WAD_CVPR2018(dataset_dir)
            self.classes = ['__background__'] + [c for c in self.WAD_CVPR2018.eval_cat]
            self.num_classes = len(self.WAD_CVPR2018.eval_cat) + 1
            self.keypoints = None
            self.eval_class = self.WAD_CVPR2018.eval_class

        elif self.dataset_name == 'coco_2017_train':
            self.COCO = COCO(DATASETS[name][ANN_FN])
            # Set up dataset classes
            category_ids = self.COCO.getCatIds()
            categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
            self.category_to_id_map = dict(zip(categories, category_ids))
            self.classes = ['__background__'] + categories
            self.num_classes = len(self.classes)
            self.json_category_id_to_contiguous_id = {
                v: i + 1
                for i, v in enumerate(self.COCO.getCatIds())
            }
            self.contiguous_category_id_to_json_id = {
                v: k
                for k, v in self.json_category_id_to_contiguous_id.items()
            }
            self._init_keypoints()

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map', 'width', 'height']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            gt=False,
            crowd_filter_thresh=0,
            list_flag='train',
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
        :param list_flag: ['train', 'val', test']

           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        cache_filepath = os.path.join(self.cache_path, self.name + '_' + list_flag + '_gt_roidb.pkl')

        if gt and os.path.exists(cache_filepath):
            # check whether we have our previously precomputed filtered ROI
            cache_filepath_filtered = os.path.join(self.cache_path, self.name + '_' + list_flag + '_gt_roidb_filtered.pkl')
            if os.path.exists(cache_filepath_filtered):
                with open(cache_filepath_filtered, 'rb') as fp:
                    roidb = pickle.load(fp)
                print('Load_gt_from_cache with filtered ops.')
                return roidb
            else:
                # Include ground-truth object annotations
                with open(cache_filepath, 'rb') as fp:
                    roidb = pickle.load(fp)
                print('Load_gt_from_cache')
                return roidb
        else:
            # We recalculate the data
            if self.dataset_name == 'coco_2017_train':
                image_ids = self.COCO.getImgIds()
                image_ids.sort()
                roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
                for entry in roidb:
                    self._prep_roidb_entry(entry)
            elif self.dataset_name == 'wad':
                # image_ids = os.listdir(self.WAD_CVPR2018.train_image_dir)[:10]  # This is for debug use
                image_ids = os.listdir(self.WAD_CVPR2018.train_image_dir)
                roidb = []
                for entry in image_ids:
                    roidb.append(self._prep_roidb_entry_wad(entry))
            elif self.dataset_name == 'ApolloScape':
                # image_ids = os.listdir(self.WAD_CVPR2018.train_image_dir)[:10]
                image_ids = self.ApolloScape.get_img_list(list_flag=list_flag)
                roidb = []
                for entry in image_ids:
                    roidb.append(self._prep_roidb_entry_ApolloScape(entry))

            elif self.dataset_name == 'Car3D':
                # image_ids = os.listdir(self.WAD_CVPR2018.train_image_dir)[:10]
                image_ids = self.Car3D.get_img_list(list_flag=list_flag, with_valid=False)
                roidb = []
                for entry in image_ids:
                    roidb.append(self._prep_roidb_entry_Car3D(entry))
            if gt and not list_flag == 'test':
                self.debug_timer.tic()
                for entry in tqdm(roidb):
                    if self.dataset_name == 'coco_2017_train':
                        self._add_gt_annotations(entry)
                    elif self.dataset_name == 'wad':
                        self._add_gt_annotations_wad(entry)
                    elif self.dataset_name == 'ApolloScape':
                        self._add_gt_annotations_ApolloScape(entry)
                    elif self.dataset_name == 'Car3D':
                        self._add_gt_annotations_Car3d(entry)

                logger.debug('_add_gt_annotations took {:.3f}s'.format(self.debug_timer.toc(average=False)))

                if cfg.TRAIN.USE_FLIPPED and not list_flag == 'val':
                    logger.info('Appending horizontally-flipped training examples...')
                    extend_with_flipped_entries(roidb)
                logger.info('Loaded dataset: {:s}'.format(self.name + '_' + list_flag))

            if not self.dataset_name == 'Car3D':
                # for Car3D, we don't have background class
                _add_class_assignments(roidb)

            with open(cache_filepath, 'wb') as fp:
                pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
            logger.info('Cache ground truth roidb to %s', cache_filepath)

            return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _prep_roidb_entry_wad(self, entry_id):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry = {}
        entry['entry_id'] = entry_id
        # Make file_name an abs path
        im_path = os.path.join(self.image_directory, self.image_prefix + entry_id)
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)

        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        return entry

    def _prep_roidb_entry_ApolloScape(self, entry_id):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry = {}
        entry['entry_id'] = entry_id
        # Make file_name an abs path
        im_path = os.path.join(self.ApolloScape.data_dir, entry_id)
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)

        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty(0, dtype=np.int32)
        entry['seg_areas'] = np.empty(0, dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(np.empty((0, self.num_classes), dtype=np.float32))
        entry['is_crowd'] = np.empty(0, dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty(0, dtype=np.int32)
        return entry

    def _prep_roidb_entry_Car3D(self, entry_id):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry = {}
        entry['entry_id'] = entry_id
        # Make file_name an abs path
        im_path = os.path.join(self.Car3D.data_dir, 'images', entry_id+'.jpg')
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)

        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        entry['has_poses'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty(0, dtype=np.int32)
        entry['seg_areas'] = np.empty(0, dtype=np.float32)

        # this is a legacy network from WAD Mask-RCNN
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(np.empty((0, 8), dtype=np.float32))
        entry['is_crowd'] = np.empty(0, dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty(0, dtype=np.int32)

        # newly added for 3d car
        entry['visible_rate'] = np.empty(0, dtype=np.float32)
        entry['poses'] = np.empty((0, 6), dtype=np.float32)
        entry['car_cat_classes'] = np.empty(0, dtype=np.int32)
        entry['quaternions'] = np.empty((0, 4), dtype=np.float32)
        return entry

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_gt_annotations_wad(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        label_image_name = entry['entry_id'][:-4] + '_instanceIds.png'
        # color_img = Image.open(os.path.join(self.WAD_CVPR2018.train_image_dir, entry['entry_id']))
        label_image = os.path.join(self.WAD_CVPR2018.train_label_dir, label_image_name)
        assert os.path.exists(label_image), 'Label \'{}\' not found'.format(label_image)
        l_img = Image.open(label_image)
        l_img = np.asarray(l_img)

        entry['height'] = self.WAD_CVPR2018.image_shape[0]
        entry['width'] = self.WAD_CVPR2018.image_shape[1]
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []

        for label in np.unique(l_img):
            class_id = label // 1000
            if class_id in self.WAD_CVPR2018.eval_class:
                area = np.sum(l_img == label)
                if area < cfg.TRAIN.GT_MIN_AREA:
                    continue
                # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
                mask = l_img == label
                mask_f = np.array(mask, order='F', dtype=np.uint8)
                rle = COCOmask.encode(mask_f)
                valid_segms.append(rle)

                xd, yd = np.where(mask)
                x1, y1, x2, y2 = yd.min(), xd.min(), yd.max(), xd.max()
                x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(x1, y1, x2, y2, entry['height'], entry['width'])
                # Require non-zero seg area and more than 1x1 box size\
                obj = {'area': area, 'clean_bbox': [x1, y1, x2, y2], 'category_id': class_id}
                valid_objs.append(obj)

        num_valid_objs = len(valid_objs)
        boxes = np.zeros((num_valid_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_valid_objs), dtype=np.int32)
        gt_overlaps = np.zeros((num_valid_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_valid_objs), dtype=np.float32)
        is_crowd = np.zeros((num_valid_objs), dtype=np.bool)
        box_to_gt_ind_map = np.zeros((num_valid_objs), dtype=np.int32)

        for ix, obj in enumerate(valid_objs):
            cls = self.WAD_CVPR2018.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = False  # TODO: What's this flag for?
            box_to_gt_ind_map[ix] = ix
            gt_overlaps[ix, cls] = 1.0

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(entry['gt_overlaps'].toarray(), gt_overlaps, axis=0)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(entry['box_to_gt_ind_map'], box_to_gt_ind_map)

    def _add_gt_annotations_ApolloScape(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        label_dir = os.path.join(entry['entry_id'].split('/')[0], 'Label', '/'.join(entry['entry_id'].split('/')[2:]))
        label_image_name = label_dir[:-4] + '_instanceIds.png'
        # color_img = Image.open(os.path.join(self.WAD_CVPR2018.train_image_dir, entry['entry_id']))
        label_image = os.path.join(self.ApolloScape.data_dir, label_image_name)
        assert os.path.exists(label_image), 'Label \'{}\' not found'.format(label_image)
        l_img = Image.open(label_image)
        l_img = np.asarray(l_img)

        entry['height'] = self.ApolloScape.image_shape[0]
        entry['width'] = self.ApolloScape.image_shape[1]
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []

        for label in np.unique(l_img):
            class_id = label // 1000
            if class_id in self.ApolloScape.eval_class:
                area = np.sum(l_img == label)
                if area < cfg.TRAIN.GT_MIN_AREA:
                    continue
                # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
                mask = l_img == label
                mask_f = np.array(mask, order='F', dtype=np.uint8)
                rle = COCOmask.encode(mask_f)
                valid_segms.append(rle)

                xd, yd = np.where(mask)
                x1, y1, x2, y2 = yd.min(), xd.min(), yd.max(), xd.max()
                x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(x1, y1, x2, y2, entry['height'], entry['width'])
                # Require non-zero seg area and more than 1x1 box size\
                obj = {'area': area, 'clean_bbox': [x1, y1, x2, y2], 'category_id': class_id}
                valid_objs.append(obj)

        num_valid_objs = len(valid_objs)
        boxes = np.zeros((num_valid_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_valid_objs), dtype=np.int32)
        gt_overlaps = np.zeros((num_valid_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_valid_objs), dtype=np.float32)
        is_crowd = np.zeros((num_valid_objs), dtype=np.bool)
        box_to_gt_ind_map = np.zeros((num_valid_objs), dtype=np.int32)

        for ix, obj in enumerate(valid_objs):
            cls = self.ApolloScape.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = False  # TODO: What's this flag for?
            box_to_gt_ind_map[ix] = ix
            gt_overlaps[ix, cls] = 1.0

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(entry['gt_overlaps'].toarray(), gt_overlaps, axis=0)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(entry['box_to_gt_ind_map'], box_to_gt_ind_map)

    def _add_gt_annotations_Car3d(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        entry_id = entry['entry_id']
        # Make file_name an abs path
        car_pose_file = os.path.join(self.Car3D.data_dir, 'car_poses', entry_id+'.json')
        assert os.path.exists(car_pose_file), 'Label \'{}\' not found'.format(car_pose_file)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        entry['height'] = self.Car3D.image_shape[0]
        entry['width'] = self.Car3D.image_shape[1]

        intrinsic_mat = self.Car3D.get_intrinsic_mat()
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for i, car_pose in enumerate(car_poses):
            car_name = self.Car3D.car_id2name[car_pose['car_id']].name
            car = self.car_models[car_name]
            pose = np.array(car_pose['pose'])

            # project 3D points to 2d image plane
            rot_mat = euler_angles_to_rotation_matrix(pose[:3])
            rvect, _ = cv2.Rodrigues(rot_mat)
            imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), rvect, pose[3:], intrinsic_mat, distCoeffs=None)

            imgpts = np.int32(imgpts).reshape(-1, 2)

            x1, y1, x2, y2 = imgpts[:, 0].min(), imgpts[:, 1].min(), imgpts[:, 0].max(), imgpts[:, 1].max()
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(x1, y1, x2, y2, entry['height'], entry['width'])
            # Require non-zero seg area and more than 1x1 box size\
            obj = {'area': car_pose['area'], 'clean_bbox': [x1, y1, x2, y2], 'category_id': 33,
                   'car_id': car_pose['car_id'], 'visible_rate': car_pose['visible_rate'],
                   'pose': car_pose['pose']}

            valid_objs.append(obj)

        num_valid_objs = len(valid_objs)
        boxes = np.zeros((num_valid_objs, 4), dtype=np.float32)
        # this is a legacy network from WAD Mask-RCNN
        car_class = 4
        gt_overlaps = np.zeros((num_valid_objs, 8), dtype=np.float32)
        seg_areas = np.zeros((num_valid_objs), dtype=np.float32)
        is_crowd = np.zeros((num_valid_objs), dtype=np.bool)
        box_to_gt_ind_map = np.zeros((num_valid_objs), dtype=np.int32)

        # newly added for 3d car
        visible_rate = np.zeros((num_valid_objs), dtype=np.float32)
        poses = np.zeros((num_valid_objs, 6), dtype=np.float32)
        quaternions = np.zeros((num_valid_objs, 4), dtype=np.float32)

        car_cat_classes = np.zeros((num_valid_objs), dtype=np.int32)

        for ix, obj in enumerate(valid_objs):
            cls = np.where(self.Car3D.unique_car_models == obj['car_id'])[0][0]
            boxes[ix, :] = obj['clean_bbox']
            car_cat_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = False  # TODO: What's this flag for?
            box_to_gt_ind_map[ix] = ix
            gt_overlaps[ix, car_class] = 1.0
            visible_rate[ix] = obj['visible_rate']
            poses[ix] = obj['pose']
            quaternions[ix] = euler_angles_to_quaternions(np.array([obj['pose'][:3]]))

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(entry['gt_overlaps'].toarray(), gt_overlaps, axis=0)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(entry['box_to_gt_ind_map'], box_to_gt_ind_map)
        # newly added for 3d car
        entry['visible_rate'] = np.append(entry['visible_rate'], visible_rate)
        entry['poses'] = np.append(entry['poses'], poses, axis=0)
        entry['car_cat_classes'] = np.append(entry['car_cat_classes'], car_cat_classes)
        entry['quaternions'] = np.append(entry['quaternions'], quaternions, axis=0)

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        return cached_roidb

    def _add_proposals_from_file(self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros((num_boxes, entry['gt_overlaps'].shape[1]), dtype=entry['gt_overlaps'].dtype)
        box_to_gt_ind_map = -np.ones((num_boxes), dtype=entry['box_to_gt_ind_map'].dtype)

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb, allow_zero=False):
    """Compute object category assignment for each box associated with each roidb entry.
    allow_zero: whether the 0 is the backgroud class, default False-->BG is zeros
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        if not allow_zero:
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]


def extend_with_flipped_entries(roidb):
    """Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.

    "Flipping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are horizontally flipped.
    """
    flipped_roidb = []
    for entry in tqdm(roidb):
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        flipped_entry['segms'] = segm_utils.flip_segms(
            entry['segms'], entry['height'], entry['width']
        )
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)
    return roidb
