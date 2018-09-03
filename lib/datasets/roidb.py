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

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
from tqdm import tqdm
import scipy
import os
import pickle

import utils.boxes as box_utils
from core.config import cfg
from .json_dataset import JsonDataset
from .json_dataset import _add_class_assignments

logger = logging.getLogger(__name__)


def combined_roidb_for_training(dataset_names, dataset_dir=None, list_flag='train'):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """

    def get_roidb(dataset_name, dataset_dir):
        ds = JsonDataset(dataset_name, dataset_dir)
        roidb = ds.get_roidb(gt=True, list_flag=list_flag)
        return roidb, ds

    roidbs, ds = get_roidb(dataset_names, dataset_dir)
    # I have cleaned very thing, so next line does not execute, but very important tho
    cache_filepath_filtered = os.path.join(ds.cache_path,  ds.name + '_' + list_flag + '_gt_roidb_filtered.pkl')
    if not os.path.exists(cache_filepath_filtered):
        if ds.name == 'Car3D':
            roidbs = filter_for_training_Car3D(roidbs, cache_filepath_filtered, ds.num_classes)
        else:
            roidbs = filter_for_training(roidbs, cache_filepath_filtered)

    if cfg.TRAIN.ASPECT_GROUPING or cfg.TRAIN.ASPECT_CROPPING:
        logger.info('Computing image aspect ratios and ordering the ratios...')
        ratio_list, ratio_index = rank_for_training(roidbs)
        logger.info('done')
    else:
        ratio_list, ratio_index = None, None

    if 'bbox_targets' not in roidbs[0].keys():
        logger.info('Computing bounding-box regression targets...')
        add_bbox_regression_targets(roidbs)
        logger.info('done')

    if dataset_names == 'Car3D':
        _compute_and_log_stats_Car3d(roidbs, ds)
    else:
        _compute_and_log_stats(roidbs, ds)

    # If 3D to 2D, we need dataset car model, a bit hack here
    if cfg.MODEL.LOSS_3D_2D_ON:
        return roidbs, ratio_list, ratio_index, ds
    else:
        return roidbs, ratio_list, ratio_index


def filter_for_training(roidb, cache_filepath_filtered):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    total_obj_count = 0
    valid_obj_count = 0
    filtered_roidb = []
    print('Remove roidb entry with small area')
    for entry in tqdm(roidb):
        filtered_entry = {}
        # we also get rid of the tiny objects
        valid_idx = []
        total_obj_count += len(entry['seg_areas'])
        for i, area in enumerate(entry['seg_areas']):
            if area >= cfg.TRAIN.MIN_AREA:
                valid_idx.append(i)
        valid_obj_count += len(valid_idx)
        obj_keys = ['boxes', 'gt_classes', 'segms', 'seg_areas', 'is_crowd', 'bbox_targets']
        total_keys = entry.keys()
        for key in total_keys:
            if key in obj_keys:
                if type(entry[key]) == list:
                    filtered_entry[key] = [entry[key][x] for x in valid_idx]
                else:
                    filtered_entry[key] = entry[key][valid_idx]
            elif key != 'gt_overlaps':
                filtered_entry[key] = entry[key]

        box_to_gt_ind_map = np.zeros((len(valid_idx)), dtype=np.int32)
        gt_overlaps = np.zeros((len(valid_idx), 8), dtype=np.float32)

        for ix in range(len(valid_idx)):
            cls = filtered_entry['gt_classes'][ix]
            box_to_gt_ind_map[ix] = ix
            gt_overlaps[ix, cls] = 1.0

        filtered_entry['box_to_gt_ind_map'] = box_to_gt_ind_map
        filtered_entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)

        # We only add the images with valid instances
        if len(entry['seg_areas']) > 0 and len(valid_idx) > 0:
            filtered_roidb.append(filtered_entry)
    _add_class_assignments(filtered_roidb)

    logger.info('Filtered {} obj entries: {} -> {}'.format(total_obj_count - valid_obj_count, total_obj_count, valid_obj_count))
    logger.info('Filtered {} img entries: {} -> {}'.format(len(roidb) - len(filtered_roidb), len(roidb), len(filtered_roidb)))

    with open(cache_filepath_filtered, 'wb') as fp:
        pickle.dump(filtered_roidb, fp, pickle.HIGHEST_PROTOCOL)
    logger.info('Cache ground truth roidb to %s', cache_filepath_filtered)

    return filtered_roidb


def filter_for_training_Car3D(roidb, cache_filepath_filtered, num_classes):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    total_obj_count = 0
    valid_obj_count = 0
    filtered_roidb = []
    print('Remove roidb entry with small area')
    for entry in tqdm(roidb):
        filtered_entry = {}
        # we also get rid of the tiny objects
        valid_idx = []
        total_obj_count += len(entry['seg_areas'])
        for i, area in enumerate(entry['seg_areas']):
            if area >= cfg.TRAIN.MIN_AREA:
                valid_idx.append(i)
        valid_obj_count += len(valid_idx)
        obj_keys = ['boxes', 'car_cat_classes', 'seg_areas', 'is_crowd', 'bbox_targets', 'poses', 'car_cat_classes']
        total_keys = entry.keys()
        for key in total_keys:
            if key in obj_keys:
                if type(entry[key]) == list:
                    filtered_entry[key] = [entry[key][x] for x in valid_idx]
                else:
                    filtered_entry[key] = entry[key][valid_idx]
            elif key != 'gt_overlaps':
                filtered_entry[key] = entry[key]

        box_to_gt_ind_map = np.zeros((len(valid_idx)), dtype=np.int32)
        gt_overlaps = np.zeros((len(valid_idx), 8), dtype=np.float32)

        for ix in range(len(valid_idx)):
            box_to_gt_ind_map[ix] = ix
            # this is a legecy network from WAD MaskRCNN
            gt_overlaps[ix, 4] = 1.0

        filtered_entry['box_to_gt_ind_map'] = box_to_gt_ind_map
        filtered_entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
        filtered_entry['gt_classes'] = np.ones(filtered_entry['car_cat_classes'].shape) * 4
        filtered_entry['gt_classes'] = filtered_entry['gt_classes'].astype(np.int8)
        # We only add the images with valid instances
        if len(entry['seg_areas']) > 0 and len(valid_idx) > 0:
            filtered_roidb.append(filtered_entry)

    _add_class_assignments(filtered_roidb, allow_zero=True)

    logger.info('Filtered {} obj entries: {} -> {}'.format(total_obj_count - valid_obj_count, total_obj_count, valid_obj_count))
    logger.info('Filtered {} img entries: {} -> {}'.format(len(roidb) - len(filtered_roidb), len(roidb), len(filtered_roidb)))

    with open(cache_filepath_filtered, 'wb') as fp:
        pickle.dump(filtered_roidb, fp, pickle.HIGHEST_PROTOCOL)
    logger.info('Cache ground truth roidb to %s', cache_filepath_filtered)

    return filtered_roidb


def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index


def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in tqdm(roidb):
        entry['bbox_targets'] = _compute_targets(entry)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _compute_and_log_stats(roidb, ds):
    classes = ds.classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.info('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.info('{:d}{:s}: {:d}'.format(i, classes[i].rjust(char_len), v))
    logger.info('-' * char_len)
    logger.info('{:s}: {:d}'.format('total'.rjust(char_len), np.sum(gt_hist)))


def _compute_and_log_stats_Car3d(roidb, ds):
    """ We compute the statistics for each car models"""
    classes = ds.Car3D.unique_car_names
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes)+1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_classes = entry['car_cat_classes']
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.info('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.info('{:d}{:s}: {:d}'.format(i, classes[i].rjust(char_len), v))
    logger.info('-' * char_len)
    logger.info('{:s}: {:d}'.format('total'.rjust(char_len), np.sum(gt_hist)))
