import os
import time
import json
import copy
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018
import numpy as np
from pycocotools import mask as maskUtils

from collections import OrderedDict
import cv2
from datasets import car_models
import pickle as pkl
# Apollo Given utils import
import utilities.utils as uts
import utilities.eval_utils as eval_uts
import logging


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Car3D(WAD_CVPR2018):
    def __init__(self, dataset_dir):
        """
        Constructor of ApolloScape helper class for reading and visualizing annotations.
        Modified from: https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/data.py
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.name = 'Car3D'
        self.image_shape = (2710, 3384)  # Height, Width
        self.eval_cat = ['bus', 'tricycle', 'motorcycle', 'car', 'truck', 'pedestrian', 'bicycle']
        self.data_dir = dataset_dir
        self.img_list_dir = os.path.join(self.data_dir, 'ImageLists')
        self.train_list_all = []
        self.label_list_all = []
        # Due to previous training, we need to set the order as follows
        self.eval_class = [39, 40, 34, 33, 38, 36, 35]
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.eval_class)
            }
        self.category_to_id_map = {
            'car': 33,
            'motorcycle': 34,
            'bicycle': 35,
            'pedestrian': 36,
            'rider': 37,
            'truck': 38,
            'bus': 39,
            'tricycle': 40,
            'others': 0,
            'rover': 1,
            'sky': 17,
            'car_groups': 161,
            'motorbicycle_group': 162,
            'bicycle_group': 163,
            'person_group': 164,
            'rider_group': 165,
            'truck_group': 166,
            'bus_group': 167,
            'tricycle_group': 168,
            'road': 49,
            'siderwalk': 50,
            'traffic_cone': 65,
            'road_pile': 66,
            'fence': 67,
            'traffic_light': 81,
            'pole': 82,
            'traffic_sign': 83,
            'wall': 84,
            'dustbin': 85,
            'billboard': 86,
            'building': 97,
            'bridge': 98,
            'tunnel': 99,
            'overpass': 100,
            'vegatation': 113,
            'unlabeled': 255,
        }
        self.dataset = dict()
        self.id_map_to_cat = dict(zip(self.category_to_id_map.values(), self.category_to_id_map.keys()))
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }

        # Apollo 3d init
        from collections import namedtuple
        Setting = namedtuple('Setting', ['image_name', 'data_dir'])
        setting = Setting([], self.data_dir)
        self.dataset = car_models.ApolloScape(setting)
        self._data_config = self.dataset.get_3d_car_config()
        self.car_id2name = car_models.car_id2name
        self.unique_car_models = np.array([2,  6,  7,  8,  9, 12, 14, 16, 18, 19, 20, 23, 25, 27, 28, 31, 32, 35, 37,
                                           40, 43, 46, 47, 48, 50, 51, 54, 56, 60, 61, 66, 70, 71, 76])
        self.unique_car_names = [self.car_id2name[x].name for x in self.unique_car_models]

    def get_img_list(self, list_flag='train', with_valid=False):
        """
        Get the image list,
        :param list_flag: ['train', 'val', test']
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :param roads: road indices, currently we used only [1,2,3] but 4 will also feasible
        :return:
        """

        train_list_all = [line.rstrip('\n')[:-4] for line in open(os.path.join(self.data_dir, 'split',  list_flag + '.txt'))]
        valid_list_all = []
        if with_valid:
            valid_list_all = [line.rstrip('\n')[:-4] for line in open(os.path.join(self.data_dir, 'split', 'val.txt'))]

        # The Following list of images are noisy images that should be deleted:
        train_list_delete = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'split',  'Mesh_overlay_train_error _delete.txt'))]
        val_list_delete = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'split',  'Mesh_overlay_val_error_delete.txt'))]
        noisy_list = train_list_delete + val_list_delete
        print("Train delete %d images, val delete %d images." % (len(train_list_delete), len(val_list_delete)))

        self.train_list_all = [x for x in train_list_all + valid_list_all if x not in noisy_list]
        return self.train_list_all

    def load_car_models(self):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            car_model = os.path.join(self.data_dir, 'car_models', model.name+'.pkl')
            # with open(car_model) as f:
            #     self.car_models[model.name] = pkl.load(f)
            #
            # This is a python 3 compatibility
            self.car_models[model.name] = pkl.load(open(car_model, "rb"), encoding='latin1')
            # fix the inconsistency between obj and pkl
            self.car_models[model.name]['vertices'][:, [0, 1]] *= -1
        return self.car_models

    def get_intrinsic_mat(self, image_name):
        #intrinsic = self.dataset.get_intrinsic(image_name)
        # Intrinsic should always use camera 5
        intrinsic = self.dataset.get_intrinsic('Camera_5')
        intrinsic_mat = np.zeros((3, 3))
        intrinsic_mat[0, 0] = intrinsic[0]
        intrinsic_mat[1, 1] = intrinsic[1]
        intrinsic_mat[0, 2] = intrinsic[2]
        intrinsic_mat[1, 2] = intrinsic[3]
        intrinsic_mat[2, 2] = 1
        self.intrinsic_mat = intrinsic_mat
        return intrinsic_mat

    def loadGt(self, roidb, range_idx, type='boxes'):
        """
        Load result file and return a result api object.
        :param   range     : range of image file
        :param: type      : boxes, or segms
        """
        print('Loading and preparing results...')
        res = Car3D()
        res.dataset['categories'] = copy.deepcopy(self.category_to_id_map)
        res.dataset['images'] = []
        anns = []
        count = 1
        tic = time.time()
        if range_idx is not None:
            start, end = range_idx
            for i in range(start, end):
                entry = roidb[i]
                res.dataset['images'].append({'id': entry['image']})
                if type == 'boxes':
                    for id in range(len(entry['boxes'])):
                        ann = dict()
                        ann['image_id'] = entry['image']
                        ann['category_id'] = self.contiguous_category_id_to_json_id[entry['gt_classes'][id]]
                        bb = entry['boxes'][id]
                        x1, x2, y1, y2 = bb[0], bb[2], bb[1], bb[3]
                        w = x2 - x1
                        h = y2 - y1
                        x_c = x1
                        y_c = y1
                        ann['bbox'] = [x_c, y_c, w, h]
                        ann['area'] = (bb[2] - bb[0]) * (bb[3] - bb[1])
                        ann['id'] = count
                        ann['iscrowd'] = 0
                        count += 1
                        anns.append(ann)

                elif type == 'segms':
                    for id in range(len(entry['segms'])):
                        ann = dict()
                        ann['segms'] = entry['segms'][id]
                        ann['image_id'] = entry['image']
                        ann['category_id'] = self.contiguous_category_id_to_json_id[entry['gt_classes'][id]]
                        # now only support compressed RLE format as segmentation results
                        ann['area'] = maskUtils.area(entry['segms'][id])
                        if not 'boxes' in ann:
                            ann['boxes'] = maskUtils.toBbox(ann['segms'])
                        ann['id'] = count
                        count += 1
                        ann['iscrowd'] = 0
                        anns.append(ann)

        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        print('Loading and preparing results...')
        res = Car3D()
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        res.dataset['categories'] = copy.deepcopy(self.category_to_id_map)
        if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0

        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


