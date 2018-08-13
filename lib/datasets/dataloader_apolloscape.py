import os
import time
import json
import copy
from collections import defaultdict
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018
import itertools
import numpy as np
from pycocotools import mask as maskUtils
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class ApolloScape(WAD_CVPR2018):
    def __init__(self):
        """
        Constructor of ApolloScape helper class for reading and visualizing annotations.
        Modified from: https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/data.py
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.name = 'ApolloScape'
        self.image_shape = (2710, 3384)  # Height, Width
        self.eval_cat = ['bus', 'tricycle', 'motorcycle', 'car', 'truck', 'pedestrian', 'bicycle']
        self.data_dir = '/media/samsumg_1tb/ApolloScape'
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

    def get_img_list(self, list_flag='train', with_valid=False, roads=[1, 2, 3]):
        """
        Get the image list,
        :param list_flag: ['train', 'val', test']
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :param roads: road indices, currently we used only [1,2,3] but 4 will also feasible
        :return:
        """
        train_list_all = []
        label_list_all = []
        for road_idx in roads:
            file_name = 'road%02d' % road_idx + '_ins_' + list_flag + '.lst'
            train_list_file = os.path.join(self.img_list_dir, file_name)
            lines = [line.rstrip('\n') for line in open(train_list_file)]
            for line in lines:
                img_name = line.split('\t')[0]
                label_name = line.split('\t')[1]
                train_list_all.append(img_name)
                label_list_all.append(label_name)

            if with_valid:
                file_name = 'road%02d_ins_val.lst' % road_idx
                train_list_file = os.path.join(self.img_list_dir, file_name)
                lines = [line.rstrip('\n') for line in open(train_list_file)]
                for line in lines:
                    img_name = line.split('\t')[0]
                    label_name = line.split('\t')[1]
                    train_list_all.append(img_name)
                    label_list_all.append(label_name)

        self.train_list_all = train_list_all
        self.label_list_all = label_list_all
        return self.train_list_all

    def get_3d_car_config(self):
        """get configuration of the dataset for 3d car understanding
        """
        ROOT = self._data_dir + '3d_car_instance/' if self._args is None else \
            self._args.data_dir

        self._data_config['image_dir'] = ROOT + 'images/'
        self._data_config['pose_dir'] = ROOT + 'car_poses/'
        self._data_config['train_list'] = ROOT + 'split/train.txt'
        self._data_config['val_list'] = ROOT + 'split/val.txt'
        self._data_config['image_size'] = [2710, 3384]
        self._data_config['intrinsic'] = {
            'Camera_5': np.array(
                [2304.54786556982, 2305.875668062,
                 1686.23787612802, 1354.98486439791]),
            'Camera_6': np.array(
                [2300.39065314361, 2301.31478860597,
                 1713.21615190657, 1342.91100799715])}

        # normalized intrinsic
        cam_names = self._data_config['intrinsic'].keys()
        for c_name in cam_names:
            self._data_config['intrinsic'][c_name][[0, 2]] /= self._data_config['image_size'][1]
            self._data_config['intrinsic'][c_name][[1, 3]] /= self._data_config['image_size'][0]
        self._data_config['car_model_dir'] = ROOT + 'car_models/'

        return self._data_config

    def get_intrinsic(self, image_name):
        assert self._data_config
        for name in self._data_config['intrinsic'].keys():
            if name in image_name:
                return self._data_config['intrinsic'][name]
        raise ValueError('%s has no provided intrinsic' % image_name)

    def loadGt(self, roidb, range_idx, type='boxes'):
        """
        Load result file and return a result api object.
        :param   range     : range of image file
        :param: type      : boxes, or segms
        """
        print('Loading and preparing results...')
        res = ApolloScape()
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
        res = ApolloScape()
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


