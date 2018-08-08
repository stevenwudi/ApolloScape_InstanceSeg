import os

import numpy as np


class ApolloScape(object):
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
        self.eval_cat = {'bus', 'tricycle', 'motorcycle', 'car', 'truck', 'pedestrian', 'bicycle'}
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

    def get_train_img_list(self, with_valid=False, roads=[1, 2, 3]):
        """
        Get the image list,
        :param with_valid:  if with_valid set to True, then validation data is also used for training
        :param roads: road indices, currently we used only [1,2,3] but 4 will also feasible
        :return:
        """
        train_list_all = []
        label_list_all = []
        for road_idx in roads:
            file_name = 'road%02d_ins_train.lst' % road_idx
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
