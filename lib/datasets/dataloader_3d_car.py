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

        self.train_list_all = train_list_all + valid_list_all
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
        intrinsic = self.dataset.get_intrinsic(image_name)
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


class CarPoseVisualizer(object):
    def __init__(self, args=None, scale=1.0, linewidth=0.):
        """Initializer
        Input:
            scale: whether resize the image in case image is too large
            linewidth: 0 indicates a binary mask, while > 0 indicates
                       using a frame.
        """
        self.dataset = car_models.ApolloScape(args)
        self._data_config = self.dataset.get_3d_car_config()

        self.MAX_DEPTH = 1e4
        self.MAX_INST_NUM = 100
        h, w = self._data_config['image_size']
        # must round prop to 4 due to renderer requirements
        # this will change the original size a bit, we usually need rescale
        # due to large image size
        self.image_size = np.uint32(uts.round_prop_to(np.float32([h * scale, w * scale])))
        self.scale = scale
        self.linewidth = linewidth
        self.colors = np.random.random((self.MAX_INST_NUM, 3)) * 255

    def load_car_models(self):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            car_model = '%s%s.pkl' % (self._data_config['car_model_dir'], model.name)
            # with open(car_model) as f:
            #     self.car_models[model.name] = pkl.load(f)
            #
            # This is a python 3 compatibility
            self.car_models[model.name] = pkl.load(open(car_model, "rb"), encoding='latin1')
            # fix the inconsistency between obj and pkl
            self.car_models[model.name]['vertices'][:, [0, 1]] *= -1

    def render_car_cv2(self, pose, car_name, image):
        """Render a car instance given pose and car_name
        """
        car = self.car_models[car_name]
        pose = np.array(pose)
        # project 3D points to 2d image plane
        imgpts, jac = cv2.projectPoints(np.float32(car['vertices']), pose[:3], pose[3:], self.intrinsic, distCoeffs=np.asarray([]))

        mask = np.zeros(image.shape)
        for face in car['faces'] - 1:
            pts = np.array([[imgpts[idx, 0, 0], imgpts[idx, 0, 1]] for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], True, (0, 255, 0))

        return mask

    def compute_reproj_sim(self, car_names, out_file=None):
        """Compute the similarity matrix between every pair of cars.
        """
        if out_file is None:
            out_file = './sim_mat.txt'

        sim_mat = np.eye(len(self.car_model))
        for i in range(len(car_names)):
            for j in range(i, len(car_names)):
                name1 = car_names[i][0]
                name2 = car_names[j][0]
                ind_i = self.car_model.keys().index(name1)
                ind_j = self.car_model.keys().index(name2)
                sim_mat[ind_i, ind_j] = self.compute_reproj(name1, name2)
                sim_mat[ind_j, ind_i] = sim_mat[ind_i, ind_j]

        np.savetxt(out_file, sim_mat, fmt='%1.6f')

    def compute_reproj(self, car_name1, car_name2):
        """Compute reprojection error between two cars
        """
        sims = np.zeros(10)
        for i, rot in enumerate(np.linspace(0, np.pi, num=10)):
            pose = np.array([0, rot, 0, 0, 0, 5.5])
            depth1, mask1 = self.render_car(pose, car_name1)
            depth2, mask2 = self.render_car(pose, car_name2)
            sims[i] = eval_uts.IOU(mask1, mask2)

        return np.mean(sims)

    def merge_inst(self,
                   depth_in,
                   inst_id,
                   total_mask,
                   total_depth):
        """Merge the prediction of each car instance to a full image
        """

        render_depth = depth_in.copy()
        render_depth[render_depth <= 0] = np.inf
        depth_arr = np.concatenate([render_depth[None, :, :],
                                    total_depth[None, :, :]], axis=0)
        idx = np.argmin(depth_arr, axis=0)

        total_depth = np.amin(depth_arr, axis=0)
        total_mask[idx == 0] = inst_id

        return total_mask, total_depth

    def rescale(self, image, intrinsic):
        """resize the image and intrinsic given a relative scale
        """

        intrinsic_out = uts.intrinsic_vec_to_mat(intrinsic, self.image_size)
        hs, ws = self.image_size
        image_out = cv2.resize(image.copy(), (ws, hs))

        return image_out, intrinsic_out

    def showAnn(self, image_name, settings):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """
        from matplotlib import pyplot as plt

        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        intrinsic = self.dataset.get_intrinsic(image_name)
        image, self.intrinsic = self.rescale(image, intrinsic)
        im_shape = image.shape
        mask_all = np.zeros(im_shape)
        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            mask = self.render_car_cv2(car_pose['pose'], car_name, im_shape)
            mask_all += mask

        mask_all = mask_all*200 / mask_all.max()
        merged_image = cv2.addWeighted(image.astype(np.uint8), 1.0, mask_all.astype(np.uint8), 0.8, 0)
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(merged_image)
        fig.savefig('/home/stevenwudi/PycharmProjects/dataset-api/Outputs/imgs/' +
                    settings + '/' + image_name + '.png', dpi=300)
        return image

    def findArea(self, image_name):
        """accumuate the areas of cars in an image
        Input:
            image_name: the name of image
        Output:

        """
        car_pose_file = '%s/%s.json' % (self._data_config['pose_dir'], image_name)
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        areas = []
        for pose in car_poses:
            areas.append(pose['area'])
        return areas


class LabelResaver(object):
    """ Resave the raw labeled file to the required json format for evaluation
    """

    # (TODO Peng) Figure out why running pdb it is correct, but segment fault when
    # running
    def __init__(self, args):
        self.visualizer = CarPoseVisualizer(args, scale=0.5)
        self.visualizer.load_car_models()

    def strs_to_mat(self, strs):
        """convert str to numpy matrix
        """
        assert len(strs) == 4
        mat = np.zeros((4, 4))
        for i in range(4):
            mat[i, :] = np.array([np.float32(str_f) for str_f in strs[i].split(' ')])

        return mat

    def read_car_pose(self, file_name):
        """load the labelled car pose
        """
        cars = []
        lines = [line.strip() for line in open(file_name)]
        i = 0
        while i < len(lines):
            car = OrderedDict([])
            line = lines[i].strip()
            if 'Model Name :' in line:
                car_name = line[len('Model Name : '):]
                car['car_id'] = car_models.car_name2id[car_name].id
                pose = self.strs_to_mat(lines[i + 2: i + 6])
                pose[:3, 3] = pose[:3, 3] / 100.0  # convert cm to meter
                rot = uts.rotation_matrix_to_euler_angles(
                    pose[:3, :3], check=False)
                trans = pose[:3, 3].flatten()
                pose = np.hstack([rot, trans])
                car['pose'] = pose
                i += 6
                cars.append(car)
            else:
                i += 1

        return cars

    def convert(self, pose_file_in, pose_file_out):
        """ Convert the raw labelled file to required json format
        Input:
            file_name: str filename
        """
        car_poses = self.read_car_pose(pose_file_in)
        car_num = len(car_poses)
        MAX_DEPTH = self.visualizer.MAX_DEPTH
        image_size = self.visualizer.image_size
        intrinsic = self.visualizer.dataset.get_intrinsic(pose_file_in)
        self.visualizer.intrinsic = uts.intrinsic_vec_to_mat(intrinsic,
                                                             image_size)
        self.depth = MAX_DEPTH * np.ones(image_size)
        self.mask = np.zeros(self.depth.shape)
        vis_rate = np.zeros(car_num)

        for i, car_pose in enumerate(car_poses):
            car_name = car_models.car_id2name[car_pose['car_id']].name
            depth, mask = self.visualizer.render_car(car_pose['pose'], car_name)
            self.mask, self.depth = self.visualizer.merge_inst(
                depth, i + 1, self.mask, self.depth)
            vis_rate[i] = np.float32(np.sum(mask == (i + 1))) / (
                np.float32(np.sum(mask)) + np.spacing(1))

        keep_idx = []
        for i, car_pose in enumerate(car_poses):
            area = np.round(np.float32(np.sum(self.mask == (i + 1))) / (self.visualizer.scale ** 2))
            if area > 1:
                keep_idx.append(i)

            car_pose['pose'] = car_pose['pose'].tolist()
            car_pose['area'] = int(area)
            car_pose['visible_rate'] = float(vis_rate[i])
            keep_idx.append(i)

        car_poses = [car_poses[idx] for idx in keep_idx]
        with open(pose_file_out, 'w') as f:
            json.dump(car_poses, f, sort_keys=True, indent=4,
                      ensure_ascii=False)


