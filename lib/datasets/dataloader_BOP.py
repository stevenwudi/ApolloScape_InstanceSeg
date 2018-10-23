import os
import yaml
import time
from datasets.dataloader_wad_cvpr2018 import WAD_CVPR2018


class BOP(WAD_CVPR2018):
    def __init__(self, dataset_dir, dataset_name='TLESS', num_classes=30):
        """
        Constructor of ApolloScape helper class for reading and visualizing annotations.
        Modified from: https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/data.py
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.name = dataset_name
        #self.image_shape = (1280, 1024)  # Height, Width
        self.image_shape = (400, 400)  # Height, Width
        self.eval_cat = [str(i) for i in range(num_classes)]
        self.data_dir = dataset_dir
        self.img_list_dir = os.path.join(self.data_dir, 'ImageLists')
        self.train_list_all = []
        self.label_list_all = []
        # Due to previous training, we need to set the order as follows
        self.eval_class = [i for i in range(num_classes)]
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.eval_class)
            }
        self.dataset = dict()
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.gt_all = {}
        self.info_all = {}

    def get_img_list(self, list_flag='train'):
        """
        Get the image list,
        :param list_flag: ['train', 'val', test']
        :return:
        """
        list_all = []
        tstart = time.time()
        train_models = os.listdir(os.path.join(self.data_dir, list_flag))
        for model_num in sorted(train_models):
            img_list = sorted(os.listdir(os.path.join(self.data_dir, list_flag, model_num, 'rgb')))
            list_tmp = [os.path.join(self.data_dir, list_flag, model_num, 'rgb', x) for x in img_list]
            list_all.append(list_tmp)
            # We also read the GT info here
            # Read YAML file
            with open(os.path.join(self.data_dir, list_flag, model_num, "gt.yml"), 'r') as stream:
                self.gt_all[model_num] = yaml.load(stream)

            with open(os.path.join(self.data_dir, list_flag, model_num, "info.yml"), 'r') as stream:
                self.info_all[model_num] = yaml.load(stream)

        self.list_all = [item for sublist in list_all for item in sublist]
        print('Data loading elapsed: %s' % (time.time() - tstart))
        return self.list_all

