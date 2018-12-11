"""Perform inference on one or more datasets."""

import argparse
import numpy as np
import os
from collections import OrderedDict
import logging
from utilities import car_models as car_models_all
import pickle as pkl
import _init_paths  # pylint: disable=unused-import
from utilities.eval_local import Detect3DEval
from datasets import task_evaluation
from core.config import cfg
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
import glob
from utils.io import save_object, load_object
from datasets.json_dataset import JsonDataset


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', default='ApolloScape', help='Dataset to use')
    parser.add_argument('--dataset_dir', default='/media/samsumg_1tb/ApolloScape/ECCV2018_apollo/train/')

    parser.add_argument('--load_ckpt', default='/media/samsumg_1tb/ApolloScape/ApolloScape_InstanceSeg/e2e_3d_car_101_FPN_triple_head_non_local_weighted/Nov03-21-05-13_N606-TITAN32_step/ckpt/model_step46952.pth', help='checkpoint path to load')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/e2e_3d_car_101_FPN_triple_head_non_local_weighted.yaml', help='Config file for training (and optionally testing)')
    parser.add_argument('--set', dest='set_cfgs', help='set config keys, will overwrite config in the cfg_file. See lib/core/config.py for all options', default=[], nargs='*')

    parser.add_argument('--test_dir', default='/media/samsumg_1tb/ApolloScape/ApolloScape_InstanceSeg/e2e_3d_car_101_FPN/Aug31-11-41-25_N606-TITAN32_step/test/json_val_trans_not_filtered')
    parser.add_argument('--list_flag', default='val', help='Choosing between [val, test]')
    parser.add_argument('--iou_ignore_threshold', default=.5, help='Filter out by this iou')
    parser.add_argument('--simType', default=None, help='Detection Score for visualisation')
    parser.add_argument('--dtScores', default=0.1, help='Detection Score for visualisation')
    return parser.parse_args()


def open_3d_vis(args):
    # The following evaluate the detection result from Faster-RCNN Head
    # if args.range is not None:
    #     if cfg.TEST.SOFT_NMS.ENABLED:
    #         det_name = 'detection_range_%s_%s_soft_nms' % tuple(args.range)
    #     else:
    #         det_name = 'detection_range_(%d_%d)_nms_%.1f' % (args.range[0], args.range[1], cfg.TEST.NMS)
    #     if cfg.TEST.BBOX_AUG.ENABLED:
    #         det_name += '_multiple_scale'
    #     det_name += '.pkl'
    # else:
    #     det_name = 'detections.pkl'
    #
    # det_files = sorted(glob.glob(args.output_dir+'/detection_range_*.pkl'))
    #
    # det_file = det_files[0]
    # if os.path.exists(det_file):
    #     obj = load_object(det_file)
    #     all_boxes = obj['all_boxes']
    #
    # dataset = JsonDataset(args.dataset, args.dataset_dir)
    #
    # results = task_evaluation.evaluate_boxes(dataset, all_boxes, args.output_dir, args)

    # The following evaluate the mAP of car poses
    args.gt_dir = args.dataset_dir + 'car_poses'
    det_3d_metric = Detect3DEval(args)
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


if __name__ == '__main__':
    args = parse_args()
    args.output_dir = os.path.join(os.path.dirname(os.path.dirname(args.load_ckpt)), 'test')
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)
    # Wudi hard coded the following range
    if args.list_flag == 'test':
        args.range = [0, 1041]
    elif args.list_flag == 'val':
        args.range = [0, 206]
    elif args.list_flag == 'train':
        args.range = [0, 3888]

    open_3d_vis(args)
