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


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', default='ApolloScape', help='Dataset to use')
    parser.add_argument('--dataset_dir', default='/media/samsumg_1tb/ApolloScape/ECCV2018_apollo/train/')
    parser.add_argument('--test_dir', default='/media/samsumg_1tb/ApolloScape/ApolloScape_InstanceSeg/e2e_3d_car_101_FPN_triple_head_non_local_weighted/Nov03-21-05-13_N606-TITAN32_step/test/json_val_trans_iou_0.5_BBOX_AUG_multiple_scale_CAR_CLS_AUG_multiple_scale')
    parser.add_argument('--list_flag', default='val', help='Choosing between [val, test]')
    parser.add_argument('--iou_ignore_threshold', default=1.0, help='Filter out by this iou')
    parser.add_argument('--simType', default=None, help='Detection Score for visualisation')
    parser.add_argument('--dtScores', default=0.1, help='Detection Score for visualisation')
    return parser.parse_args()


def open_3d_vis(args):
    # The following evaluate the detection result from Faster-RCNN Head
    #results = task_evaluation.evaluate_all(dataset, all_boxes, all_segms, all_keyps, output_dir, args)

    # The following evaluate the mAP of car poses
    args.gt_dir = args.dataset_dir + 'car_poses'
    det_3d_metric = Detect3DEval(args)
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


if __name__ == '__main__':
    args = parse_args()
    # Wudi hard coded the following range
    if args.list_flag == 'test':
        args.range = [0, 1041]
    elif args.list_flag == 'val':
        args.range = [0, 206]
    elif args.list_flag == 'train':
        args.range = [0, 3888]

    open_3d_vis(args)
