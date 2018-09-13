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
    parser.add_argument('--dataset_dir', default=r'E:\Thunder\train\\')
    parser.add_argument('--output_dir', default=r'D:\Github\ApolloScape_InstanceSeg\Outputs\e2e_3d_car_101_FPN_triple_head\Sep09-23-42-21_N606-TITAN32_step')
    parser.add_argument('--list_flag', default='val', help='Choosing between [val, test]')
    parser.add_argument('--iou_ignore_threshold', default=1.0, help='Filter out by this iou')
    parser.add_argument('--vis_num', default=160, help='Choosing which image to view')
    parser.add_argument('--criterion_num', default=0, help='[0,1,2,...9]')
    parser.add_argument('--dtScores', default=0.9, help='Detection Score for visualisation')
    return parser.parse_args()


def open_3d_vis(args, output_dir):
    json_dir = os.path.join(output_dir, 'json_' + args.list_flag + '_trans') + '_iou' + str(1.0)

    args.test_dir = json_dir
    args.gt_dir = args.dataset_dir + 'car_poses'
    args.res_file = os.path.join(output_dir, 'json_' + args.list_flag + '_res.txt')
    args.simType = None

    det_3d_metric = Detect3DEval(args)
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Wudi hard coded the following range
    if args.list_flag == 'test':
        args.range = [0, 1041]
    elif args.list_flag == 'val':
        args.range = [0, 206]
    elif args.list_flag == 'train':
        args.range = [0, 3888]

    open_3d_vis(args, output_dir=args.output_dir)
