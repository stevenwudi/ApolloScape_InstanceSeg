"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', default='ApolloScape', help='Dataset to use')
    parser.add_argument('--cfg', dest='cfg_file', default='./configs/e2e_mask_rcnn_R-101-FPN_2x.yaml', help='Config file for training (and optionally testing)')
    parser.add_argument('--load_ckpt', default='/media/samsumg_1tb/stevenwudi/stevenwudi/PycharmProjects/CVPR_2018_WAD/Outputs/e2e_mask_rcnn_R-101-FPN_2x/Jun13-15-31-20_n606_step/ckpt/model_step29999.pth', help='checkpoint path to load')
    parser.add_argument('--load_detectron', help='path to the detectron weight pickle file')
    parser.add_argument('--output_dir', help='output directory to save the testing results. If not provided defaults to [args.load_ckpt|args.load_detectron]/../test.')
    parser.add_argument('--set', dest='set_cfgs', help='set config keys, will overwrite config in the cfg_file. See lib/core/config.py for all options', default=[], nargs='*')
    # val: [0, 8327], test: [0, 6597]
    parser.add_argument('--range', default=[0, 8327], help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
    parser.add_argument('--multi-gpu-testing', help='using multiple gpus for inference', default=False, action='store_true')
    parser.add_argument('--vis', default=False,  dest='vis', help='visualize detections', action='store_true')
    parser.add_argument('--list_flag', default='val', help='Choosing between [val, test]')
    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), 'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    # Manually change the following:
    cfg.TEST.DATASETS = ['ApolloScape',]
    cfg.MODEL.NUM_CLASSES = 8
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)

"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=all                  | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50      | area=all                  | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.75      | area=all                  | maxDets=100 ] = 0.330
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=extra-small (0-14)   | maxDets=100 ] = 0.043 | (numGT, numDt) = 28858 163896
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium(28-56)        | maxDets=100 ] = 0.390 | (numGT, numDt) = 38493 134063
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=large(56-112)        | maxDets=100 ] = 0.478 | (numGT, numDt) = 21262 51016
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=extra-large(112-512) | maxDets=100 ] = 0.533 | (numGT, numDt) = 14916 24344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=uber-large(512 !!!!) | maxDets=100 ] = 0.820 | (numGT, numDt) =  1254  3050
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=all                  | maxDets=  1 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=all                  | maxDets= 10 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=all                  | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=extra-small (0-14)   | maxDets=100 ] = 0.139 | (numGT, numDt) = 28858 163896
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium(28-56)        | maxDets=100 ] = 0.511 | (numGT, numDt) = 38493 134063
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=large(56-112)        | maxDets=100 ] = 0.602 | (numGT, numDt) = 21262 51016
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=extra-large(112-512) | maxDets=100 ] = 0.677 | (numGT, numDt) = 14916 24344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=uber-large(512 !!!!) | maxDets=100 ] = 0.869 | (numGT, numDt) =  1254  3050
"""