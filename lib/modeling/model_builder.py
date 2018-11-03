from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.car_3d_pose_heads as car_3d_pose_heads
from modeling.car_3d_pose_heads import plane_projection_loss
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import numpy as np
from utilities.eval_utils import shape_sim, rot_sim, trans_sim
logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self, ds=None):

        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)

        # BBOX Branch for finer car model classification
        if cfg.MODEL.CAR_CLS_HEAD_ON:
            self.car_cls_Head = get_func(cfg.CAR_CLS.ROI_BOX_HEAD)(self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.car_cls_Outs = car_3d_pose_heads.fast_rcnn_outputs_car_cls_rot(self.car_cls_Head.dim_out)
            self.shape_sim_mat = np.loadtxt('./utilities/sim_mat.txt')
        # TRANS Branch for car translation regression
        if cfg.MODEL.TRANS_HEAD_ON:
            if cfg.TRANS_HEAD.INPUT_CONV_BODY:
                self.car_trans_Head = get_func(cfg.TRANS_HEAD.TRANS_HEAD)(self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale, cfg.TRANS_HEAD.INPUT_DIM)
                self.car_trans_Outs = car_3d_pose_heads.car_trans_outputs(self.car_trans_Head.dim_out)
            elif cfg.TRANS_HEAD.INPUT_TRIPLE_HEAD:
                # We use the 1024 dim from car_cls+rot head
                self.car_trans_Head = get_func(cfg.TRANS_HEAD.TRANS_HEAD)(cfg.TRANS_HEAD.INPUT_DIM)
                self.car_trans_Outs = car_3d_pose_heads.car_trans_triple_outputs(self.car_trans_Head.dim_out, self.car_cls_Head.dim_out)
            else:
                self.car_trans_Head = get_func(cfg.TRANS_HEAD.TRANS_HEAD)(cfg.TRANS_HEAD.INPUT_DIM)
                self.car_trans_Outs = car_3d_pose_heads.car_trans_outputs(self.car_trans_Head.dim_out)
        # 3D to 2D projection error for multi-loss
        if cfg.MODEL.LOSS_3D_2D_ON:
            self.car_models = ds.load_car_models()
            self.car_names = ds.unique_car_names
            self.intrinsic_mat = ds.get_intrinsic_mat()

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

        if cfg.TRAIN.FREEZE_RPN:
            for p in self.RPN.parameters():
                p.requires_grad = False

        if cfg.TRAIN.FREEZE_FPN:
            for p in self.Box_Head.parameters():
                p.requires_grad = False
            for p in self.Box_Outs.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        return_dict = {}  # A dict to collect return variables
        if cfg.FPN.NON_LOCAL:
            blob_conv, f_div_C = self.Conv_Body(im_data)
            if cfg.MODEL.NON_LOCAL_TEST:
                return_dict['f_div_C'] = f_div_C
        else:
            blob_conv = self.Conv_Body(im_data)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
            else:
                box_feat = self.Box_Head(blob_conv, rpn_ret)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
        else:
            # TODO: complete the returns for RPN only situation
            pass

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            # we only use the car cls
            if np.sum(rpn_ret['labels_int32']) <= 0:
                print("ZERO POSITIVE")
            if cfg.MODEL.CAR_CLS_HEAD_ON:
                if getattr(self.car_cls_Head, 'SHARE_RES5', False):
                    # TODO: add thos shared_res5 module
                    pass
                else:
                    car_cls_rot_feat = self.car_cls_Head(blob_conv, rpn_ret)
                    car_cls_score, car_cls, rot_pred = self.car_cls_Outs(car_cls_rot_feat)
                    # car classification loss, we only fine tune the labelled cars

                # we only use the car cls
                care_idx = np.where(list(map(lambda x: x in cfg.TRAIN.CARE_CLS, rpn_ret['labels_int32'])))
                if len(cfg.TRAIN.CE_CAR_CLS_FINETUNE_WIGHT):
                    ce_weight = np.array(cfg.TRAIN.CE_CAR_CLS_FINETUNE_WIGHT)
                else:
                    ce_weight = []

                loss_car_cls, loss_rot, accuracy_car_cls = car_3d_pose_heads.fast_rcnn_car_cls_rot_losses(car_cls_score[care_idx],
                                                                                                          rot_pred[care_idx],
                                                                                                          car_cls[care_idx],
                                                                                                          rpn_ret['car_cls_labels_int32'][care_idx],
                                                                                                          rpn_ret['quaternions'][care_idx],
                                                                                                          ce_weight,
                                                                                                          shape_sim_mat=self.shape_sim_mat)

                return_dict['losses']['loss_car_cls'] = loss_car_cls
                return_dict['losses']['loss_rot'] = loss_rot
                if cfg.CAR_CLS.CLS_LOSS:
                    return_dict['metrics']['accuracy_car_cls'] = accuracy_car_cls
                    return_dict['metrics']['shape_sim'] = shape_sim(car_cls[care_idx].data.cpu().numpy(), self.shape_sim_mat, rpn_ret['car_cls_labels_int32'][care_idx].astype('int64'))
                return_dict['metrics']['rot_diff_degree'] = rot_sim(rot_pred[care_idx].data.cpu().numpy(), rpn_ret['quaternions'][care_idx])

            if cfg.MODEL.TRANS_HEAD_ON:
                pred_boxes = car_3d_pose_heads.bbox_transform_pytorch(rpn_ret['rois'], bbox_pred, im_info,
                                                                      cfg.MODEL.BBOX_REG_WEIGHTS)
                care_idx = np.where(list(map(lambda x: x in cfg.TRAIN.CARE_CLS, rpn_ret['labels_int32'])))

                # Build translation head heres from the bounding box
                if cfg.TRANS_HEAD.INPUT_CONV_BODY:
                    pred_boxes_tmp = [pred_boxes[i, 4 * rpn_ret['labels_int32'][i]:4 * (rpn_ret['labels_int32'][i] + 1)] for i in range(rpn_ret['labels_int32'].shape[0])]
                    pred_boxes_car = torch.stack(pred_boxes_tmp).squeeze(dim=0)
                    car_trans_feat = self.car_trans_Head(blob_conv, rpn_ret, pred_boxes_car)
                    car_trans_pred = self.car_trans_Outs(car_trans_feat)
                    car_trans_pred = car_trans_pred[care_idx]
                elif cfg.TRANS_HEAD.INPUT_TRIPLE_HEAD:
                    pred_boxes_tmp = [pred_boxes[i, 4 * rpn_ret['labels_int32'][i]:4 * (rpn_ret['labels_int32'][i] + 1)] for i in range(rpn_ret['labels_int32'].shape[0])]
                    pred_boxes_car = torch.stack(pred_boxes_tmp).squeeze(dim=0)
                    car_trans_feat = self.car_trans_Head(pred_boxes_car)
                    car_trans_pred = self.car_trans_Outs(car_trans_feat, car_cls_rot_feat)
                    car_trans_pred = car_trans_pred[care_idx]
                else:
                    car_cls_int = 4
                    pred_boxes_car = pred_boxes[care_idx, 4 * car_cls_int:4 * (car_cls_int + 1)].squeeze(dim=0)
                    car_trans_feat = self.car_trans_Head(pred_boxes_car)
                    car_trans_pred = self.car_trans_Outs(car_trans_feat)

                label_trans = rpn_ret['car_trans'][care_idx]
                if cfg.MODEL.Z_MEAN > 0:
                    label_trans = label_trans / 100
                    #label_trans[:, -1] = label_trans[:, -1] - 600
                loss_trans = car_3d_pose_heads.car_trans_losses(car_trans_pred, label_trans)
                return_dict['losses']['loss_trans'] = loss_trans
                return_dict['metrics']['trans_diff_meter'], return_dict['metrics']['trans_thresh_per'] = \
                    trans_sim(car_trans_pred.data.cpu().numpy(), rpn_ret['car_trans'][care_idx],
                              cfg.TRANS_HEAD.TRANS_MEAN, cfg.TRANS_HEAD.TRANS_STD)

            # A 3D to 2D projection loss
            if cfg.MODEL.LOSS_3D_2D_ON:
                # During the mesh generation, using GT(True) or predicted(False) Car ID
                if cfg.LOSS_3D_2D.MESH_GEN_USING_GT:
                    # Acquire car id
                    car_ids = rpn_ret['car_cls_labels_int32'][care_idx].astype('int64')
                else:
                    # Using the predicted car id
                    print("Not properly implemented for pytorch")
                    car_ids = car_cls_score[care_idx].max(dim=1)
                # Get mesh vertices and generate loss
                UV_projection_loss = plane_projection_loss(car_trans_pred, label_trans,
                                                      rot_pred[care_idx], rpn_ret['quaternions'][care_idx],
                                                      car_ids, im_info,
                                                      self.car_models, self.intrinsic_mat, self.car_names)

                return_dict['losses']['UV_projection_loss'] = UV_projection_loss

            if cfg.MODEL.MASK_TRAIN_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                               roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                return_dict['losses']['loss_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    # No corresponding keypoint head implemented yet (Neither in Detectron)
                    # Also, rpn need to generate the label 'roi_has_keypoints_int32'
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret,
                                                  roi_has_keypoints_int32=rpn_ret['roi_has_keypoint_int32'])
                else:
                    kps_feat = self.Keypoint_Head(blob_conv, rpn_ret)
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss
                if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'])
                else:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'],
                        rpn_ret['keypoint_loss_normalizer'])
                return_dict['losses']['loss_kps'] = loss_keypoints

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                if type(v) == np.float64:
                    return_dict['metrics'][k] = v
                else:
                    return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        if cfg.MODEL.NON_LOCAL_TEST:
            blob_conv, f_div_C = self.Conv_Body(data)
        else:
            blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def car_cls_net(self, blob_conv, rpn_blob):
        """For inference"""
        car_cls_feat = self.car_cls_Head(blob_conv, rpn_blob)
        car_cls_score, car_cls, rot_pred = self.car_cls_Outs(car_cls_feat)
        if cfg.TRANS_HEAD.INPUT_TRIPLE_HEAD:
            return car_cls_score, car_cls, rot_pred, car_cls_feat
        else:
            return car_cls_score, car_cls, rot_pred

    @check_inference
    def car_trans_net(self, bbox_pred, im_scale, device_id):
        """For inference"""
        pred_boxes = car_3d_pose_heads.bbox_transform_pytorch_out(bbox_pred, im_scale, device_id)

        # Build translation head heres from the bounding box
        car_trans_feat = self.car_trans_Head(pred_boxes)
        car_trans_pred = self.car_trans_Outs(car_trans_feat)

        return car_trans_pred

    @check_inference
    def car_trans_triple(self, bbox_pred, im_scale, car_cls_feat, device_id):
        """For inference"""
        pred_boxes = car_3d_pose_heads.bbox_transform_pytorch_out(bbox_pred, im_scale, device_id)

        # Build translation head heres from the bounding box
        car_trans_feat = self.car_trans_Head(pred_boxes)
        car_trans_pred = self.car_trans_Outs(car_trans_feat, car_cls_feat)

        return car_trans_pred

    @check_inference
    def car_trans_net_conv_body(self, bbox_pred, im_scale, blob_conv, rpn_blob, device_id):
        """For inference"""
        pred_boxes = car_3d_pose_heads.bbox_transform_pytorch_out(bbox_pred, im_scale, device_id)

        # Build translation head heres from the bounding box
        car_trans_feat = self.car_trans_Head(blob_conv, rpn_blob, pred_boxes)
        car_trans_pred = self.car_trans_Outs(car_trans_feat)

        return car_trans_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
