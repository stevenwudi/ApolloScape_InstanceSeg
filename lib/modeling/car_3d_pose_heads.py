import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import numpy as np


class fast_rcnn_outputs_car_cls_rot(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUMBER_CARS)
        if cfg.CAR_CLS.CLS_SPECIFIC_ROT:
            self.rot_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUMBER_CARS)
        else:
            self.rot_pred = nn.Linear(dim_in, 4)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.rot_pred.weight, std=0.001)
        init.constant_(self.rot_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'rot_pred.weight': 'rot_pred',
            'rot_pred.bias': 'rot_pred'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        #if not self.training:
        cls = F.softmax(cls_score, dim=1)

        rot_pred = self.rot_pred(x)
        return cls_score, cls, rot_pred


def fast_rcnn_car_cls_rot_losses(cls_score, rot_pred, car_cls, label_int32, quaternions,
                                 ce_weight=None, shape_sim_mat_loss_mat=None):
    # For car classification loss, we only have classification losses
    # Or should we use sim_mat?
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)

    if len(shape_sim_mat_loss_mat):
        if len(ce_weight):
            coeff = shape_sim_mat_loss_mat * ce_weight
        else:
            coeff = shape_sim_mat_loss_mat

        loss_cls = Variable(torch.from_numpy(np.array(0)).float()).cuda(device_id)
        for i in range(len(cls_score)):
            coeff_car = Variable(torch.from_numpy(np.array(coeff[i])).float()).cuda(device_id)
            loss_cls += F.cross_entropy(cls_score[i].unsqueeze(0), rois_label[i].unsqueeze(0), coeff_car)
        loss_cls /= len(cls_score)
    else:
        if len(ce_weight):
            ce_weight = Variable(torch.from_numpy(np.array(ce_weight)).float()).cuda(device_id)
            loss_cls = F.cross_entropy(cls_score, rois_label, ce_weight)
        else:
            loss_cls = F.cross_entropy(cls_score, rois_label)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

    # loss rot
    quaternions = Variable(torch.from_numpy(quaternions.astype('float32'))).cuda(device_id)
    loss_rot = torch.abs(rot_pred - quaternions)
    N = loss_rot.size(0)  # batch size
    loss_rot = loss_rot.view(-1).sum(0) / N
    return loss_cls, loss_rot, accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #
class roi_car_cls_rot_head(nn.Module):
    """Add a ReLU MLP with two hidden layers.2048 -- 1024"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.CAR_CLS.MLP_HEAD_DIM

        roi_size = cfg.CAR_CLS.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.CAR_CLS.ROI_XFORM_METHOD,
            resolution=cfg.CAR_CLS.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.CAR_CLS.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


# ---------------------------------------------------------------------------- #
# TRANS heads
# ---------------------------------------------------------------------------- #
def bbox_transform_pytorch(rois, deltas, im_info, weights=(1.0, 1.0, 1.0, 1.0), ):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    This is a pytorch head
    """

    device_id = deltas.get_device()

    boxes = Variable(torch.from_numpy(rois[:, 1:].astype('float32'))).cuda(device_id)
    weights = Variable(torch.from_numpy(np.array(weights).astype('float32'))).cuda(device_id)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    bb_xform_clip = torch.from_numpy(np.array(cfg.BBOX_XFORM_CLIP).astype('float32')).cuda(device_id)
    dw = torch.min(dw, bb_xform_clip)
    dh = torch.min(dh, bb_xform_clip)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_w = torch.exp(dw) * widths[:, np.newaxis]
    pred_h = torch.exp(dh) * heights[:, np.newaxis]

    pred_boxes = torch.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # # y1
    # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    # # x1
    pred_boxes[:, 0::4] = pred_ctr_x
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y
    # w (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_w
    # h (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_h

    # Normalise box: NOT DONE properly yet! Hard coded

    im_shape = im_info[0][:2]
    car_shape = (120, 120)
    pred_boxes[:, 0::4] -= (im_shape[1]/2)
    pred_boxes[:, 0::4] /= im_shape[1]
    pred_boxes[:, 1::4] -= (im_shape[0]/2)
    pred_boxes[:, 1::4] /= im_shape[0]

    pred_boxes[:, 2::4] -= (car_shape[0]/2)
    pred_boxes[:, 2::4] /= car_shape[0]
    pred_boxes[:, 3::4] -= (car_shape[1]/2)
    pred_boxes[:, 3::4] /= car_shape[1]

    pred_boxes = pred_boxes.cuda(device_id)
    return pred_boxes


class bbox_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = hidden_dim = cfg.TRANS_HEAD.MLP_HEAD_DIM

        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x


class car_trans_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.trans_pred = nn.Linear(dim_in, cfg.TRANS_HEAD.OUTPUT_DIM)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.trans_pred.weight, std=0.1)
        init.constant_(self.trans_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'trans_pred.weight': 'trans_pred_w',
            'trans_pred.bias': 'trans_pred_b',
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        trans_pred = self.trans_pred(x)
        return trans_pred


def car_trans_losses(trans_pred, label_trans):
    # For car classification loss, we only have classification losses
    # Or should we use sim_mat?
    device_id = trans_pred.get_device()
    label_trans = Variable(torch.from_numpy(label_trans.astype('float32'))).cuda(device_id)

    # loss rot
    N = trans_pred.shape[0]
    if cfg.TRANS_HEAD.LOSS == 'MSE':
        loss = nn.MSELoss()
        loss_trans = loss(trans_pred, label_trans)
    elif cfg.TRANS_HEAD.LOSS == 'L1':
        loss_trans = torch.abs(trans_pred - label_trans)
    loss_trans = loss_trans.view(-1).sum(0) / N
    return loss_trans


def infer_car_3d_translation(pred_boxes_car, car_model, quaternions_gt, quaternions_dt, rpn_ret):
    from matplotlib import pyplot as plt
    import cv2
    image_file = '/media/SSD_1TB/ApolloScape/ECCV2018_apollo/train/images/180310_025828603_Camera_5.jpg'
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    intrinsic = np.array(
                [2304.54786556982, 2305.875668062,
                 1686.23787612802, 1354.98486439791]),
    image, intrinsic_mat = self.rescale(image, intrinsic)
    im_shape = image.shape
    mask_all = np.zeros(im_shape)

    for i, car_pose in enumerate(car_poses):
        car_name = car_models.car_id2name[car_pose['car_id']].name
        mask = self.render_car(car_pose['pose'], car_name, im_shape)
        mask_all += mask

    return image