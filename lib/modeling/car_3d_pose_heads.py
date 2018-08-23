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
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)

        rot_pred = self.rot_pred(x)
        return cls_score, rot_pred


def fast_rcnn_car_cls_rot_losses(cls_score, rot_pred, label_int32, quaternions,
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
