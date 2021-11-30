import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
from mmdet.datasets.custom import CustomDataset
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class OCR2FCBBoxHead(ConvFCBBoxHead):
    CLASSES = CustomDataset.get_classes('sample_test/labelList.txt')

    def __init__(self, fc_out_channels=1024, 
                ocr_channels=36,
                loss_ocr=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                *args, **kwargs):
        super(OCR2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.loss_ocr = build_loss(loss_ocr)
        self.fc_ocr = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=ocr_channels)
        # class -> indices in alphabet [a, b, ..., 0, 1, .., 9]
        self.cls2inds = []
        oa, oz, o0, o9 = ord('a'), ord('z'), ord('0'), ord('9')
        for name in self.CLASSES:
            inds = []
            for c in name.lower():
                oc = ord(c)
                if oa <= oc <= oz:
                    inds.append(oc - oa)
                elif o0 <= oc <= o9:
                    inds.append(oc - o0)
            self.cls2inds.append(inds)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        ocr_score = self.fc_ocr(x_cls) if self.with_cls else None
        self.ocr_score = ocr_score

        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = super().loss(cls_score,
                            bbox_pred,
                            rois,
                            labels,
                            label_weights,
                            bbox_targets,
                            bbox_weights,
                            reduction_override)
        # cal OCR loss
        ocr_score = self.ocr_score
        ocr_target = ocr_score.new_zeros(ocr_score.size(), dtype=torch.long)
        for i, c in enumerate(labels):
            if c < self.num_classes:
                inds = self.cls2inds[c]
                ocr_target[i, inds] = 1
        losses['loss_ocr'] = self.loss_ocr(ocr_score, ocr_target)
        return losses