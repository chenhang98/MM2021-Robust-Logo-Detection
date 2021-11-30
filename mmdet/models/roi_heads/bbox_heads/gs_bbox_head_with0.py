import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from mmdet.core import multiclass_nms
from mmcv.runner import force_fp32
from .convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class GSBBoxHeadWith0(Shared2FCBBoxHead):

    def __init__(self,
                 fc_out_channels=1024,
                 gs_config=None,
                 *args,
                 **kwargs):
        super(GSBBoxHeadWith0, self).__init__(
                fc_out_channels=fc_out_channels,
                *args, **kwargs)
        self.fc_cls = nn.Linear(self.cls_last_dim,
                                self.num_classes + gs_config.num_bins)

        # self.loss_bg = build_loss(gs_config.loss_bg)

        self.loss_bins = []
        for i in range(gs_config.num_bins):
            self.loss_bins.append(build_loss(gs_config.loss_bin))

        # self.label2binlabel = torch.load(gs_config.label2binlabel).cuda()
        # self.pred_slice = torch.load(gs_config.pred_slice).cuda()
        with open(gs_config.label2binlabel, 'rb') as f:
            self.label2binlabel = torch.from_numpy(
                pickle.load(f)
            ).cuda()
        with open(gs_config.pred_slice, 'rb') as f:
            self.pred_slice = torch.from_numpy(
                pickle.load(f)
            ).cuda()

        # TODO: update this ugly implementation. Save fg_split to a list and
        #  load groups by gs_config.num_bins
        with open(gs_config.fg_split, 'rb') as fin:
            fg_split = pickle.load(fin)

        self.fg_splits = []
        # self.fg_splits.append(torch.from_numpy(fg_split['(0, 10)']).cuda())
        # self.fg_splits.append(torch.from_numpy(fg_split['[10, 100)']).cuda())
        # self.fg_splits.append(torch.from_numpy(fg_split['[100, 1000)']).cuda())
        # self.fg_splits.append(torch.from_numpy(fg_split['[1000, ~)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['(0, 1e2)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['[1e2, 1e3)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['[1e3, 1e4)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['[1e4, ~)']).cuda())

        self.others_sample_ratio = gs_config.others_sample_ratio


    def _sample_others(self, label):

        # only works for non bg-fg bins

        fg = torch.where(label > 0, torch.ones_like(label),
                         torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = 1 - fg
        bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.others_sample_ratio)

        if bg_sample_num >= bg_num:
            weight = torch.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.cpu().numpy(),
                                          (bg_sample_num, ), replace=False)
            sample_idx = torch.from_numpy(sample_idx).cuda()
            fg[sample_idx] = 1
            weight = fg

        return weight

    def _remap_labels(self, labels_):
        # BAGS: mmdet now use self.num_clases represent bg
        labels = labels_.clone()
        bg = (labels == self.num_classes)
        labels += 1
        labels[bg] = 0

        num_bins = self.label2binlabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i]
            new_bin_label = mapping[labels]

            if i < 1:
                weight = torch.ones_like(new_bin_label)
                # weight = torch.zeros_like(new_bin_label)
            else:
                weight = self._sample_others(new_bin_label)
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _slice_preds(self, cls_score):

        new_preds = []

        num_bins = self.pred_slice.shape[0]
        for i in range(num_bins):
            start = self.pred_slice[i, 0]
            length = self.pred_slice[i, 1]
            sliced_pred = cls_score.narrow(1, start, length)
            new_preds.append(sliced_pred)

        return new_preds

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
        losses = dict()

        if cls_score is not None:
            # Original label_weights is 1 for each roi.
            new_labels, new_weights, new_avgfactors = self._remap_labels(labels)
            new_preds = self._slice_preds(cls_score)

            num_bins = len(new_labels)
            for i in range(num_bins):
                losses['loss_cls_bin{}'.format(i)] = self.loss_bins[i](
                    new_preds[i],
                    new_labels[i],
                    new_weights[i],
                    avg_factor=new_avgfactors[i],
                    reduction_override=reduction_override
                )

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score'))
    def _merge_score(self, cls_score):
        '''
        Do softmax in each bin. Decay the score of normal classes
        with the score of fg.
        From v1.
        '''
        batch_inference = False
        if cls_score.dim() == 3:
            cls_score.squeeze_(0)
            batch_inference = True

        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)    # split into each bins
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, self.num_classes+1)).cuda()
        merge = torch.zeros((num_proposals, self.num_classes+1)).cuda()

        # import pdb
        # pdb.set_trace()
        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i][:, 1:]

        weight = bg_score.narrow(1, 1, 1)

        # Whether we should add this? Test
        fg_merge = weight * fg_merge

        # merge[:, 0] = bg_score[:, 0]
        # merge[:, 1:] = fg_merge[:, 1:]

        # BAGS: mmdet has moved bg to the last dim 
        merge[:, :self.num_classes] = fg_merge[:, 1:]
        merge[:, self.num_classes] = bg_score[:, 0]

        # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        # erge[fg_idx] = fg_merge[fg_idx]
        if batch_inference:
            return merge.unsqueeze(0)
        else:
            return merge


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    # def get_det_bboxes(self,
    def get_bboxes(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None):
        # BAGS: F.softmax -> self._merge_score
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        # if self.custom_cls_channels:
        #     scores = self.loss_cls.get_activation(cls_score)
        # else:
        #     scores = F.softmax(
        #         cls_score, dim=-1) if cls_score is not None else None
        assert not self.custom_cls_channels
        scores = self._merge_score(cls_score)

        batch_mode = True
        if rois.ndim == 2:
            # e.g. AugTest, Cascade R-CNN, HTC, SCNet...
            batch_mode = False

            # add batch dimension
            if scores is not None:
                scores = scores.unsqueeze(0)
            if bbox_pred is not None:
                bbox_pred = bbox_pred.unsqueeze(0)
            rois = rois.unsqueeze(0)

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        if rescale and bboxes.size(-2) > 0:
            if not isinstance(scale_factor, tuple):
                scale_factor = tuple([scale_factor])
            # B, 1, bboxes.size(-1)
            scale_factor = bboxes.new_tensor(scale_factor).unsqueeze(1).repeat(
                1, 1,
                bboxes.size(-1) // 4)
            bboxes /= scale_factor

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_size = scores.shape[0]
            # ignore background class
            scores = scores[..., :self.num_classes]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', cfg.max_per_img)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels
        det_bboxes = []
        det_labels = []
        for (bbox, score) in zip(bboxes, scores):
            if cfg is not None:
                det_bbox, det_label = multiclass_nms(bbox, score,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img)
            else:
                det_bbox, det_label = bbox, score
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if not batch_mode:
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
        return det_bboxes, det_labels

    # BAGS: not used codes
    # def _remap_labels1(self, labels):

    #     num_bins = self.label2binlabel.shape[0]
    #     new_labels = []
    #     new_weights = []
    #     new_avg = []
    #     for i in range(num_bins):
    #         mapping = self.label2binlabel[i]
    #         new_bin_label = mapping[labels]

    #         weight = torch.ones_like(new_bin_label)

    #         new_labels.append(new_bin_label)
    #         new_weights.append(weight)

    #         avg_factor = max(torch.sum(weight).float().item(), 1.)
    #         new_avg.append(avg_factor)

    #     return new_labels, new_weights, new_avg

    # @force_fp32(apply_to=('cls_score'))
    # def _merge_score1(self, cls_score):
    #     '''
    #     Do softmax in each bin. Merge the scores directly.
    #     '''
    #     num_proposals = cls_score.shape[0]

    #     new_preds = self._slice_preds(cls_score)
    #     new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

    #     bg_score = new_scores[0]
    #     fg_score = new_scores[1:]

    #     fg_merge = torch.zeros((num_proposals, 1231)).cuda()
    #     merge = torch.zeros((num_proposals, 1231)).cuda()

    #     for i, split in enumerate(self.fg_splits):
    #         fg_merge[:, split] = fg_score[i]

    #     merge[:, 0] = bg_score[:, 0]
    #     fg_idx = (bg_score[:,1] > 0.5).nonzero(as_tuple=True)[0]
    #     merge[fg_idx] = fg_merge[fg_idx]

    #     return merge

    # @force_fp32(apply_to=('cls_score'))
    # def _merge_score2(self, cls_score):
    #     '''
    #     Do softmax in each bin. Softmax again after merge.
    #     '''
    #     num_proposals = cls_score.shape[0]

    #     new_preds = self._slice_preds(cls_score)
    #     new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

    #     bg_score = new_scores[0]
    #     fg_score = new_scores[1:]

    #     fg_merge = torch.zeros((num_proposals, 1231)).cuda()
    #     merge = torch.zeros((num_proposals, 1231)).cuda()

    #     for i, split in enumerate(self.fg_splits):
    #         fg_merge[:, split] = fg_score[i]

    #     merge[:, 0] = bg_score[:, 0]
    #     fg_idx = (bg_score[:,1] > 0.5).nonzero(as_tuple=True)[0]
    #     merge[fg_idx] = fg_merge[fg_idx]
    #     merge = F.softmax(merge)

    #     return merge

    # @force_fp32(apply_to=('cls_score'))
    # def _merge_score4(self, cls_score):
    #     '''
    #     Do softmax in each bin.
    #     Do softmax on merged fg classes.
    #     Decay the score of normal classes with the score of fg.
    #     From v2 and v3
    #     '''
    #     num_proposals = cls_score.shape[0]

    #     new_preds = self._slice_preds(cls_score)
    #     new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

    #     bg_score = new_scores[0]
    #     fg_score = new_scores[1:]

    #     fg_merge = torch.zeros((num_proposals, 1231)).cuda()
    #     merge = torch.zeros((num_proposals, 1231)).cuda()

    #     for i, split in enumerate(self.fg_splits):
    #         fg_merge[:, split] = fg_score[i]

    #     fg_merge = F.softmax(fg_merge, dim=1)
    #     weight = bg_score.narrow(1, 1, 1)
    #     fg_merge = weight * fg_merge

    #     merge[:, 0] = bg_score[:, 0]
    #     merge[:, 1:] = fg_merge[:, 1:]
    #     # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
    #     # erge[fg_idx] = fg_merge[fg_idx]

    #     return merge

    # @force_fp32(apply_to=('cls_score'))
    # def _merge_score5(self, cls_score):
    #     '''
    #     Do softmax in each bin.
    #     Pick the bin with the max score for each box.
    #     '''
    #     num_proposals = cls_score.shape[0]

    #     new_preds = self._slice_preds(cls_score)
    #     new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

    #     bg_score = new_scores[0]
    #     fg_score = new_scores[1:]
    #     max_scores = [s.max(dim=1, keepdim=True)[0] for s in fg_score]
    #     max_scores = torch.cat(max_scores, 1)
    #     max_idx = max_scores.argmax(dim=1)

    #     fg_merge = torch.zeros((num_proposals, 1231)).cuda()
    #     merge = torch.zeros((num_proposals, 1231)).cuda()

    #     for i, split in enumerate(self.fg_splits):
    #         tmp_merge = torch.zeros((num_proposals, 1231)).cuda()
    #         tmp_merge[:, split] = fg_score[i]
    #         roi_idx = torch.where(max_idx == i,
    #                               torch.ones_like(max_idx),
    #                               torch.zeros_like(max_idx)).nonzero(
    #             as_tuple=True)[0]
    #         fg_merge[roi_idx] = tmp_merge[roi_idx]

    #     merge[:, 0] = bg_score[:, 0]
    #     fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
    #     merge[fg_idx] = fg_merge[fg_idx]

    #     return merge
