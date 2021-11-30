import copy

import cv2
import mmcv
import numpy as np
import os.path as osp 
import imgaug.augmenters as iaa
import albumentations as A
# import random

from ..builder import PIPELINES

from numpy import random
from .compose import Compose as AugCompose


@PIPELINES.register_module()
class MixUp(object):
    def __init__(self, p=0.3, lambd=0.5, sub_pipeline = None, keep_extra_img = False):
        self.lambd = lambd
        self.p = p
        self.keep_extra_img = keep_extra_img
        self.sub_pipeline = AugCompose(sub_pipeline)

    def __call__(self, results):
        assert 'extra_aug' in results

        if random.random() < self.p:
            img1, boxes1, labels1 = [
                results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
            ]
            extra_aug = results['extra_aug']
            results_2 = extra_aug[0]
            results_2 = self.sub_pipeline(results_2)
            img2, boxes2, labels2 = [
                results_2[k] for k in ('img', 'gt_bboxes', 'gt_labels')
            ] # img h w c

            img2, ws, hs = mmcv.imresize_like(img2, img1, return_scale=True)
            boxes2[:,0] = boxes2[:,0] * ws
            boxes2[:,2] = boxes2[:,2] * ws
            boxes2[:,1] = boxes2[:,1] * hs
            boxes2[:,3] = boxes2[:,3] * hs

            mixup_image = img1.astype('float32') * self.lambd + img2.astype('float32') * (1 - self.lambd)
            mixup_image = mixup_image.astype('uint8')
            mixup_boxes = np.vstack((boxes1, boxes2))
            mixup_label = np.hstack((labels1,labels2))
            results['img'] = mixup_image
            results['gt_bboxes'] = mixup_boxes
            results['gt_labels'] = mixup_label

            if self.keep_extra_img:
                return results
            else:
                results['extra_aug'] = extra_aug[1:]
                return results
        else:
            if self.keep_extra_img:
                return results
            else:
                extra_aug = results['extra_aug']
                results['extra_aug'] = extra_aug[1:]
                return results


@PIPELINES.register_module()
class CopyPaste(object):
    def __init__(self, sub_pipeline = None, p=0.5, keep_extra_img=False):
        self.p = p
        self.sub_pipeline = AugCompose(sub_pipeline)
        self.keep_extra_img = keep_extra_img

    def __call__(self, results):
        assert 'extra_aug' in results

        if random.random() < self.p:
            extra_aug = results['extra_aug']
            results_1 = extra_aug[0]
            results_1 = self.sub_pipeline(results_1)
            img1, boxes1, labels1 = [
                results_1[k] for k in ('img', 'gt_bboxes', 'gt_labels')
            ] # img h w c

            # handle the bbox
            patchlist = []
            hweps = []
            prs = [random.randint(15) for i in range(4)] # 15 pixels
            for box in boxes1:
                y1, y2 = max(int(box[1])-prs[0], 0), min(int(box[3])+prs[1], img1.shape[1])
                x1, x2 = max(int(box[0])-prs[2], 0), min(int(box[2])+prs[3], img1.shape[0])
                patchlist.append(img1[y1:y2, x1:x2])
                hweps.append((box[0]-x1, box[1]-y1))

            rh, rw, _ = results['img'].shape
            for i in range(len(boxes1)):
                box = boxes1[i]
                cnt = 30
                gt_boxes = results['gt_bboxes']
                while(cnt > 0):
                    w, h = box[2] - box[0], box[3] - box[1]
                    if rw - w < 32 or rh - h < 32:
                        break
                    x, y = random.randint(int(rw - w - 30)), random.randint(int(rh - h - 30))
                    newbox = np.array([x, y, x + w, y + h])
                    inter = False
                    for gt_box in gt_boxes:
                        if self._intersect(newbox, gt_box):
                            inter = True
                            break
                    if not inter:
                        box = newbox
                        sh, sw, _ = patchlist[i].shape
                        results['img'][int(box[1]):int(box[1])+sh, int(box[0]):int(box[0])+sw] = patchlist[i]
                        box = np.array([x+hweps[i][0], y+hweps[i][1], x + w + hweps[i][0], y + h + hweps[i][1]]).astype(np.float32)
                        results['gt_bboxes'] = np.vstack((results['gt_bboxes'], box))
                        results['gt_labels'] = np.hstack((results['gt_labels'], labels1[i]))
                        break
                    else:
                        cnt -= 1

            if self.keep_extra_img:
                return results
            else:
                results['extra_aug'] = extra_aug[1:]
                return results
        else:
            if self.keep_extra_img:
                return results
            else:
                extra_aug = results['extra_aug']
                results['extra_aug'] = extra_aug[1:]
                return results

    def _intersect(self, bbox1, bbox2):
        cx1, cy1, w1, h1 = self._convert(bbox1)
        cx2, cy2, w2, h2 = self._convert(bbox2)
        if np.abs(cx1 - cx2) + np.abs(cy1 - cy2) > (w1 + w2 + h1 + h2) / 2:
            return False
        else:
            return True
    
    def _convert(self, bbox): # [x1,y1,x2,y2] -> [cx, cy, w, h]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return cx, cy, w, h


@PIPELINES.register_module()
class MyCutOut(object):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 p = None):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.p = p
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, results):
        """Call function to drop some regions of image."""
        if random.random() > self.p:    # fix bug
            return results 

        h, w, c = results['img'].shape
        img = np.copy(results['img'])
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            img[y1:y2, x1:x2, :] = self.fill_in
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str


class AlbuImgOps:
    def __call__(self, results):
        # note mmdet using BGR before normalization
        img = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        img = self.f(image=img)["image"]
        results['img'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return results


@PIPELINES.register_module()
class Spatter:
    def __init__(self,
                severity=[2,3]):
        assert isinstance(severity, (list, int))
        self.f = iaa.imgcorruptlike.Spatter(severity=severity)

    def __call__(self, results):
        img = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        img = self.f(image=img)
        results['img'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return results


@PIPELINES.register_module()
class ChannelDropout(AlbuImgOps):
    def __init__(self):
        self.f = A.ChannelDropout(always_apply=True)


@PIPELINES.register_module()
class ChannelShuffle(AlbuImgOps):
    def __init__(self):
        self.f = A.ChannelShuffle(always_apply=True)


@PIPELINES.register_module()
class ColorJitter(AlbuImgOps):
    def __init__(self,
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.2):
        self.f = A.ColorJitter(
            always_apply=True,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation, 
            hue=hue)

@PIPELINES.register_module()
class ToGray(AlbuImgOps):
    def __init__(self):
        self.f = A.ToGray(always_apply=True)


@PIPELINES.register_module()
class ToSepia(AlbuImgOps):
    def __init__(self):
        self.f = A.ToSepia(always_apply=True)


@PIPELINES.register_module()
class FDA:
    def __init__(self,
                beta_limit=0.1,
                keep_extra_img=False):
        self.beta_limit = beta_limit
        self.keep_extra_img = keep_extra_img

    def __call__(self, results):
        assert 'extra_aug' in results

        extra_aug = results['extra_aug']
        refer_img = osp.join(extra_aug[0]['img_prefix'],
                extra_aug[0]['img_info']['filename'])
        fda = A.FDA(
            always_apply=True,
            beta_limit=self.beta_limit,
            reference_images=[refer_img])

        img = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
        img = fda(image=img)["image"]
        results['img'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.keep_extra_img:
            return results
        else:
            results['extra_aug'] = extra_aug[1:]
            return results
