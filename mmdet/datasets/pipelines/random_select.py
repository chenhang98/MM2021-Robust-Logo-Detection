import collections
from logging import warn, warning

import numpy as np
import warnings
import random
from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomSelect:
    """Random select one of multiple transforms.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be selected. The probability of each transform object
            being selected is specified by p.
    """

    def __init__(self, transforms, p=None):
        assert isinstance(transforms, collections.abc.Sequence)

        self.p = p
        self.ps = [_.pop('p') for _ in transforms]
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

        p = sum(self.ps)
        if p < 1:
            self.transforms.append(None)
            self.ps.append(1 - p)
            warnings.warn('Add identity transform because the sum of probabilities is less than 1', UserWarning)
        elif p > 1:
            raise Exception('Sum of probabilities is larger than 1')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        if data.get('disable_RandomSelect', False):
            return data
        if random.random() > self.p:
            return data
        N = len(self.transforms)
        i = np.random.choice(N, 1, p=self.ps)[0]
        t = self.transforms[i]
        if t is not None:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
