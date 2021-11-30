try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

from numpy import random
import numpy as np 

from ..builder import PIPELINES


@PIPELINES.register_module()
class MyCorrupt:
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.
    """

    def __init__(self, settings = None, severity=[2,3,4]):
        # self.corruption = corruption
        self.severity = severity
        # self.p = p
        self.p = 1.0

        self.ccs = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
        if settings is not None:
            self.settings = settings
        else:
            self.settings = {'gaussian_noise': 0.03, 'shot_noise': 0.03, 'impulse_noise': 0.03, 'speckle_noise': 0.03,
                'gaussian_blur': 0.04, 'defocus_blur': 0.04,
                'motion_blur': 0.05,
                'snow': 0.03, 'frost': 0.05, 'elastic_transform': 0.05,
                'pixelate': 0.01, 'jpeg_compression': 0.01}
            # 'glass_blur': 0.05, 'zoom_blur': 0.05
        self.ckey = list(self.settings)
        self.cp = [self.settings[key] for key in self.settings]
        self.cp = self.cp / np.sum(self.cp)

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'

        if random.random() < self.p:
            s = random.choice(self.severity)
            index = random.choice(len(self.ckey), p = self.cp)
            cname = self.ckey[index]
            results['img'] = corrupt(
                results['img'].astype(np.uint8),
                corruption_name=cname,
                severity=s)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
