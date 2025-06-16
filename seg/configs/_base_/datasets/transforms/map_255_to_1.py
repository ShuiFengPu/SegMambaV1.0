from mmseg.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class Map255To1:
    def __call__(self, results):
        gt = results['gt_seg_map']
        results['gt_seg_map'] = (gt == 255).astype(np.uint8)
        return results
