# mmseg/datasets/transforms/debug_print.py

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DebugPrint(BaseTransform):
    """调试 transform：打印图像与标签的 shape 与唯一值"""
    def transform(self, results):
        print(f"[Debug] img shape: {results['img'].shape}, dtype: {results['img'].dtype}")
        if 'gt_seg_map' in results:
            print(f"[Debug] gt_seg_map unique: {np.unique(results['gt_seg_map'])}")
        return results
