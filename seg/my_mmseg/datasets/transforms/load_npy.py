# mmseg/datasets/transforms/load_npy.py

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromNpy(BaseTransform):
    """从 .npy 文件加载单帧图像，根据 frame_idx 切片"""
    def transform(self, results):
        img_path = results['img_path']
        frame_idx = results.get('frame_idx', None)
        arr = np.load(img_path)
        # 如果存在 frame_idx，则选取对应帧，否则认为是单张 HWC
        if frame_idx is not None:
            img = arr[frame_idx]
        else:
            img = arr
        img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


@TRANSFORMS.register_module()
class LoadAnnotationsFromNpy(BaseTransform):
    """从 .npy 文件加载单帧标签，根据 frame_idx 切片，并映射无效值"""
    def __init__(self, num_valid_classes=9, ignore_index=255):
        self.num_valid_classes = num_valid_classes
        self.ignore_index = ignore_index

    def transform(self, results):
        seg_path = results['seg_map_path']
        frame_idx = results.get('frame_idx', None)
        arr = np.load(seg_path)
        if frame_idx is not None:
            seg_map = arr[frame_idx]
        else:
            seg_map = arr
        seg_map = seg_map.astype(np.int32)
        # 映射无效标签
        seg_map[seg_map >= self.num_valid_classes] = self.ignore_index

        results['gt_seg_map'] = seg_map
        results['seg_fields'] = ['gt_seg_map']
        return results