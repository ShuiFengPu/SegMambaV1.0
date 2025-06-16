# mmseg/datasets/npy_dataset.py

import os.path as osp
import numpy as np
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class NpyDataset(BaseSegDataset):
    """
    自定义 NPY 数据集：支持帧级索引
    假设有多个文件，如 train_data_1.npy ～ train_data_5.npy,
    每个文件内部存储形状为 (N, H, W, C) 的图像序列。
    同理，标签文件 train_label_1.npy ～ train_label_5.npy 存储 (N, H, W) 的标签序列。
    Data prefix 定义为：
        img_path='train_data'
        seg_map_path='train_label'
    """

    METAINFO = dict(
        classes=('class_0', 'class_1', 'class_2', 'class_3', 'class_4'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]
    )

    def load_data_list(self):
        data_list = []
        # 遍历每个 npy 文件
        for i in range(1, 6):
            img_file = osp.join(self.data_root,
                                 f"{self.data_prefix['img_path']}_{i}.npy")
            seg_file = osp.join(self.data_root,
                                 f"{self.data_prefix['seg_map_path']}_{i}.npy")
            # 加载以获取帧数 N
            arr = np.load(img_file)
            n_frames = arr.shape[0]
            for fid in range(n_frames):
                data_info = dict(
                    img_path=img_file,
                    seg_map_path=seg_file,
                    frame_idx=fid,
                    seg_fields=[]
                )
                data_list.append(data_info)
        return data_list