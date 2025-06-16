from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class KvasirDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'polyp'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.jpg', **kwargs)
