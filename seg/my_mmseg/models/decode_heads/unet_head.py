from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class UNetHead(BaseDecodeHead):
    def __init__(self, num_convs=2, dilations=(1,), **kwargs):
        super(UNetHead, self).__init__(**kwargs)
        self.dilations = dilations

        convs = []
        in_channels = self.in_channels
        for i in range(num_convs):
            dilation = self.dilations[i] if i < len(self.dilations) else 1
            padding = dilation
            convs.append(
                nn.Conv2d(
                    in_channels if i == 0 else self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=padding,
                    dilation=dilation
                )
            )
            convs.append(nn.BatchNorm2d(self.channels))
            convs.append(nn.ReLU(inplace=True))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout2d(self.dropout_ratio)
        self.cls_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        

        try:
            x = self.convs(x)
            x = self.dropout(x)
            output = self.cls_seg(x)
        except Exception as e:
            print("Forward error:", e)
            raise
        return output

