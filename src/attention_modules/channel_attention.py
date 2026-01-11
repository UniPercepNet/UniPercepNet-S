import torch.nn as nn
from mmcv.cnn import ConvModule

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels // reduction_rate,
                kernel_size=1,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=channels // reduction_rate,
                out_channels=channels,
                kernel_size=1,
                bias=True,
                norm_cfg=None,
                act_cfg=None
            )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)

        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)

        attention = self.sigmoid(avg_out + max_out)

        return attention * x
