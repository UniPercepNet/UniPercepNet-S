from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init

class SemanticSegmentationBranch(nn.Module):
    def __init__(
        self, 
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        last_conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
    ):
        """
        Args:
            input_shape: shape (channels and stride) of the input feature
            num_classes: number of classes to predict
            last_conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        feature_stride = input_shape.stride
        feature_channels = input_shape.channels
        self.last_conv_dims = last_conv_dims
        self.common_stride = common_stride
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight 
        self.scale_factor = 2

        head_ops = []
        head_length = max(1, int(np.log2(feature_stride) - np.log2(self.common_stride)))
        in_channels = feature_channels
        for k in range(head_length):
            out_channels = max(self.last_conv_dims, feature_channels // (self.scale_factor * (k + 1)))
            norm_module = get_norm(norm, out_channels)
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not norm,
                norm=norm_module,
                activation=F.relu,  # TODO: change to SiLU later
            )
            in_channels = out_channels
            weight_init.c2_msra_fill(conv)
            head_ops.append(conv)
            head_ops.append(nn.Upsample(
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False
            ))

        self.sem_seg_head = nn.Sequential(*head_ops)
        self.predictor = Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)


    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.sem_seg_head(features)
        x = self.predictor(x)
        if self.training:
            return None, self.losses(x, targets)
        else:
            x = F.interpolate(
                x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return x, {}

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses