"""
Implementation of ConvNeXt models from paper "A ConvNet for the 2020s".

This code is adapted from https://github.com/facebookresearch/ConvNeXt with minimal modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.layers import trunc_normal_, DropPath
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.stride = 1

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(Backbone):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self, 
        in_chans=3, 
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0., 
        layer_scale_init_value=1e-6, 
        out_features=["stage_1", "stage_2", "stage_3", "stage_4"],
        size_divisibility=0,
        pretrained=False,
        pretrained_path=None
    ):
        super().__init__()

        self._out_feature_channels = dict()
        self._out_feature_strides = dict()
        self.downsample_layers = nn.ModuleDict() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers[f"downsample_layer_1"] = stem
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers[f"downsample_layer_{i+2}"] = downsample_layer

        self.stages = nn.ModuleDict() # 4 feature resolution stages, each consisting of multiple residual blocks

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        cur_stride = 1
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages[f"stage_{i+1}"] = stage
            self._out_feature_channels[f"stage_{i+1}"] = list(stage.children())[-1].pwconv2.out_features

            downsample_stride = np.prod([l.stride[0] for l in self.downsample_layers[f"downsample_layer_{i+1}"] if isinstance(l, nn.Conv2d)])
            stage_stride = np.prod([s.stride for s in stage])
            cur_stride *= downsample_stride * stage_stride
            self._out_feature_strides[f"stage_{i+1}"] = cur_stride
            cur += depths[i]

        self._out_features = out_features

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm_{i_layer+1}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

        if pretrained:
            assert pretrained_path, "`pretrained_path` is None"
            renamed_state_dict = self._rename_state_dict(pretrained_path)
            self.load_state_dict(renamed_state_dict, strict=False)

        self._size_divisibility = size_divisibility

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = dict()
        for i in range(4):
            stage_name = f"stage_{i+1}"
            x = self.downsample_layers[f"downsample_layer_{i+1}"](x)
            x = self.stages[stage_name](x)
            if stage_name in self._out_features:
                norm_layer = getattr(self, f'norm_{i+1}')
                x_out = norm_layer(x)
                outputs[stage_name] = x_out

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def _rename_state_dict(self, pretrained_path):
        """
        Rename the state_dict from pretrained_model in order to make it 
        compatible with our version.
        """
        checkpoint = torch.hub.load_state_dict_from_url(
            url=pretrained_path, 
            map_location="cpu", 
            check_hash=True
        )
        state_dict = checkpoint["model"]
        renamed_state_dict = {}

        for key, value in state_dict.items():
            if "downsample_layers" in key:
                old_keys = key.split(".")
                new_keys = old_keys
                new_keys[1] = f"downsample_layer_{int(old_keys[1]) + 1}"
                new_key = ".".join(new_keys)
            elif "stages" in key:
                old_keys = key.split(".")
                new_keys = old_keys
                new_keys[1] = f"stage_{int(old_keys[1]) + 1}"
                new_key = ".".join(new_keys)
            else:
                continue
            renamed_state_dict[new_key] = value

        return renamed_state_dict

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x