# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn.parameter import Parameter
from torch.nn import init

from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from typing import Callable, List, Optional, Sequence, Tuple, Union

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model



class Conv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        if dtype is not None:
            self.to(dtype=dtype)
        if device is not None:
            self.to(device=device)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], p_in=0, p_out=0):
        #TODO: input muss reduziert werden. 
        if p_out > weight.shape[0]:
            raise ValueError(f"Input tensor has too many elements: {weight.shape[0]} > {bias.shape[0]}")
            # print("padding weight")
            # padding = p_out - weight.shape[0]
            # zeros = torch.zeros(padding, *weight.shape[1:], dtype=weight.dtype, device=weight.device)
            # weight = torch.cat([weight, zeros], dim=0)
            # if bias != None:
            #     zeros = torch.zeros(padding, *bias.shape[1:], dtype=bias.dtype, device=bias.device)
            #     bias = torch.cat([bias, zeros], dim=0)


        if self.groups == 1:
                if p_in == 0:
                    p_in = input.shape[1]

                #padding = p_in - weight.shape[1]
                if bias != None:
                    if weight.shape[0] > bias.shape[0]:
                        raise ValueError(f"Input tensor has too many elements: {weight.shape[0]} > {bias.shape[0]}")

                        # zeros = torch.zeros(weight.shape[0], *bias.shape[1:], dtype=bias.dtype, device=bias.device)
                        # bias = nn.Parameter(torch.cat([bias, zeros], dim=0))
                    else:
                        bias = bias[:weight.shape[0]]
                
                if p_in > weight.shape[1]:
                    raise ValueError(f"Input tensor has too many elements: {p_in} > {weight.shape[1]}")

                    # zeros = torch.zeros(weight.shape[0], padding, weight.shape[2], weight.shape[3],
                    # dtype=weight.dtype, device=weight.device)
                    # weight = nn.Parameter(torch.cat([weight, zeros], dim=1))
                else:
                    weight = weight[:,:p_in]

                return F.conv2d(input, weight, bias, self.stride,
                    self.padding, self.dilation, self.groups)
        else:
            # TODO:
            # Groups so we need to slim the tensor after training?
            # We could also cut weights and do self.groups - len(cut_weights)
            # TODO weights and bias are a still problem
            # handel groups: 
            # 1. calculate groups ?dim / (P_in / p_out)? 
            # self.groups = p_out | self.groups = p_in
            # 2. just cut weights and set groups to weight size 
            groups = p_out
            if bias != None:
                return F.conv2d(input, weight[:p_out, :p_in], bias[:p_out], self.stride,
                    self.padding, self.dilation, groups)
            else:
                
                return F.conv2d(input, weight[:p_out, :p_in], bias, self.stride,
                        self.padding, self.dilation, groups)
        

    def forward(self, input: Tensor, p_out=0) -> Tensor:
        p_in = input.shape[1]
        if p_out == 0:
            #print("-------Error p_out was empty-------")
            #p_in = p_out
            p_out = p_in
        
        # p_percent = p_list.pop(0)
        # if p_percent > 1:
        #     print(f"got the percent {p_percent}, which is > 1") #TODO
        # if len(input.shape) == 4:
        #     p_out = round(self.weight.shape[0] * p_percent)
            # print(f"Input shape {self.weight.shape} selected: {self.weight.shape[0]}")
            # print(f"calculated self.weight.shape[0] ({self.weight.shape[0]}) * p_percent ({p_percent}) =  {self.weight.shape[0] * p_percent}")
        # if len(input.shape) == 3:
        #     p_out = round(len(input[0,0,:]) * p_percent)
            
        return self._conv_forward(input, self.weight, self.bias, p_in, p_out)  

class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.last_layer_shape = 0
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        
    def forward(self, input: Tensor, p_out=0) -> Tensor:
        p_in = input.shape[-1]

        if p_out == 0:
            return F.linear(input, self.weight[:, :input.shape[1]], self.bias[:self.weight.shape[0]] if self.bias is not None else None)

        if p_out > self.weight.shape[0]:
            raise ValueError(f"Input tensor has too many elements: {p_out} > {self.weight.shape[0]}")
            # print("padding weight")
            # _weight = self.weight
            # _bias = self.bias
            # padding = p_out - _weight.shape[0]
            # zeros = torch.zeros(padding, *_weight.shape[1:], dtype=_weight.dtype, device=_weight.device)
            # self.weight = nn.Parameter(torch.cat([_weight, zeros], dim=0))
            # if _bias != None:
            #     zeros = torch.zeros(padding, *_bias.shape[1:], dtype=_bias.dtype, device=_bias.device)
            #     _bias = nn.Parameter(torch.cat([_bias, zeros], dim=0))
            #     self.bias = _bias
        actual_in_dim = min(p_in, self.weight.shape[1]) #NOTE: Is this correct?
        input_slice = input[..., :actual_in_dim]    
        weight_slice = self.weight[:p_out, :actual_in_dim]

        return F.linear(input_slice, weight_slice, self.bias[:p_out])

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
    
    def forward(self, x, p_out=0):
        normalized_shape = self.normalized_shape
        if self.data_format == "channels_last":
            if p_out != 0:
                normalized_shape = (p_out,)
                _weight = self.weight
                _bias = self.bias
                if p_out > _weight.shape[0]:
                    raise ValueError(f"Input tensor has too many elements: {p_out} > {_weight.shape[0]}")
                    # padding_size = p_out - _weight.shape[0]
                    
                    # zeros_w = torch.zeros(padding_size, *_weight.shape[1:], dtype=_weight.dtype, device=_weight.device)
                    # self.weight = nn.Parameter(torch.cat([_weight.data, zeros_w], dim=0))

                    # if _bias is not None:
                    #     zeros_b = torch.zeros(padding_size, *_bias.shape[1:], dtype=_bias.dtype, device=_bias.device)
                    #     self.bias = nn.Parameter(torch.cat([_bias.data, zeros_b], dim=0))
                
                return F.layer_norm(x, normalized_shape, self.weight[:p_out], self.bias[:p_out] if self.bias is not None else None, self.eps)
            else:
                self.normalized_shape = (x.shape[1], )
                weight = self.weight[:x.shape[1]]
                bias = self.bias[:x.shape[1]]
                return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        elif self.data_format == "channels_first": #TODO
            if x.shape[1] > self.weight.shape[0]:
                raise ValueError(f"Input tensor has too many elements: {x.shape[1]} > {self.weight.shape[0]}")

                # _weight = self.weight
                # _bias = self.bias
                # padding_size = x.shape[1] - _weight.shape[0]
                # zeros_w = torch.zeros(padding_size, *_weight.shape[1:], dtype=_weight.dtype, device=_weight.device)
                # self.weight = nn.Parameter(torch.cat([_weight.data, zeros_w], dim=0))
                # zeros_b = torch.zeros(padding_size, *_bias.shape[1:], dtype=_bias.dtype, device=_bias.device)
                # self.bias = nn.Parameter(torch.cat([_bias.data, zeros_b], dim=0))
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            p_out = x.shape[1] #FIXME: Is this correct or does the tensor goes to 0?
            x = self.weight[:p_out, None, None] * x + self.bias[:p_out, None, None]
            return x

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
        self.dim = dim

        self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, p_percent):

        #print(f"dims are: {self.dim}")
        p_out = round(self.dim * p_percent)
        #print(f"new Dim is: {p_out}")

        #TODO: slim whole block
        #For Conv: output = input
        #Linear Layer: 
        # FIXME: p_out can also get bigger with padding with 0er
        # slim at the beginning


        if p_out > x.shape[1]:
            pedding_dim =  p_out - x.shape[1]
            zeros = torch.zeros(x.size(0), pedding_dim, x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, zeros], dim=1)
        else:
            x = x[:,:p_out]
    
        input = x
        x = self.dwconv(x, p_out)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x, p_out)

        x = self.pwconv1(x, 4*p_out) # Note here we want 4 * p_out
        x = self.act(x)
        x = self.pwconv2(x, p_out)
        if self.gamma is not None:
            if p_out <= self.gamma.shape[0]:
                x = self.gamma[:p_out] * x
            else: # p_out_block ist hier größer als self.gamma.shape[0]
                raise ValueError(f"Input tensor has too many elements: {p_out} > {self.gamma.shape[0]}")
                # gamma_padded = torch.ones(p_out, device=self.gamma.device, dtype=self.gamma.dtype)
                # gamma_padded[:self.gamma.shape[0]] = self.gamma
                # x = gamma_padded * x

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)

        #TODO: 
        #1. include conv layer to make up, down scaling (projection)
        #2. just remove dimensions between layers
        #3. TODO: padding of the skip connection
        #4. p_procent can only get smaller => prob worse!
        #5. keep dimension of beginning of block and end of block
        return x
    

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        p_rates (list[float]): rates at which each block should be slimmed. Default: [1, 1, 1, 1]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            #TODO: look into the downsample Layers
            #Maybe not possible with Sequential-

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])] #FIXME: p_rates are hard coded
            )
            self.stages.append(stage)
            cur += depths[i]
        

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_last") # final norm layer #NOTE: was nn.Layernorm before
        self.head = Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, p_list):

        for i in range(4): # TODO split open and give p_out!
            x = self.downsample_layers[i][0](x) # self.downsample_layers[i][0](x) to access LayerNorm (with 1 for downsampling)
            #print(f"The Down1 has the shape: {x.shape}")
            x = self.downsample_layers[i][1](x) 
            #print(f"The Down2 has the shape: {x.shape}")


            for s in enumerate(self.stages[i]): # iterate through stages to add p_values (maybe enumerate)#
                p_percent = p_list.pop()
                x = s[1](x, p_percent)
                #print(f"The Tensor has the shape: {x.shape} and the last p_percent was: {p_percent}")
                #print(p_list)

        
        #print(f"The Tensor has the shape: {x.shape} and the last p_percent was: {p_percent}")

        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, p_list): # ADD: give p_out list
        x = self.forward_features(x, p_list)
        x = self.head(x)
        #print(f"Head has: {x.shape}")
        return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


# if __name__ == '__main__':
#     print("--- Testing Elastic Linear Layer ---")
#     batch_size = 2
#     channels = 96
#     height = width = 56
#     dummy_input = torch.randn(batch_size, channels, height, width)

#     # Schritt 2: Testobjekt initialisieren
#     p_rates = [0.5, 1.5, 0.8]  # kepp 50 % of channels or increase channels by 150 %
#     block = Block(dim=channels, drop_path=0.1, layer_scale_init_value=1e-6, p_rates=p_rates)

#     # Schritt 3: Vorwärtsdurchlauf
#     output = block(dummy_input)
#     print(f"Output shape: {output.shape}")


