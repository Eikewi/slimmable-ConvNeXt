import math
import torch.nn as nn
import torch
#from torch.nn import Linear
from torch import Tensor, Size
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F

from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from typing import Callable, List, Optional, Sequence, Tuple, Union

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

    def forward(self, input: Tensor, p_list:list[float]) -> Tensor:
        p_out = 0
        p_in = input.shape[1]
        p_percent = p_list.pop()
        if p_percent > 1:
            print(f"got the percent {p_percent}, which is > 1")
        if len(input.shape) == 2:
            p_out = round(len(input[0,:]) * p_percent)
            print(p_out)
        if len(input.shape) == 3:
            p_out = round(len(input[0,0,:]) * p_percent)

        if p_out:
            if len(input.shape) == 2:
                return F.linear(input,
                         self.weight[0:p_out,0:p_in],
                         self.bias[0:p_out])
            if len(input.shape) == 3:
                return F.linear(input,
                         self.weight[0:p_out,0:p_in],
                         self.bias[0:p_out])
            return F.linear(input, self.weight, self.bias)
        


class Conv2d(_ConvNd):

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
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], p_in: int, p_out: int):
        # if self.padding_mode != 'zeros':
        #     return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
        #                     weight[:p_out], bias[:p_out], self.stride,
        #                     _pair(0), self.dilation, self.groups)
        #TODO: input muss reduziert werden. 
        if self.groups == 1:
                if bias != None:
                    return F.conv2d(input, weight[:p_out, :p_in], bias[:p_out], self.stride,
                        self.padding, self.dilation, self.groups)
                else:
                    return F.conv2d(input, weight[:p_out, :p_in], bias, self.stride,
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
            print(p_out)
            self.groups = p_out
            if bias != None:
                return F.conv2d(input, weight[:p_out, :p_in], bias[:p_out], self.stride,
                    self.padding, self.dilation, self.groups)
            else:
                return F.conv2d(input, weight[:p_out, :p_in], bias, self.stride,
                        self.padding, self.dilation, self.groups)
        

    def forward(self, input: Tensor, p_out: list) -> Tensor:
        #p_in = input.shape[1]
        p_in = p_out
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
    
    def forward(self, x, p_out):
        self.normalized_shape = (p_out, )
        if self.data_format == "channels_last":
            print(self.weight.shape)
            return F.layer_norm(x, self.normalized_shape, self.weight[:p_out], self.bias[:p_out], self.eps)
        elif self.data_format == "channels_first": #TODO
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
    # def forward_Fix_Value(self, input: Tensor, p_list:list, p_out=-1) -> Tensor:
    #     p_in = len(input[:])
    #     print(p_in)
    #     if len(p_list) > 0:
    #         p_out = p_list.pop()
    #         # Eher p_out definieren
    #         # p_in dann später durch input berechnen
    #     if p_out== -1:
    #         p_out = p_in
    #     if p_in:
    #         if len(input.shape) == 2:
    #             return F.linear(input[:,0:p_in], #TODO F.linear verstehen
    #                      self.weight[0:p_out,0:p_in],
    #                      self.bias[0:p_out])
    #         if len(input.shape) == 3:
    #             return F.linear(input[:,:,0:p_in],
    #                      self.weight[0:p_out,0:p_in],
    #                      self.bias[0:p_out])
    #         return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    




p_percent = [0.5, 0.1, 0.3] 
#p_in = [10, 5, 3]

#------------Linear Layer--------------
# m = Linear(3, 10)
# n = nn.Linear(3,10)
# input = torch.randn(5, 3)
# # m.weight.data.fill_(0.1)
# # m.bias.data.zero_()
# # n.weight.data.fill_(0.1)
# # n.bias.data.zero_()
# n.weight.data = m.weight.data
# n.bias.data = m.bias.data
# #output = m(input)
# output = m(input, p_percent)
# output2 = n(input)

# print(f"Weights:{m.weight}\n")
# print(f"Bias: {m.bias}\n" )
# print("Normal Output: \n")
# print(output2)
# print("slim output:\n")
# print(output)

#------------Conv Layer--------------
# input_data = [
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0],
#     [7.0, 8.0, 9.0]
# ]
# input_tensor = torch.tensor([[input_data, input_data]])
# print("Eingabe-Tensor (Shape: {}):\n{}".format(input_tensor.shape, input_tensor))
# print("-" * 30)

# k = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride=1, padding=0, bias=False, groups=4)
# l = Conv2d(in_channels=4, out_channels=4, kernel_size=2, stride=1, padding=0, bias=False, groups=4)

# custom_kernel = [[[1.0, 0.0],
#                  [0.0, 1.0]]]

# kernel_list = [custom_kernel, custom_kernel, custom_kernel, custom_kernel]

# kernel = torch.tensor(kernel_list)
# print(kernel.shape)

# with torch.no_grad(): # Wichtig, da Gewichte Parameter sind und Gradientenverfolgung haben
#     k.weight = nn.Parameter(kernel)
#     l.weight = nn.Parameter(kernel)

# print("Kernel der Conv-Layer (Shape: {}):\n{}".format(k.weight.shape, k.weight.data))
# print("-" * 30)

# #output_tensor = k(input_tensor)
# print("tensor 2:::")
# output_tensor2 = l(input_tensor, 2)

# #print("Output-Tensor (Shape: {}):\n{}".format(output_tensor.shape, output_tensor))
# print("-" * 30)

# print("Slim Output-Tensor (Shape: {}):\n{}".format(output_tensor2.shape, output_tensor2))
# print("-" * 30)

# # 5. Den Unterschied verstehen (manuelle Berechnung als Check)

# # Der 2x2 Kernel [[1,0],[0,1]] wird über den 3x3 Input geschoben:

# # Erste Position (links oben):
# # [1, 2]   *   [1, 0]  =  1*1 + 2*0 + 4*0 + 5*1 = 1 + 0 + 0 + 5 = 6
# # [4, 5]       [0, 1]
# # Output[0,0] = 6

# # Zweite Position (rechts oben):
# # [2, 3]   *   [1, 0]  =  2*1 + 3*0 + 5*0 + 6*1 = 2 + 0 + 0 + 6 = 8
# # [5, 6]       [0, 1]
# # Output[0,1] = 8

# # Dritte Position (links unten):
# # [4, 5]   *   [1, 0]  =  4*1 + 5*0 + 7*0 + 8*1 = 4 + 0 + 0 + 8 = 12
# # [7, 8]       [0, 1]
# # Output[1,0] = 12

# # Vierte Position (rechts unten):
# # [5, 6]   *   [1, 0]  =  5*1 + 6*0 + 8*0 + 9*1 = 5 + 0 + 0 + 9 = 14
# # [8, 9]       [0, 1]
# # Output[1,1] = 14

# # Erwarteter Output sollte also sein:
# # [[[[ 6.,  8.],
# #    [12., 14.]]]]

# # Die Größe des Outputs:
# # H_out = (H_in - KernelH + 2*Padding) / Stride + 1 = (3 - 2 + 2*0) / 1 + 1 = 1 + 1 = 2
# # W_out = (W_in - KernelW + 2*Padding) / Stride + 1 = (3 - 2 + 2*0) / 1 + 1 = 1 + 1 = 2
# # Also ein 2x2 Output, was wir auch sehen.


# print("Manuelle Berechnung (erwartet):")
# print("[[[[ 6.,  8.],")
# print("   [12., 14.]]]]")


#------------LayerNorm--------------
# x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # Shape: (1, 1, 1, 4)
# norm = LayerNorm(normalized_shape=4, data_format="channels_last")
# out = norm(x, p_out=4)

# print("Input:", x)
# print("Output:", out)
#---------------padding ------------------------

import torch

# Erstelle einen Tensor mit Batch=2, Channels=3, Höhe=4, Breite=4
x = torch.tensor([
    [  # Erstes Bild
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
        
        [[2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2]],
        
        [[3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3]]
    ],
    
    [  # Zweites Bild
        [[4, 4, 4, 4],
         [4, 4, 4, 4],
         [4, 4, 4, 4],
         [4, 4, 4, 4]],
        
        [[5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5],
         [5, 5, 5, 5]],
        
        [[6, 6, 6, 6],
         [6, 6, 6, 6],
         [6, 6, 6, 6],
         [6, 6, 6, 6]]
    ]
], dtype=torch.float32)

print(x.shape)  # Ausgabe: torch.Size([2, 3, 4, 4])
# Wir wollen 2 zusätzliche Null-Kanäle anhängen
pad_channels = 2
zeros = torch.zeros(x.size(0), pad_channels, x.size(2), x.size(3))

x_padded = torch.cat([x, zeros], dim=1)

print(x_padded.shape)  # Ausgabe: torch.Size([2, 5, 4, 4])
print(x_padded)