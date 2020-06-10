# Copyright 2020 Barış Deniz Sağlam.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn.functional as F

from torch_scae import nn_ext


def conv_output_size(in_size: int,
                     kernel_size: int,
                     stride: int = 1,
                     padding: int = 0) -> int:
    return (in_size - kernel_size + 2 * padding) // stride + 1


def conv_output_shape(input_shape: Tuple[int, int, int],
                      out_channels: int,
                      kernel_size: int,
                      stride: int = 1,
                      padding: int = 0) -> Tuple[int, int, int]:
    return (
        out_channels,
        conv_output_size(input_shape[1],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding),
        conv_output_size(input_shape[2],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)
    )


def measure_shape(network, input_shape, input_dtype=torch.float32):
    device = next(iter(network.parameters())).device
    with torch.no_grad():
        input = torch.rand(1, *input_shape, dtype=input_dtype, device=device)
        return network(input).shape[1:]


def choose_activation(name):
    if name == 'sigmoid':
        return torch.sigmoid
    if name == 'relu1':
        return nn_ext.relu1

    act_fn = getattr(F, name, None)

    if act_fn is None:
        raise ValueError('Invalid activation function: "{}".'.format(name))

    return act_fn
