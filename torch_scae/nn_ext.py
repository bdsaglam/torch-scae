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

import torch.nn as nn
import torch.nn.functional as F


def MLP(sizes, activation=nn.ReLU, activate_final=True, bias=True):
    n = len(sizes)
    assert n >= 2, "There must be at least two sizes"

    layers = []
    for j in range(n - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=bias))
        layers.append(activation())

    if not activate_final:
        layers.pop()

    return nn.Sequential(*layers)


def Conv2dStack(in_channels,
                out_channels,
                kernel_sizes,
                strides,
                activation=nn.ReLU,
                activate_final=True):
    assert len(out_channels) == len(kernel_sizes) == len(strides)

    channels = [in_channels] + list(out_channels)
    layers = []
    for i in range(len(channels) - 1):
        in_channels = channels[i]
        out_channels = channels[i + 1]
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride)
        layers.append(conv)
        layers.append(activation())

    if not activate_final:
        layers.pop()

    return nn.Sequential(*layers)


def soft_attention(feature_map, attention_map):
    assert feature_map.shape[0] == attention_map.shape[0]
    assert attention_map.shape[1] == 1
    assert feature_map.shape[2:] == attention_map.shape[2:]
    batch_size, n_channels, height, width = feature_map.shape

    feature_map = feature_map.view(batch_size, n_channels, -1)
    attention_map = attention_map.view(batch_size, 1, -1)
    mask = F.softmax(attention_map, dim=-1)
    x = feature_map * mask
    x = x.view(batch_size, n_channels, height, width)
    return x  # (B, C, H, W)


def multiple_soft_attention(feature_map, n_attention_map):
    batch_size, n_channels, height, width = feature_map.shape
    assert n_attention_map > 0
    assert n_channels > n_attention_map, "Attention maps cannot be more than feature maps"
    assert n_channels % n_attention_map == 0, "Incompatible attention map count"

    feature_map = feature_map.view(batch_size,
                                   n_attention_map,
                                   n_channels // n_attention_map,
                                   -1)

    real_feature_map = feature_map[:, :, :-1, :]
    attention_map = feature_map[:, :, -1:, :]
    attention_mask = F.softmax(attention_map, dim=-1)

    x = real_feature_map * attention_mask
    x = x.view(batch_size, n_channels - n_attention_map, height, width)
    return x  # (B, C - A, H, W)


def multiple_attention_pooling_2d(feature_map, n_attention_map):
    x = multiple_soft_attention(feature_map, n_attention_map)  # (B, C - A, H, W)
    b, c = x.shape[:2]
    x = x.view(b, c, -1)
    x = x.sum(-1, keepdim=True).unsqueeze(-1)  # (B, C - A, 1, 1)
    return x


def attention_pooling_2d_explicit(feature_map, attention_map):
    x = soft_attention(feature_map, attention_map)
    b, c = x.shape[:2]
    x = x.view(b, c, -1)
    x = x.sum(-1, keepdim=True).unsqueeze(-1)
    return x


def attention_pooling_2d(feature_map, attention_channel_index):
    batch_size, n_channels, height, width = feature_map.shape
    if attention_channel_index < 0:
        attention_channel_index = attention_channel_index + n_channels

    feature_map = feature_map.view(batch_size, n_channels, -1)

    attention_map = feature_map[:, [attention_channel_index], :]
    indices = [i for i in range(n_channels) if i != attention_channel_index]
    real_feature_map = feature_map[:, indices, :]

    # (B, C-1, 1, 1)
    x = attention_pooling_2d_explicit(real_feature_map, attention_map)
    return x


class AttentionAveragedPooling2d(nn.Module):

    def __init__(self, attention_channel_index):
        super().__init__()
        self.attention_channel_index = attention_channel_index

    def forward(self, feature_map):
        return attention_pooling_2d(feature_map,
                                    self.attention_channel_index)


def relu1(x):
    return F.relu6(x * 6.) / 6.
