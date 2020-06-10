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
import torch.nn as nn
from monty.collections import AttrDict

from torch_scae import cv_ops
from torch_scae.nn_ext import Conv2dStack, multiple_attention_pooling_2d
from torch_scae.nn_utils import measure_shape


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 out_channels,
                 kernel_sizes,
                 strides,
                 activation=nn.ReLU,
                 activate_final=True):
        super().__init__()
        self.network = Conv2dStack(in_channels=input_shape[0],
                                   out_channels=out_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   activation=activation,
                                   activate_final=activate_final)
        self.output_shape = measure_shape(self.network, input_shape=input_shape)

    def forward(self, image):
        return self.network(image)


class CapsuleImageEncoder(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 encoder: CNNEncoder,
                 n_caps: int,
                 n_poses: int,
                 n_special_features: int = 0,
                 noise_scale: float = 4.,
                 similarity_transform: bool = False,
                 ):

        super().__init__()
        self.input_shape = input_shape
        self.encoder = encoder
        self.n_caps = n_caps  # M
        self.n_poses = n_poses  # P
        self.n_special_features = n_special_features  # S
        self.noise_scale = noise_scale
        self.similarity_transform = similarity_transform

        self._build()

        self.output_shapes = AttrDict(
            pose=(n_caps, n_poses),
            presence=(n_caps,),
            feature=(n_caps, n_special_features),
        )

    def _build(self):
        self.img_embedding_bias = nn.Parameter(
            data=torch.zeros(self.encoder.output_shape, dtype=torch.float32),
            requires_grad=True
        )
        in_channels = self.encoder.output_shape[0]
        self.caps_dim_splits = [self.n_poses, 1, self.n_special_features]  # 1 for presence
        self.n_total_caps_dims = sum(self.caps_dim_splits)
        out_channels = self.n_caps * (self.n_total_caps_dims + 1)  # 1 for attention
        self.att_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, image):  # (B, C, H, H)
        batch_size = image.shape[0]  # B

        img_embedding = self.encoder(image)  # (B, D, G, G)

        h = img_embedding + self.img_embedding_bias.unsqueeze(0)  # (B, D, G, G)
        h = self.att_conv(h)  # (B, M * (P + 1 + S + 1), G, G)
        h = multiple_attention_pooling_2d(h, self.n_caps)  # (B, M * (P + 1 + S), 1, 1)
        h = h.view(batch_size, self.n_caps, self.n_total_caps_dims)  # (B, M, (P + 1 + S))
        del img_embedding

        # (B, M, P), (B, M, 1), (B, M, S)
        pose, presence_logit, special_feature = torch.split(h, self.caps_dim_splits, -1)
        del h

        if self.n_special_features == 0:
            special_feature = None

        presence_logit = presence_logit.squeeze(-1)  # (B, M)
        if self.training and self.noise_scale > 0.:
            noise = (torch.rand_like(presence_logit) - .5) * self.noise_scale
            presence_logit = presence_logit + noise  # (B, M)

        presence = torch.sigmoid(presence_logit)  # (B, M)
        pose = cv_ops.geometric_transform(pose, self.similarity_transform)  # (B, M, P)
        return AttrDict(pose=pose,
                        presence=presence,
                        feature=special_feature)
