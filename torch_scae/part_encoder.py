import collections
from typing import Tuple

import torch
import torch.nn as nn

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
        self.network = Conv2dStack(in_channel=input_shape[0],
                                   out_channels=out_channels,
                                   kernel_sizes=kernel_sizes,
                                   strides=strides,
                                   activation=activation,
                                   activate_final=activate_final)
        self.output_shape = measure_shape(self.network, input_shape=input_shape)

    def forward(self, image):
        return self.network(image)


class CapsuleImageEncoder(nn.Module):
    """Primary capsule for images."""
    Result = collections.namedtuple(
        'PrimaryCapsuleResult',
        ['pose', 'feature', 'presence', 'presence_logit', 'img_embedding']
    )

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
        self._encoder = encoder
        self._n_caps = n_caps  # M
        self._n_poses = n_poses  # P
        self._n_special_features = n_special_features  # S
        self._noise_scale = noise_scale
        self._similarity_transform = similarity_transform

        self.build()

    def build(self):
        self.img_embedding_bias = nn.Parameter(
            data=torch.zeros(self._encoder.output_shape, dtype=torch.float32),
            requires_grad=True
        )
        in_channels = self._encoder.output_shape[0]
        self.caps_dim_splits = [self._n_poses, 1, self._n_special_features]  # 1 for presence
        self.n_total_caps_dims = sum(self.caps_dim_splits)
        out_channels = self._n_caps * (self.n_total_caps_dims + 1)  # 1 for attention
        self.att_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, image):  # (B, C, H, H)
        batch_size = image.shape[0]  # B
        img_embedding = self._encoder(image)  # (B, D, G, G)

        h = img_embedding + self.img_embedding_bias.unsqueeze(0)  # (B, D, G, G)
        h = self.att_conv(h)  # (B, M * (P + S + 1 + 1), G, G)
        h = multiple_attention_pooling_2d(h, self._n_caps)  # (B, M * (P + S + 1), 1, 1)
        h = h.view(batch_size, self._n_caps, self.n_total_caps_dims)  # (B, M, (P + S + 1))

        # (B, M, P), (B, M, 1), (B, M, S)
        pose, presence_logit, special_feature = torch.split(h, self.caps_dim_splits, -1)
        if self._n_special_features == 0:
            special_feature = None

        presence_logit = presence_logit.squeeze(-1)  # (B, M)
        if self._noise_scale > 0. and self.training:
            presence_logit += (torch.rand(*presence_logit.shape) - .5) * self._noise_scale

        presence_prob = torch.sigmoid(presence_logit)
        pose = cv_ops.geometric_transform(pose, self._similarity_transform)
        return self.Result(pose, special_feature, presence_prob, presence_logit, img_embedding)
