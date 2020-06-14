# Copyright 2020 The Google Research Authors.
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

import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monty.collections import AttrDict

from torch_scae import math_ops
from torch_scae.distributions import GaussianMixture
from torch_scae.general_utils import prod
from torch_scae.nn_ext import relu1, MLP
from torch_scae.nn_utils import choose_activation


class TemplateGenerator(nn.Module):
    """Template-based primary capsule decoder for images."""

    def __init__(self,
                 n_templates,
                 n_channels,
                 template_size,
                 template_nonlin='relu1',
                 dim_feature=None,
                 colorize_templates=False,
                 color_nonlin='relu1'):

        super().__init__()
        self.n_templates = n_templates  # M
        self.template_size = template_size  # (H, W)
        self.n_channels = n_channels  # C
        self.template_nonlin = choose_activation(template_nonlin)
        self.dim_feature = dim_feature  # F
        self.colorize_templates = colorize_templates
        self.color_nonlin = choose_activation(color_nonlin)

        self._build()

    def _build(self):
        # create templates
        template_shape = (
            1, self.n_templates, self.n_channels, *self.template_size
        )

        # make each templates orthogonal to each other at init
        n_elems = prod(template_shape[2:])  # channel, height, width
        n = max(self.n_templates, n_elems)
        q = np.random.uniform(size=[n, n])
        q = np.linalg.qr(q)[0]
        q = q[:self.n_templates, :n_elems].reshape(template_shape)
        q = q.astype(np.float32)
        q = (q - q.min()) / (q.max() - q.min())
        self.template_logits = nn.Parameter(torch.from_numpy(q),
                                            requires_grad=True)

        if self.colorize_templates:
            self.templates_color_mlp = MLP(
                sizes=[self.dim_feature, 32, self.n_channels])

    def forward(self, feature=None, batch_size=None):
        """
        Args:
          feature: [B, n_templates, dim_feature] tensor; these features
          are used to change templates based on the input, if present.
          batch_size (int): batch_size in case feature is None

        Returns:
          (B, n_templates, n_channels, *template_size) tensor.
        """
        # (B, M, F)
        if feature is not None:
            batch_size = feature.shape[0]

        # (1, M, C, H, W)
        raw_templates = self.template_nonlin(self.template_logits)

        if self.colorize_templates and feature is not None:
            n_templates = feature.shape[1]
            template_color = self.templates_color_mlp(
                feature.view(batch_size * n_templates, -1)
            )  # (BxM, C)
            if self.color_nonlin == relu1:
                template_color += .99
            template_color = self.color_nonlin(template_color)
            template_color = template_color.view(
                batch_size, n_templates, template_color.shape[1]
            )  # (B, M, C)
            templates = raw_templates * template_color[:, :, :, None, None]
        else:
            templates = raw_templates.repeat(batch_size, 1, 1, 1, 1)

        return AttrDict(
            raw_templates=raw_templates,
            templates=templates,
        )


class TemplateBasedImageDecoder(nn.Module):
    """Template-based primary capsule decoder for images."""

    def __init__(self,
                 n_templates: int,
                 template_size: Tuple[int, int],
                 output_size: Tuple[int, int],
                 learn_output_scale=False,
                 use_alpha_channel=False,
                 background_value=True):

        super().__init__()
        self.n_templates = n_templates
        self.template_size = template_size
        self.output_size = output_size
        self.learn_output_scale = learn_output_scale
        self.use_alpha_channel = use_alpha_channel
        self.background_value = background_value

        self._build()

    def _build(self):
        if self.use_alpha_channel:
            shape = (1, self.n_templates, 1, *self.template_size)
            self.templates_alpha = nn.Parameter(torch.zeros(*shape),
                                                requires_grad=True)
        else:
            self.temperature_logit = nn.Parameter(torch.rand(1),
                                                  requires_grad=True)

        if self.learn_output_scale:
            self.scale = nn.Parameter(torch.rand(1), requires_grad=True)

        self.bg_mixing_logit = nn.Parameter(torch.tensor([0.0]),
                                            requires_grad=True)
        if self.background_value:
            self.bg_value = nn.Parameter(torch.tensor([0.0]),
                                         requires_grad=True)

    def forward(self,
                templates,
                pose,
                presence=None,
                bg_image=None):
        """Builds the module.

        Args:
          templates: (B, n_templates, n_channels, *template_size) tensor
          pose: [B, n_templates, 6] tensor.
          presence: [B, n_templates] tensor.
          bg_image: [B, n_channels, *output_size] tensor representing the background.

        Returns:
          (B, n_templates, n_channels, *output_size) tensor.
        """
        device = templates.device

        # B, M, C, H, W
        batch_size, n_templates, n_channels, height, width = templates.shape

        # transform templates
        templates = templates.view(batch_size * n_templates,
                                   *templates.shape[2:])  # (B*M, C, H, W)
        affine_matrices = pose.view(batch_size * n_templates, 2, 3)  # (B*M, 2, 3)
        target_size = [
            batch_size * n_templates, n_channels, *self.output_size]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            affine_grids = F.affine_grid(affine_matrices, target_size)
            transformed_templates = F.grid_sample(
                templates, affine_grids, align_corners=False)
        transformed_templates = transformed_templates.view(
            batch_size, n_templates, *target_size[1:])
        del templates, target_size, affine_matrices

        # background image
        if bg_image is not None:
            bg_image = bg_image.unsqueeze(1)
        else:
            bg_image = torch.sigmoid(self.bg_value).repeat(
                *transformed_templates[:, :1].shape)

        transformed_templates = torch.cat([transformed_templates, bg_image], 1)
        del bg_image

        if self.use_alpha_channel:
            template_mixing_logits = self.templates_alpha.repeat(
                batch_size, 1, 1, 1, 1)
            template_mixing_logits = template_mixing_logits.view(
                batch_size * n_templates, *template_mixing_logits.shape[2:])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                template_mixing_logits = F.grid_sample(
                    template_mixing_logits, affine_grids, align_corners=False)
            template_mixing_logits = template_mixing_logits.view(
                batch_size, n_templates, *template_mixing_logits.shape[1:])

            bg_mixing_logit = F.softplus(self.bg_mixing_logit).repeat(
                *template_mixing_logits[:, :1].shape)
            template_mixing_logits = torch.cat(
                [template_mixing_logits, bg_mixing_logit], dim=1)
            del bg_mixing_logit
        else:
            temperature = F.softplus(self.temperature_logit + .5) + 1e-4
            template_mixing_logits = transformed_templates / temperature
            del temperature

        if self.learn_output_scale:
            scale = F.softplus(self.scale) + 1e-4
        else:
            scale = torch.tensor([1.0], device=device)

        if presence is not None:
            bg_presence = presence.new_ones([batch_size, 1])
            presence = torch.cat([presence, bg_presence], dim=1)
            presence = presence.view(
                *presence.shape, *([1] * len(template_mixing_logits.shape[2:])))
            template_mixing_logits += math_ops.log_safe(presence)
            del bg_presence, presence

        rec_pdf = GaussianMixture.make_from_stats(
            loc=transformed_templates,
            scale=scale,
            mixing_logits=template_mixing_logits
        )

        return AttrDict(
            transformed_templates=transformed_templates,
            mixing_logits=template_mixing_logits,
            pdf=rec_pdf,
        )
