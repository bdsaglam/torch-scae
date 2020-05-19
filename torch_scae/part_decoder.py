import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monty.collections import AttrDict

from torch_scae import math_ops
from torch_scae.distributions import GaussianMixture
from torch_scae.general_utils import prod
from torch_scae.nn_ext import relu1, MLP


class TemplateBasedImageDecoder(nn.Module):
    """Template-based primary capsule decoder for images."""

    _templates = None

    def __init__(self,
                 n_templates,
                 template_size,
                 output_size,
                 n_channels=1,
                 n_template_features=None,
                 learn_output_scale=False,
                 colorize_templates=False,
                 template_nonlin=relu1,
                 color_nonlin=relu1,
                 use_alpha_channel=False,
                 background_image=True):

        super().__init__()
        self._n_templates = n_templates
        self._template_size = template_size
        self._n_channels = n_channels
        self._output_size = output_size
        self._learn_output_scale = learn_output_scale
        self._colorize_templates = colorize_templates
        self._color_nonlin = color_nonlin
        self._n_template_features = n_template_features
        self._template_nonlin = template_nonlin
        self._use_alpha_channel = use_alpha_channel
        self._background_image = background_image

        self._build()

    def _build(self):
        self._setup_templates()
        self.bg_mixing_logit = nn.Parameter(torch.tensor([0.0]),
                                            requires_grad=True)
        if self._background_image:
            self.bg_value = nn.Parameter(torch.tensor([0.0]),
                                         requires_grad=True)

    def _setup_templates(self):
        # create templates
        template_shape = (
            1, self._n_templates, self._n_channels, *self._template_size
        )

        # make each templates orthogonal to each other at init
        n_elems = prod(template_shape[2:])  # height, width and channel
        n = max(self._n_templates, n_elems)
        q = np.random.uniform(size=[n, n])
        q = np.linalg.qr(q)[0]
        q = q[:self._n_templates, :n_elems].reshape(template_shape)
        q = q.astype(np.float32)
        q = (q - q.min()) / (q.max() - q.min())
        self.template_logits = nn.Parameter(torch.from_numpy(q),
                                            requires_grad=True)

        if self._use_alpha_channel:
            shape = (1, self._n_templates, 1, *self._template_size)
            self.templates_alpha = nn.Parameter(torch.zeros(*shape),
                                                requires_grad=True)
        else:
            self.temperature_logit = nn.Parameter(torch.rand(1),
                                                  requires_grad=True)

        if self._colorize_templates:
            self.templates_color_mlp = MLP(
                sizes=[self._n_template_features, 32, self._n_channels]
            )

        if self._learn_output_scale:
            self.scale = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self,
                pose,
                presence=None,
                template_feature=None,
                bg_image=None):
        """Builds the module.

        Args:
          pose: [B, n_templates, 6] tensor.
          presence: [B, n_templates] tensor.
          template_feature: [B, n_templates, n_features] tensor; these features
          are used to change templates based on the input, if present.
          bg_image: [B, *output_size] tensor representing the background.

        Returns:
          [B, n_templates, *output_size, n_channels] tensor.
        """
        device = next(iter(self.parameters())).device
        batch_size, n_templates = list(pose.shape[:2])

        raw_templates = self._template_nonlin(self.template_logits)

        if self._colorize_templates and template_feature is not None:
            template_color = self.templates_color_mlp(
                template_feature.view(batch_size * n_templates, -1)
            )  # (BxM, C)
            if self._color_nonlin == relu1:
                template_color += .99
            template_color = self._color_nonlin(template_color)
            template_color = template_color.view(
                batch_size, n_templates, template_color.shape[1]
            )  # (B, M, C)
            templates = raw_templates * template_color[:, :, :, None, None]

        if templates.shape[0] == 1:
            templates = templates.repeat(batch_size, 1, 1, 1, 1)

        # transform templates
        templates = templates.view(batch_size * n_templates,
                                   *templates.shape[2:])
        affine_matrices = pose.view(batch_size * n_templates, 2, 3)
        target_size = [
            batch_size * n_templates, self._n_channels, *self._output_size]
        affine_grids = F.affine_grid(affine_matrices, target_size)
        transformed_templates = F.grid_sample(templates, affine_grids)
        transformed_templates = transformed_templates.view(
            batch_size, n_templates, *target_size[1:])

        #
        if bg_image is not None:
            bg_image = bg_image.unsqueeze(1)
        else:
            bg_image = torch.sigmoid(self.bg_value).repeat(
                *transformed_templates[:, :1].shape)

        transformed_templates = torch.cat([transformed_templates, bg_image],
                                          dim=1)

        if presence is not None:
            bg_presence = torch.ones([batch_size, 1])
            presence = torch.cat([presence, bg_presence], dim=1)

        if self._use_alpha_channel:
            template_mixing_logits = self.templates_alpha.repeat(
                batch_size, 1, 1, 1, 1)
            template_mixing_logits = template_mixing_logits.view(
                batch_size * n_templates, *template_mixing_logits.shape[2:])
            template_mixing_logits = F.grid_sample(template_mixing_logits,
                                                   affine_grids)
            template_mixing_logits = template_mixing_logits.view(
                batch_size, n_templates, *template_mixing_logits.shape[1:])

            bg_mixing_logit = F.softplus(self.bg_mixing_logit).repeat(
                *template_mixing_logits[:, :1].shape)
            template_mixing_logits = torch.cat(
                [template_mixing_logits, bg_mixing_logit], dim=1)
        else:
            temperature = F.softplus(self.temperature_logit + .5) + 1e-4
            template_mixing_logits = transformed_templates / temperature

        if self._learn_output_scale:
            scale = F.softplus(self.scale) + 1e-4
        else:
            scale = 1

        presence = presence.view(*presence.shape,
                                 *([1] * len(template_mixing_logits.shape[2:])))
        template_mixing_logits += math_ops.log_safe(presence)

        rec_pdf = GaussianMixture.make_from_stats(
            loc=transformed_templates,
            scale=scale,
            mixing_logits=template_mixing_logits
        )

        return AttrDict(
            raw_templates=raw_templates,
            transformed_templates=transformed_templates[:, :-1],
            mixing_logits=template_mixing_logits[:, :-1],
            pdf=rec_pdf,
        )
