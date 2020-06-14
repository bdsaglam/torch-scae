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

import torch
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianMixture:
    def __init__(self, normal_dist: Normal, mixing_logits):
        """
        Args:
          normal_dist: torch normal distribution, [B, K, ...]
          mixing_logits: tensor [B, K, ...] with K the number of components.
        """
        self.dist = normal_dist
        self.mixing_logits = mixing_logits

    @property
    def n_components(self):
        return self.mixing_logits.shape[1]

    def mixing_log_prob(self):
        return F.log_softmax(self.mixing_logits, 1)

    def mean(self):
        mixing_prob = F.softmax(self.mixing_logits, 1)
        return torch.sum(mixing_prob * self.dist.mean, 1)

    def log_prob(self, x):  # (B, ...)
        x = x.unsqueeze(1)  # (B, 1, ...)
        lp = self._component_log_prob(x)  # (B, K, ...)
        return torch.logsumexp(lp + self.mixing_log_prob(), 1)  # (B, ...)

    def _component_log_prob(self, x):
        lp = self.dist.log_prob(x)
        return lp

    def mode(self, straight_through_gradient=False, maximum=False):
        """Mode of the distribution.

        Args:
          straight_through_gradient: Boolean; if True, it uses the straight-through
            gradient estimator for the mode. Otherwise there is no gradient
            with respect to the mixing coefficients due to the `argmax` op.
          maximum: if True, attempt to return the highest-density mode.
        Returns:
          Mode.
        """
        dist_mode_value = self.dist.loc
        mixing_log_prob = self.mixing_log_prob()

        if maximum:
            mixing_log_prob += self._component_log_prob(dist_mode_value)

        n_dim = len(mixing_log_prob.shape)
        dim_order = list(range(n_dim - 1))
        dim_order.insert(1, n_dim - 1)
        mask = F.one_hot(mixing_log_prob.argmax(1),
                         mixing_log_prob.shape[1]).permute(*dim_order)

        if straight_through_gradient:
            soft_mask = F.softmax(mixing_log_prob, 1)
            mask = (mask - soft_mask).detach() + soft_mask

        return torch.sum(mask * dist_mode_value, 1)

    @classmethod
    def make_from_stats(cls, loc, scale, mixing_logits):
        """
        Creates a Gaussian mixture by loc(mean), scale(std) and mixing logits
        with K number of components.

        loc: tensor [B, K, ...] or broadcast-able
        scale: tensor [B, K, ...] or broadcast-able
        mixing_logits: tensor [B, K, ...]
        """
        return cls(Normal(loc, scale), mixing_logits)
