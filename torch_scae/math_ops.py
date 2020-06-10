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


def log_safe(tensor, eps=1e-16):
    is_zero = tensor < eps
    tensor = torch.where(is_zero, torch.ones_like(tensor), tensor)
    tensor = torch.where(is_zero, torch.zeros_like(tensor) - 1e8, torch.log(tensor))
    return tensor


def cross_entropy_safe(true_probs, probs, dim=-1):
    return torch.mean(-torch.sum(true_probs * log_safe(probs), dim=dim))


def normalize(tensor, dim):
    return tensor / (torch.sum(tensor, dim, keepdim=True) + 1e-8)


def l2_loss(tensor):
    return torch.sum(tensor ** 2) / 2
