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

import torch
from hydra.utils import to_absolute_path
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from torch_scae_experiments.base_experiment import BaseExperiment


class MNISTExperiment(BaseExperiment):
    def make_transforms(self):
        image_size = (28, 28)
        model_input_size = tuple(self.cfg.model.image_shape[1:])

        if image_size != model_input_size:
            padding = tuple((model_input_size[i] - image_size[i]) // 2
                            for i in range(len(model_input_size)))
            translate = tuple(p / o for p, o in zip(padding, model_input_size))

            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
                torchvision.transforms.RandomAffine(degrees=0, translate=translate, fill=0),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        return transforms

    def setup(self, stage=None):
        data_dir = to_absolute_path(self.cfg.dataset.directory)

        # train and validation datasets
        transform = self.make_transforms()
        mnist_train = MNIST(data_dir, train=True, download=True, transform=transform)
        mnist_train, mnist_val = random_split(
            mnist_train, [55000, 5000],
            generator=torch.Generator().manual_seed(0))

        # test dataset, placed on the canvas the same way as the training set
        mnist_test = MNIST(data_dir, train=False, download=True, transform=transform)

        # assign to use in data loaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test
