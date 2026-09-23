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

import math
from abc import ABC

import torch
import torchvision
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader

from torch_scae import factory
from torch_scae.optimizers import RAdam, LookAhead, TFRMSprop


class BaseExperiment(LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        self.scae = factory.make_scae(
            OmegaConf.to_container(cfg.model, resolve=True))
        if cfg.get('compile', False):
            self.scae.compile()
        self._val_first_batch = None

    def forward(self, image):
        return self.scae(image=image)

    def configure_optimizers(self):
        lr = self.cfg.optimizer.learning_rate
        weight_decay = self.cfg.optimizer.weight_decay
        eps = 1e-2 / float(self.cfg.data_loader.batch_size) ** 2
        if self.cfg.optimizer.type == "TFRMSprop":
            optimizer = TFRMSprop(self.parameters(),
                                  lr=lr,
                                  momentum=self.cfg.optimizer.momentum,
                                  eps=eps)
        elif self.cfg.optimizer.type == "RMSprop":
            optimizer = RMSprop(self.parameters(),
                                lr=lr,
                                momentum=self.cfg.optimizer.momentum,
                                eps=eps,
                                weight_decay=weight_decay)
        elif self.cfg.optimizer.type == "RAdam":
            optimizer = RAdam(self.parameters(),
                              lr=lr,
                              eps=eps,
                              weight_decay=weight_decay)
        elif self.cfg.optimizer.type == "Adam":
            optimizer = Adam(self.parameters(),
                             lr=lr,
                             eps=eps,
                             weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer type.")

        if self.cfg.meta_optimizer.look_ahead:
            optimizer = LookAhead(optimizer,
                                  k=self.cfg.meta_optimizer.look_ahead_k,
                                  alpha=self.cfg.meta_optimizer.look_ahead_alpha)

        if not self.cfg.lr_scheduler.active:
            return optimizer
        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.cfg.lr_scheduler.decay_rate)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.cfg.data_loader.num_workers,
                          persistent_workers=self.cfg.data_loader.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          num_workers=self.cfg.data_loader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          num_workers=self.cfg.data_loader.num_workers)

    def _shared_step(self, batch):
        image, label = batch
        res = self(image=image)
        loss, loss_info = self.scae.loss(res,
                                         reconstruction_target=image,
                                         label=label)
        accuracy = self.scae.calculate_accuracy(res, label)
        return res, loss, loss_info, accuracy

    def training_step(self, batch, batch_idx):
        _, loss, loss_info, accuracy = self._shared_step(batch)
        log = {f'train/{k}': v.detach() for k, v in loss_info.items()}
        log.update({'train/loss': loss.detach(), 'train/accuracy': accuracy})
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        res, loss, _, accuracy = self._shared_step(batch)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy})
        if batch_idx == 0:
            res.image = batch[0]
            self._val_first_batch = res

    def on_validation_epoch_end(self):
        res = self._val_first_batch
        if res is None or not hasattr(self.logger, 'experiment'):
            return
        self._val_first_batch = None
        step = self.global_step

        # log image reconstructions
        n = min(self.cfg.data_loader.batch_size, 8)
        recons = [res.image.cpu()[:n], res.rec.pdf.mode().cpu()[:n]]
        if res.get('bottom_up_rec'):
            recons.append(res.bottom_up_rec.pdf.mode().cpu()[:n])
        if res.get('top_down_rec'):
            recons.append(res.top_down_rec.pdf.mode().cpu()[:n])
        recon = torch.cat(recons, 0)
        rg = torchvision.utils.make_grid(
            recon,
            nrow=n, pad_value=0, padding=1
        )
        self.logger.experiment.add_image('recons', rg, step)

        # log raw templates
        templates = res.templates.cpu()[0]
        n_templates = templates.shape[0]
        nrow = int(math.sqrt(n_templates))
        tg = torchvision.utils.make_grid(
            templates,
            nrow=nrow, pad_value=0, padding=1
        )
        self.logger.experiment.add_image('templates', tg, step)

        # log transformed templates
        ttg = torchvision.utils.make_grid(
            res.transformed_templates.cpu()[0],
            nrow=nrow, pad_value=0, padding=1
        )
        self.logger.experiment.add_image(
            'transformed_templates', ttg, step)

    def test_step(self, batch, batch_idx):
        _, loss, _, accuracy = self._shared_step(batch)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy})
