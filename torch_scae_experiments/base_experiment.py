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

import gc
import math
from abc import ABC

import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader

from torch_scae import factory
from torch_scae.optimizers import RAdam, LookAhead


class BaseExperiment(LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hparams = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = cfg

        self.scae = factory.make_scae(
            OmegaConf.to_container(cfg.model, resolve=True))

    def forward(self, image):
        return self.scae(image=image)

    def configure_optimizers(self):
        lr = self.cfg.optimizer.learning_rate
        weight_decay = self.cfg.optimizer.weight_decay
        eps = 1e-2 / float(self.cfg.data_loader.batch_size) ** 2
        if self.cfg.optimizer.type == "RMSprop":
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
                                  k=self.cfg.optimizer.look_ahead_k,
                                  alpha=self.cfg.optimizer.look_ahead_alpha)

        if not self.cfg.lr_scheduler.active:
            return optimizer
        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.cfg.lr_scheduler.decay_rate)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          num_workers=self.cfg.data_loader.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          num_workers=self.cfg.data_loader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.data_loader.batch_size,
                          num_workers=self.cfg.data_loader.num_workers)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        if not self.cfg.lr_scheduler.active:
            return

        current_lr = self.get_lr(self.trainer.optimizers[0])
        self.logger.experiment.add_scalar(
            'learning_rate', current_lr, self.current_epoch)

    def on_batch_end(self) -> None:
        gc.collect()

    def training_step(self, batch, batch_idx):
        image, label = batch
        reconstruction_target = image

        res = self(image=image)
        loss, loss_info = self.scae.loss(
            res,
            reconstruction_target=reconstruction_target,
            label=label
        )
        accuracy = self.scae.calculate_accuracy(res, label)

        log = dict(
            loss=loss.detach(),
            accuracy=accuracy.detach(),
            **loss_info
        )
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        reconstruction_target = image

        res = self(image=image)
        loss, loss_info = self.scae.loss(res,
                                         reconstruction_target=reconstruction_target,
                                         label=label)
        accuracy = self.scae.calculate_accuracy(res, label)

        out = {'val_loss': loss, 'accuracy': accuracy}

        if batch_idx == 0:
            res.image = image
            out['result'] = res
        return out

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'val_accuracy': avg_acc}

        res = outputs[0]['result']

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
        self.logger.experiment.add_image('recons', rg, self.current_epoch)

        # log raw templates
        templates = res.templates.cpu()[0]
        n_templates = templates.shape[0]
        nrow = int(math.sqrt(n_templates))
        tg = torchvision.utils.make_grid(
            templates,
            nrow=nrow, pad_value=0, padding=1
        )
        self.logger.experiment.add_image('templates', tg, self.current_epoch)

        # log transformed templates
        ttg = torchvision.utils.make_grid(
            res.transformed_templates.cpu()[0],
            nrow=nrow, pad_value=0, padding=1
        )
        self.logger.experiment.add_image(
            'transformed_templates', ttg, self.current_epoch)

        return {'val_loss': avg_loss, 'log': log}

    def test_step(self, batch, batch_idx):
        image, label = batch
        reconstruction_target = image

        res = self(image=image)
        loss = self.scae.loss(res,
                              reconstruction_target=reconstruction_target,
                              label=label)
        accuracy = self.scae.calculate_accuracy(res, label)

        return {'test_loss': loss, 'accuracy': accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        log = {'test_loss': avg_loss, 'test_accuracy': avg_acc}
        return {'test_loss': avg_loss, 'log': log}
