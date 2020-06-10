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
import pathlib
from argparse import ArgumentParser, Namespace

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from torch_scae import factory
from torch_scae.factory import make_config
from torch_scae.optimizers import RAdam, LookAhead


class SCAEMNIST(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.scae = factory.make_scae(hparams.model_config)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=128)
        # optimizer args
        parser.add_argument('--optimizer_type', type=str, default='RMSprop')
        parser.add_argument('--learning_rate', type=float, default=3e-5)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--look_ahead', action='store_true')
        parser.add_argument('--look_ahead_k', type=int, default=5)
        parser.add_argument('--look_ahead_alpha', type=float, default=0.5)
        parser.add_argument('--use_lr_scheduler', action='store_true')
        parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.997)

        return parser

    def forward(self, image):
        return self.scae(image=image)

    def configure_optimizers(self):
        eps = 1e-2 / float(self.hparams.batch_size) ** 2
        if self.hparams.optimizer_type == "RMSprop":
            optimizer = RMSprop(self.parameters(),
                                lr=self.hparams.learning_rate,
                                momentum=0.9,
                                eps=eps,
                                weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type == "RAdam":
            optimizer = RAdam(self.parameters(),
                              lr=self.hparams.learning_rate,
                              eps=eps,
                              weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_type == "Adam":
            optimizer = Adam(self.parameters(),
                             lr=self.hparams.learning_rate,
                             eps=eps,
                             weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError("Unknown optimizer type.")

        if self.hparams.look_ahead:
            optimizer = LookAhead(optimizer,
                                  k=self.hparams.look_ahead_k,
                                  alpha=self.hparams.look_ahead_alpha)

        if not self.hparams.use_lr_scheduler:
            return optimizer

        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.hparams.lr_scheduler_decay_rate)

        return [optimizer], [scheduler]

    def make_transforms(self):
        image_size = (28, 28)
        output_size = self.hparams.model_config['image_shape'][1:]

        if output_size[0] != image_size[0]:
            padding = tuple((output_size[i] - image_size[i]) // 2 for i in range(len(output_size)))
            translate = tuple(p / o for p, o in zip(padding, output_size))

            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
                torchvision.transforms.RandomAffine(degrees=0, translate=translate, fillcolor=0),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        return transforms

    def prepare_data(self):
        data_dir = self.hparams.data_dir

        # train and validation datasets
        mnist_train = MNIST(data_dir, train=True, download=True, transform=self.make_transforms())
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # test dataset
        mnist_test = MNIST(data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())

        # assign to use in data loaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        if not self.hparams.use_lr_scheduler:
            return

        current_lr = self.get_lr(self.trainer.optimizers[0])
        self.logger.experiment.add_scalar('learning_rate', current_lr, self.current_epoch)

    def on_batch_end(self) -> None:
        gc.collect()

    def training_step(self, batch, batch_idx):
        image, label = batch
        reconstruction_target = image

        res = self(image=image)
        loss, loss_info = self.scae.loss(res,
                                         reconstruction_target=reconstruction_target,
                                         label=label)
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
        n = min(self.hparams.batch_size, 8)
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


def train(model_params, **training_kwargs):
    model_config = make_config(**model_params)

    training_params = vars(parse_args())
    training_params.update(training_kwargs)

    hparams = dict(model_config=model_config)
    hparams.update(training_params)
    model = SCAEMNIST(Namespace(**hparams))

    if 'save_top_k' in training_params:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=training_params['save_top_k'])
        training_params.update(checkpoint_callback=checkpoint_callback)
        del training_params['save_top_k']

    trainer = Trainer(**training_params)
    trainer.fit(model)


def parse_args(argv=None):
    argv = argv or []

    parser = ArgumentParser()

    # add model specific args
    parser = SCAEMNIST.add_model_specific_args(parser)

    # add all the available trainer options to parser
    parser = Trainer.add_argparse_args(parser)

    # add other args
    parser.add_argument('--save_top_k', type=int, default=1)

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    import sys

    from torch_scae_experiments.mnist.hparams import model_params

    seed_everything(42)

    args = parse_args(sys.argv[1:])

    train(model_params, **vars(args))
