import pathlib
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from torch_scae import factory
from torch_scae.configs import mnist_config
from torch_scae.general_utils import dict_from_module


class SCAEMNIST(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.scae = factory.make_scae(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        # optimizer args
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--use_lr_scheduler', action='store_true')
        parser.add_argument('--lr_schedule_patience', type=int, default=10)
        parser.add_argument('--lr_schedule_factor', type=float, default=0.1)

        return parser

    def forward(self, image):
        return self.scae(image=image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        if not self.hparams.use_lr_scheduler:
            return optimizer

        scheduler = ReduceLROnPlateau(optimizer,
                                      patience=self.hparams.lr_schedule_patience,
                                      factor=self.hparams.lr_schedule_factor)
        return [optimizer], [scheduler]

    def prepare_data(self):
        data_dir = self.hparams.data_dir

        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # download
        mnist_train = MNIST(data_dir, train=True, download=True, transform=transform)
        mnist_test = MNIST(data_dir, train=False, download=True, transform=transform)

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [59000, 1000])

        # assign to use in data loaders
        self.train_dataset = mnist_val
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

    def training_step(self, batch, batch_idx):
        image, label = batch
        reconstruction_target = image

        res = self(image=image)
        loss = self.scae.loss(res,
                              reconstruction_target=reconstruction_target,
                              label=label)
        accuracy = self.scae.calculate_accuracy(res, label)

        log = dict(
            loss=loss.detach(),
            accuracy=accuracy.detach(),
        )
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        reconstruction_target = image

        res = self(image=image)
        loss = self.scae.loss(res,
                              reconstruction_target=reconstruction_target,
                              label=label)
        accuracy = self.scae.calculate_accuracy(res, label)

        out = {'val_loss': loss, 'accuracy': accuracy}

        if batch_idx == 0:
            out['result'] = res
        return out

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        log = {'val_loss': avg_loss, 'val_accuracy': avg_acc}

        res = outputs[0]['result']

        recon = torch.cat([res.rec.pdf.mean.cpu(),
                           res.bottom_up_rec.pdf.mean.cpu(),
                           res.top_down_rec.pdf.mean.cpu()],
                          0)
        recon_grid = torchvision.utils.make_grid(
            recon,
            nrow=self.hparams.batch_size, pad_value=0, padding=1
        )
        self.logger.experiment.add_image('recons', recon_grid, 0)

        template_grid = torchvision.utils.make_grid(
            res.templates.cpu()[0],
            nrow=10, pad_value=0, padding=1
        )
        self.logger.experiment.add_image('templates', template_grid, 0)

        trs_template_grid = torchvision.utils.make_grid(
            res.transformed_templates.cpu()[0],
            nrow=10, pad_value=0, padding=1
        )
        self.logger.experiment.add_image('transformed_templates', trs_template_grid, 0)

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


def train(args):
    hparams = dict()
    hparams.update(dict_from_module(mnist_config))
    hparams.update(args.__dict__)

    model = SCAEMNIST(Namespace(**hparams))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


def parse(argv=None):
    parser = ArgumentParser()

    # add model specific args
    parser = SCAEMNIST.add_model_specific_args(parser)

    # add all the available trainer options to parser
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train(parse())
