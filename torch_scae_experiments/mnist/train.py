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

import pathlib

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.backends import cudnn

from torch_scae_experiments.mnist.experiment import MNISTExperiment


def train(cfg: DictConfig):
    # For reproducibility
    seed_everything(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    experiment = MNISTExperiment(cfg)

    if 'save_top_k' in cfg.trainer:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=cfg.trainer.save_top_k)
        cfg.trainer.update(checkpoint_callback=checkpoint_callback)
        del cfg.trainer['save_top_k']

    trainer = Trainer(**cfg.trainer)
    trainer.fit(experiment)


@hydra.main(config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
            config_name="config")
def main(cfg) -> None:
    print(cfg.pretty())
    train(cfg)


if __name__ == "__main__":
    print(__file__)
    main()
