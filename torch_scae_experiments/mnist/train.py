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
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.backends import cudnn

from torch_scae_experiments.mnist.experiment import MNISTExperiment


def train(cfg: DictConfig):
    # For reproducibility
    seed_everything(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    experiment = MNISTExperiment(cfg)

    checkpoint_callback = ModelCheckpoint(save_last=True)
    trainer = Trainer(callbacks=[checkpoint_callback],
                      **OmegaConf.to_container(cfg.trainer))
    trainer.fit(experiment)
    trainer.test(experiment)


@hydra.main(config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
            config_name="config", version_base=None)
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    print(__file__)
    main()
