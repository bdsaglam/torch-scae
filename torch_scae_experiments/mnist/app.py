import hydra
import pathlib

from torch_scae_experiments.mnist.train import train


@hydra.main(config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
            config_name="config")
def main(cfg) -> None:
    print(cfg.pretty())
    train(cfg)


if __name__ == "__main__":
    print(__file__)
    main()
