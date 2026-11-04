import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="vopd")
    args, overrides = parser.parse_known_args()
    directory = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
