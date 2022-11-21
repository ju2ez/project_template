#!/usr/bin/env python
import logging
import sys
import yaml
from pathlib import Path

import click
from IPython.core import ultratb

from ray import tune

import ray_tune_template
from ray_tune_template.train import run_ray_experiment

from configs.ray_tune_config import get_example_search_space

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)
# turn UserWarning messages to errors to find the actual cause
# import warnings
# warnings.simplefilter("error")

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True),
    help="path to config file",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.version_option(ray_tune_template.__version__)
def main(cfg_path: Path, log_level: int):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # YOUR CODE GOES HERE! Keep the main functionality in src/ray_tune_template
    try:
        with open(cfg_path, "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            logging.info("read the config successfully")
    except FileNotFoundError:
        logging.error("config file could not be found")

    search_space = get_example_search_space()

    run_ray_experiment(config, search_space)


if __name__ == "__main__":
    main()
