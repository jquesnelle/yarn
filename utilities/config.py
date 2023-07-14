import argparse
import os
from distutils.util import strtobool
from pathlib import Path

import yaml


def _strtobool(x):
    return bool(strtobool(x))


def read_yamls(dir):
    conf = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            conf.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return conf


def rank_zero_info(msg) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank in (None, 0):
        print(msg)


def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="""
        Multiple configs can be passed to set different options.
        For example, run as:

           ./trainer_sft.py --configs default juweles

    """,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no_deepspeed", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from last saved checkpoint",
    )
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument(
        "--show_dataset_stats",
        action="store_true",
        help="Show dataset stats",
        default=False,
    )
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("configs/")

    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print(f'Error: Could not find the config "{e.args[0]}" in config.yaml')
        exit(1)

    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    if args.no_deepspeed:
        conf["deepspeed"] = None
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deeepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    return parser.parse_args(remaining)