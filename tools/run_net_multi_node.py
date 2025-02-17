#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test
from train_net import train

import os


def parse_ip(s):
    s = s.split("-")
    s = [y for x in s for y in x.split("[") if y]
    s = [y for x in s for y in x.split(",") if y ]

    return ".".join(s[2:6])


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()

    if 'SLURM_STEP_NODELIST' in os.environ:
        args.init_method = "tcp://{}:{}".format(
            parse_ip(os.environ['SLURM_STEP_NODELIST']), "9999")

        print("Init Method: {}".format(args.init_method))

        cfg = load_config(args)
        cfg = assert_and_infer_cfg(cfg)

        cfg.NUM_SHARDS = int(os.environ['SLURM_NTASKS'])
        cfg.SHARD_ID = int(os.environ['SLURM_NODEID'])

        print(f'node id > {cfg.SHARD_ID}')
    else:
        cfg = load_config(args)
        cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
