#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2022

@author: Martin Buechner, buechner@cs.uni-freiburg.de
"""
import argparse
from statistics import mean
from collections import defaultdict
import os
import json
import sys

import numpy as np

import torch
import torch_geometric
from tqdm import tqdm
import torch_geometric.data

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.render import TrackingRenderer

from batch_3dmot.utils.config import ParamLib


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!')
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Specific arguments (namespace: eval)
    parser.add_argument('--eval_config', type=str,
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--eval_set', type=str, help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--render_curves', type=int,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int,
                        help='Whether to print to stdout.')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.eval.overwrite(opt)

    # Create name for evaluation submissions JSON.
    sensors_used = ""
    for sensor, active in params.main.sensors_used.items():
        if active:
            sensors_used += str(sensor) + "_"

    results_path = os.path.join(params.paths.eval, "submission.json")
    print(results_path)

    # Specifying the evaluation configuration parameters.
    if params.eval.eval_config == 'tracking_nips_2019':
        eval_cfg = config_factory('tracking_nips_2019')
    else:
        with open(params.eval.eval_config, 'r') as _f:
            eval_cfg = TrackingConfig.deserialize(json.load(_f))
    print(vars(params.eval))
    # Running the evaluation using the specified evaluation config parameters.
    nusc_eval = TrackingEval(config=eval_cfg,
                             result_path=results_path,
                             eval_set=params.eval.eval_set,
                             output_dir=params.paths.eval,
                             nusc_version=params.main.version,
                             nusc_dataroot=params.paths.data,
                             verbose=bool(params.eval.verbose))


    nusc_eval.main(render_curves=bool(params.eval.render_curves))

    #renderer = TrackingRenderer(config['NUSCENES_PATH']+'tracking_eval')
    #renderer.render(events, timestamp, frame_gt, frame_pred)