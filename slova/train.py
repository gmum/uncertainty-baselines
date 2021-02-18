import os.path
import argparse

import numpy as np

from exp import experiment

parser = argparse.ArgumentParser(description='SLOVA CIFAR-10 ResNet-20 experiment')
parser.add_argument("--config", type=str, help="config json file",default='FLAGS.json')
parser.add_argument("--gpu_memory_fraction", type=float, help="gpu memory usage fraction",default=0.31)
args = parser.parse_args()

exp1 = experiment(exp_name='exp1',verbose=True,gpu_memory_fraction=args.gpu_memory_fraction)
exp1.load_flags(path=args.config)
exp1.prepare_exp()
exp1.run_exp()