import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import itertools
import argparse

import func_plot

parser = argparse.ArgumentParser(description='Multihead CIFAR-10 ResNet-20 plot_figs')
parser.add_argument("--datadir", type=str, help="csv metrics directory",default='metrics_summary')
parser.add_argument("--plotdir", type=str, help="figure directory",default='metrics_summary/figs')
args = parser.parse_args()


