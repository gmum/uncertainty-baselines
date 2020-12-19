import os
import argparse

from absl import app
from absl import flags
from absl import logging

import pandas as pd
import numpy as np

import itertools
import json

import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20 import create_model as resnet20
from resnet20 import configure_model as resnet20_configure
from resnet20 import load_model as resnet20_load

from func import load_datasets_basic, load_datasets_corrupted, add_dataset_flags
from func import load_datasets_OOD

from func import AttrDict, load_flags, save_flags


def prepare(FLAGS):

    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    callbacks, optimizer, loss_funcs, metrics = resnet20_configure(FLAGS)

    model = resnet20(batch_size=FLAGS.batch_size,
                                l2_weight=FLAGS.weight_decay,
                                certainty_variant=FLAGS.certainty_variant,
                                activation_type=FLAGS.activation,
                                model_variant=FLAGS.model_variant,
                                logit_variant=FLAGS.logit_variant
                              )

    model.compile(optimizer=optimizer,
                  loss=loss_funcs,
                  metrics=metrics)


    resnet20_load(model,FLAGS)
    
    return model

parser = argparse.ArgumentParser(description='Multihead CIFAR-10 ResNet-20 evaluate')
parser.add_argument("--config", type=str, help="config json file",default='FLAGS.json')
args = parser.parse_args()

FLAGS = load_flags(args.config)
# FLAGS = load_flags('FLAGS.json')

# FLAGS.output_dir = '1_vanilla_relu'
# FLAGS.model_file = '1_vanilla_relu/model.ckpt-250'

# FLAGS.activation = 'relu'
# FLAGS.certainty_variant = 'partial'
# FLAGS.model_variant = 'vanilla'
# FLAGS.logit_variant = 'affine'

model = prepare(FLAGS)

dataset_builder,train_dataset,val_dataset,test_dataset = load_datasets_basic(FLAGS)
FLAGS = add_dataset_flags(dataset_builder,FLAGS)
#df = pd.DataFrame(columns=['dataset','metric'])

# metrics_vals = model.evaluate(test_dataset)
# metrics_names = model.metrics_names 

model.evaluate(test_dataset)