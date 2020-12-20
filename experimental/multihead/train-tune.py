import os.path
import argparse

from absl import app
from absl import flags
from absl import logging

#import pandas as pd
import numpy as np

import itertools
import json

import tensorflow as tf
import kerastuner
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20 import create_model as resnet20
from resnet20 import configure_model as resnet20_configure

from func import load_datasets_basic, load_datasets_corrupted, add_dataset_flags
from func import AttrDict, load_flags, save_flags

parser = argparse.ArgumentParser(description='Multihead CIFAR-10 ResNet-20 train-tune')
parser.add_argument("--config", type=str, help="config json file",default='FLAGS.json')
args = parser.parse_args()

FLAGS = load_flags(args.config)

# FLAGS = load_flags('FLAGS.json')

# FLAGS.output_dir = '5_1vsall_dm_relu' #not used here
# FLAGS.model_file = '5_1vsall_dm_relu/model.ckpt-250' #not used here

# FLAGS.optimizer = 'adam'
# FLAGS.activation = 'relu'
# FLAGS.certainty_variant = 'partial'
# FLAGS.model_variant = '1vsall'
# FLAGS.logit_variant = 'dm'

# FLAGS.tune_hyperparams = True
# FLAGS.batch_size = 512
# FLAGS.epochs= 300 #not used here

# logging.info('Multihead CIFAR-10 ResNet-20 tune hyperparameters')

def prepare(FLAGS):
    
    def build_model(hp):
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
        
        optimizer = optimizer(hp) if FLAGS.tune_hyperparams else optimizer
            
        model.compile(optimizer=optimizer,
                  loss=loss_funcs,
                  metrics=metrics)


        #resnet20_load(model,FLAGS)

        return model

    return build_model

if FLAGS.tune_by == 'acc':
    objective = kerastuner.Objective("val_probs_acc", direction="max")
elif FLAGS.tune_by == 'ece':
    objective = kerastuner.Objective("val_probs_ece", direction="min")
else:
    raise ValueError(f'unknown tune_by={FLAGS.tune_by}')
    
tuner = kerastuner.tuners.RandomSearch(
    prepare(FLAGS),
    objective=objective,
    max_trials=FLAGS.tune_num_trials,
    executions_per_trial=1,
    directory=FLAGS.tuner_dir,
    project_name=FLAGS.tuner_subdir)


dataset_builder,train_dataset,val_dataset,test_dataset = load_datasets_basic(FLAGS)

f = FLAGS
f.batch_size = FLAGS.tune_batch_size
FLAGS = add_dataset_flags(dataset_builder,f)

callbacks, _, _, _ = resnet20_configure(FLAGS)

tuner.search(train_dataset,
             batch_size=FLAGS.tune_batch_size,
             epochs=FLAGS.tune_epochs,
             validation_data=val_dataset,
             callbacks=callbacks)

tuner.results_summary()
