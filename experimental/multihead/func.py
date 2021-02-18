#import os.path

#from absl import app
#from absl import flags
#from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um
import tensorflow_datasets as tfds

import json
import os.path

import dataset_utils #from baselines/cifar


def _reconcile_flags_with_dataset(dataset_builder,FLAGS):

    bs = FLAGS['train_params']['batch_size']
    ebs = FLAGS['train_params']['eval_batch_size']

    FLAGS['train_params']['steps_per_epoch'] = dataset_builder.info['num_train_examples'] // bs
    FLAGS['train_params']['validation_steps'] = dataset_builder.info['num_validation_examples'] // ebs
    FLAGS['train_params']['test_steps'] = dataset_builder.info['num_test_examples'] // ebs
    #FLAGS.no_classes = 10 # awful but no way to infer from dataset...
    return FLAGS

def _augment_dataset(FLAGS,dataset):
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
    
    #del xs,ys
    for i,ds in enumerate(dataset):
        x,y = ds
        if i>FLAGS['train_params']['steps_per_epoch']: break
        if 'xs' not in locals() and 'ys' not in locals():
            xs = x
            ys = y
        else:
            xs = tf.concat([xs,x],axis=0)
            ys = tf.concat([ys,y],axis=0)
    
    datagen.fit(xs)
    return datagen.flow(x=xs,y=ys,batch_size=FLAGS['train_params']['batch_size'])

def _load_datasets_cifar10_basic(FLAGS):
    
    dataset_builder = ub.datasets.Cifar10Dataset(batch_size=FLAGS['train_params']['batch_size'],
                                                 eval_batch_size=FLAGS['train_params']['eval_batch_size'],
                                                 validation_percent=FLAGS['validation_percent'])
    
    return dataset_builder

def _load_datasets_cifar10_corrupted(FLAGS):
    train_dataset = dataset_utils.load_input_fn(split=tfds.Split.TRAIN,
                                         name=FLAGS['dataset'],
                                         batch_size=FLAGS['train_params']['batch_size'],
                                         use_bfloat16=False)()
    test_datasets = {'clean': dataset_utils.load_input_fn(split=tfds.Split.TEST,
                                                  name=FLAGS['dataset'],
                                                  batch_size=FLAGS['train_params']['batch_size'],
                                                  use_bfloat16=False)()
                    }
    
    #load corrupted/modified cifar10 datasets
    load_c_input_fn = dataset_utils.load_cifar10_c_input_fn
    corruption_types, max_intensity = dataset_utils.load_corrupted_test_info(FLAGS.dataset)
    for corruption in corruption_types:
        for intensity in range(1, max_intensity + 1):
            input_fn = load_c_input_fn(corruption_name=corruption,
                                       corruption_intensity=intensity,
                                       batch_size=FLAGS['train_params']['batch_size'],
                                       use_bfloat16=False)
            test_datasets['{0}_{1}'.format(corruption, intensity)] = input_fn()
    return train_dataset, test_datasets


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    def copy(self):
        return AttrDict(self.__dict__)

# hack enabling use of save_weights_only = False
class FixedModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model
        
def update_dict(dict0,delta_dict):
    dict1 = dict0.copy()
    for key,val in dict0.items():
        if type(val)==type(dict()):
            if key in delta_dict.keys():
                dict1[key] = update_dict(val,delta_dict[key])
        else:
            if key in delta_dict.keys():
                dict1[key] = delta_dict[key]
    return dict1