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
    
#     strategy = ub.strategy_utils.get_strategy(None, False)
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    dataset_builder = ub.datasets.Cifar10Dataset(batch_size=FLAGS['train_params']['batch_size'],
                                                 eval_batch_size=FLAGS['train_params']['eval_batch_size'],
                                                 validation_percent=FLAGS['validation_percent'])
    
#     train_dataset = ub.utils.build_dataset(dataset_builder, 
#                                            strategy, 
#                                            'train', 
#                                            as_tuple=True)
    
#     if FLAGS.augment_train:
#         FLAGS = _reconcile_flags_with_dataset(dataset_builder,FLAGS)
#         train_dataset = _augment_dataset(FLAGS,train_dataset)
        
#     val_dataset = ub.utils.build_dataset(dataset_builder, 
#                                          strategy, 
#                                          'validation', 
#                                          as_tuple=True)
#     test_dataset = ub.utils.build_dataset(dataset_builder, 
#                                           strategy, 
#                                           'test', 
#                                           as_tuple=True)    
    
#     return dataset_builder,train_dataset,val_dataset,test_dataset
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

# load cifar100 and svhn datasets
def _load_datasets_OOD(FLAGS):
    ood_datasets = {}
    ood_builders = {}
    strategy = ub.strategy_utils.get_strategy(None, False)
    
    dataset_builder = ub.datasets.SvhnDataset(batch_size=FLAGS['train_params']['batch_size'],
                                              eval_batch_size=FLAGS['train_params']['eval_batch_size'])
    train_dset = ub.utils.build_dataset(dataset_builder,strategy,'train',as_tuple=True)
    ood_datasets['svhn'] = train_dset
    ood_builders['svhn'] = dataset_builder
    
    dataset_builder = ub.datasets.Cifar100Dataset(batch_size=FLAGS['train_params']['batch_size'],
                                                  eval_batch_size=FLAGS['train_params']['eval_batch_size'])
    train_dset = ub.utils.build_dataset(dataset_builder,strategy,'train',as_tuple=True)
    ood_datasets['cifar100'] = train_dset
    ood_builders['cifar100'] = dataset_builder
    
    return ood_datasets, ood_builders


#load imagenet
def _load_datasets_imagenet_basic(FLAGS):
    
    #strategy = ub.strategy_utils.get_strategy(None, False)
#     strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    dataset_builder = ub.datasets.ImageNetDataset(batch_size=FLAGS['train_params']['batch_size'],
                                                  eval_batch_size=FLAGS['train_params']['eval_batch_size'],
                                                  validation_percent=FLAGS['validation_percent'],
                                                  shuffle_buffer_size=10000,
                                                  num_parallel_parser_calls=1)
    
    
#     train_dataset = ub.utils.build_dataset(dataset_builder, 
#                                            strategy, 
#                                            'train', 
#                                            as_tuple=True)
    
    
#     if FLAGS.augment_train:
#         FLAGS = _reconcile_flags_with_dataset(dataset_builder,FLAGS)
#         train_dataset = _augment_dataset(FLAGS,train_dataset)
        
#     val_dataset = ub.utils.build_dataset(dataset_builder, 
#                                          strategy, 
#                                          'validation', 
#                                          as_tuple=True)
#     test_dataset = ub.utils.build_dataset(dataset_builder, 
#                                           strategy, 
#                                           'test', 
#                                           as_tuple=True)    
    
#     return dataset_builder,train_dataset,val_dataset,test_dataset
    return dataset_builder


def _load_datasets_imagenet_corrupted(FLAGS):
#     train_dataset = dataset_utils.load_input_fn(split=tfds.Split.TRAIN,
#                                          name=FLAGS['dataset'],
#                                          batch_size=FLAGS['train_params']['batch_size'],
#                                          use_bfloat16=False)()
#     test_datasets = {'clean': dataset_utils.load_input_fn(split=tfds.Split.TEST,
#                                                   name=FLAGS['dataset'],
#                                                   batch_size=FLAGS['train_params']['batch_size'],
#                                                   use_bfloat16=False)()
#                     }
    
#     #load corrupted/modified cifar10 datasets
#     load_c_input_fn = dataset_utils.load_cifar10_c_input_fn
#     corruption_types, max_intensity = dataset_utils.load_corrupted_test_info(FLAGS.dataset)
#     for corruption in corruption_types:
#         for intensity in range(1, max_intensity + 1):
#             input_fn = load_c_input_fn(corruption_name=corruption,
#                                        corruption_intensity=intensity,
#                                        batch_size=FLAGS['train_params']['batch_size'],
#                                        use_bfloat16=False)
#             test_datasets['{0}_{1}'.format(corruption, intensity)] = input_fn()
#     return train_dataset, test_datasets
    return 0,0

def save_flags(path,FLAGS):
    with open(path,'w') as fp:
        json.dump(FLAGS,fp,indent=4)
        
def load_flags(path):
    assert os.path.exists(path), f'file {path} does not exist'
    with open('FLAGS.json','r') as fp:
        FLAGS_default = json.load(fp)
        
    with open(path,'r') as fp:
        FLAGS = json.load(fp)
        
    for name,val in FLAGS_default.items():
        if name in FLAGS.keys():
            FLAGS_default[name] = FLAGS[name]
    return AttrDict(FLAGS_default)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    def copy(self):
        return AttrDict(self.__dict__)

    
    
    
    
# TODO(trandustin): Refactor similar to CIFAR code.
class ImageNetLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, 
               steps_per_epoch, 
               minimum_learning_rate, 
               initial_learning_rate, 
               num_epochs,
               schedule):
    super(ImageNetLearningRateSchedule, self).__init__()
    self.num_epochs = num_epochs
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.minimum_learning_rate = minimum_learning_rate
    self.schedule = schedule

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = self.schedule[0]
    # Scale learning rate schedule by total epochs at vanilla settings.
    warmup_end_epoch = (warmup_end_epoch * self.num_epochs) // 90
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in self.schedule:
      start_epoch = (start_epoch * self.num_epochs) // 90
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self.initial_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'minimum_learning_rate': self.minimum_learning_rate,
        'initial_learning_rate': self.initial_learning_rate,
        'num_epochs': self.num_epochs,
        'schedule': self.schedule,
    }


class ImageNetWarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(self, lr_schedule, warmup_steps):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps

    """
    super(ImageNetWarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps

  def __call__(self, step):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      minimum_learning_rate = tf.convert_to_tensor(
          self._lr_schedule.minimum_learning_rate, name='minimum_learning_rate')
      initial_learning_rate = tf.convert_to_tensor(
          self._lr_schedule.initial_learning_rate, name='initial_learning_rate')
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = minimum_learning_rate + (initial_learning_rate - minimum_learning_rate) * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps,
                   lambda: warmup_lr,
                   lambda: lr)
    return lr

  def get_config(self):
    config = self._lr_schedule.get_config()
    config.update({
        'warmup_steps': self._warmup_steps,
    })
    return config


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