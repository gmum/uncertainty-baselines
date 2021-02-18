# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for CIFAR-10 and CIFAR-100."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_cifar100_c_input_fn(corruption_name,
                             corruption_intensity,
                             batch_size,
                             use_bfloat16,
                             path,
                             drop_remainder=True,
                             normalize=True,
                             standarize=True):
  """Loads CIFAR-100-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  filename = path + '{0}-{1}.tfrecords'.format(corruption_name,
                                               corruption_intensity)
  def preprocess(serialized_example):
    """Preprocess a serialized example for CIFAR100-C."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.cast(tf.reshape(image, [32, 32, 3]), dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image = image / 255  # to convert into the [0, 1) range
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
      image = (image - mean) / std
    elif standarize:
      # Normalize per-image using mean/stddev computed across pixels.
      image = tf.image.per_image_standardization(image)
    label = tf.cast(features['label'], dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    dataset = tf.data.TFRecordDataset(filename, buffer_size=16 * 1000 * 1000)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


def load_cifar10_c_input_fn(corruption_name,
                            corruption_intensity,
                            batch_size,
                            use_bfloat16,
                            drop_remainder=True,
                            normalize=True):
  """Loads CIFAR-10-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  corruption = corruption_name + '_' + str(corruption_intensity)
  def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    dataset = tfds.load(name='cifar10_corrupted/{}'.format(corruption),
                        split=tfds.Split.TEST,
                        as_supervised=True)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


# TODO(ghassen,trandustin): Push this metadata upstream to TFDS.
def load_corrupted_test_info(corruptions_variant):
  """Loads information for CIFAR-10-C."""

#corruptions for cifar10-c defined in Hendrycks/Dietterich; https://arxiv.org/pdf/1903.12261.pdf
  hd_corruptions = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression']
    
#corruptions found within the ub framework    
  all_corruptions = hd_corruptions + ['spatter',
                                      'speckle_noise',
                                      'saturate',
                                      'gaussian_blur']

#corruptions used in figure from https://arxiv.org/abs/1906.02530
  googlefig_corruptions = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    #
    'zoom_blur',
    #
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    #
    'spatter',
    'speckle_noise',
    'saturate',
    'gaussian_blur']

  dictionary = {'glass_blur':'frosted_glass_blur','elastic_transform':'elastic'}
  googlefig_corruptions = [dictionary[c] if c in dictionary.keys() else c for c in googlefig_corruptions]    
    
  if corruptions_variant == 'hd':
    corruption_types = hd_corruptions
  elif corruptions_variant == 'all':
    corruption_types = all_corruptions
  elif corruptions_variant == 'googlefig':
    corruption_types = googlefig_corruptions
  else:
    corruption_types = all_corruptions
    
  max_intensity = 5
  return corruption_types, max_intensity


def load_input_fn(split,
                  batch_size,
                  name,
                  use_bfloat16,
                  normalize=True,
                  drop_remainder=True,
                  repeat=False,
                  proportion=1.0):
  """Loads CIFAR dataset for training or testing.

  Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    use_bfloat16: data type, bfloat16 precision or float32.
    normalize: Whether to apply mean-std normalization on features.
    drop_remainder: bool.
    repeat: bool.
    proportion: float, the proportion of dataset to be used.

  Returns:
    Input function which returns a locally-sharded dataset batch.
  """
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  ds_info = tfds.builder(name).info
  image_shape = ds_info.features['image'].shape
  dataset_size = ds_info.splits['train'].num_examples

  def preprocess(image, label):
    """Image preprocessing function."""
    if split == tfds.Split.TRAIN:
      image = tf.image.resize_with_crop_or_pad(
          image, image_shape[0] + 4, image_shape[1] + 4)
      image = tf.image.random_crop(image, image_shape)
      image = tf.image.random_flip_left_right(image)

    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    if proportion == 1.0:
      dataset = tfds.load(name, split=split, as_supervised=True)
    else:
      new_name = '{}:3.*.*'.format(name)
      if split == tfds.Split.TRAIN:
        # use round instead of floor to resolve bug when e.g. using
        # proportion = 1 - 0.8 = 0.19999999
        new_split = 'train[:{}%]'.format(round(100 * proportion))
      elif split == tfds.Split.VALIDATION:
        new_split = 'train[-{}%:]'.format(round(100 * proportion))
      elif split == tfds.Split.TEST:
        new_split = 'test[:{}%]'.format(round(100 * proportion))
      else:
        raise ValueError('Provide valid split.')
      dataset = tfds.load(new_name, split=new_split, as_supervised=True)
    if split == tfds.Split.TRAIN or repeat:
      dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn
