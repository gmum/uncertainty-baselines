
import os.path
from absl import app
from absl import flags
from absl import logging
from typing import Any, Dict

import tensorflow as tf
import tensorflow.keras as keras

import uncertainty_baselines as ub
import uncertainty_metrics as um

import numpy as np
# import sklearn.isotonic
# import sklearn.neural_network

from metrics import BrierScore
from metrics import MMC
from metrics import nll


def one_vs_all_loss_fn(dm_alpha: float = 1., from_logits: bool = True,reduction = tf.keras.losses.Reduction.SUM,one_hot=False):
    """Requires from_logits=True to calculate correctly."""
    if not from_logits:
        raise ValueError('One-vs-all loss requires inputs to the '
                         'loss function to be logits, not probabilities.')

    def one_vs_all_loss(labels: tf.Tensor, logits: tf.Tensor,reduction=reduction):
        r"""Implements the one-vs-all loss function.

        As implemented in https://arxiv.org/abs/1709.08716, multiplies the output
        logits by dm_alpha (if using a distance-based formulation) before taking K
        independent sigmoid operations of each class logit, and then calculating the
        sum of the log-loss across classes. The loss function is calculated from the
        K sigmoided logits as follows -

        \mathcal{L} = \sum_{i=1}^{K} -\mathbb{I}(y = i) \log p(\hat{y}^{(i)} | x)
        -\mathbb{I} (y \neq i) \log (1 - p(\hat{y}^{(i)} | x))

        Args:
          labels: Integer Tensor of dense labels, shape [batch_size].
          logits: Tensor of shape [batch_size, num_classes].

        Returns:
          A scalar containing the mean over the batch for one-vs-all loss.
        """
        #eps = tf.keras.backend.epsilon()
        eps = 1e-6
        #eps = 1e-10
        logits = logits * dm_alpha
        n_classes = tf.cast(logits.shape[1], tf.float32)

        if one_hot:
            labels = tf.argmax(labels, axis=-1) #decode one_hot
        
        one_vs_all_probs = tf.math.sigmoid(logits)
        labels = tf.cast(tf.squeeze(labels), tf.int32)
        row_ids = tf.range(tf.shape(one_vs_all_probs)[0], dtype=tf.int32)
        idx = tf.stack([row_ids, labels], axis=1)

        # Shape of class_probs is [batch_size,].
        class_probs = tf.gather_nd(one_vs_all_probs, idx)
        
        s1 = tf.math.log(class_probs + eps)
        s2 = tf.reduce_sum(tf.math.log(1. - one_vs_all_probs + eps),axis=-1)
        s3 = - tf.math.log(1. - class_probs + eps)
        
        loss = -s1 - s2 - s3
        if reduction == tf.keras.losses.Reduction.NONE:
            return loss
          
        elif reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_mean(loss)

    return one_vs_all_loss

# def one_vs_all_loss_fn(dm_alpha: float = 1., from_logits: bool = True):
#     """Requires from_logits=True to calculate correctly."""
#     if not from_logits:
#         raise ValueError('One-vs-all loss requires inputs to the '
#                          'loss function to be logits, not probabilities.')

#     def one_vs_all_loss(labels: tf.Tensor, logits: tf.Tensor):
#         r"""Implements the one-vs-all loss function.

#         As implemented in https://arxiv.org/abs/1709.08716, multiplies the output
#         logits by dm_alpha (if using a distance-based formulation) before taking K
#         independent sigmoid operations of each class logit, and then calculating the
#         sum of the log-loss across classes. The loss function is calculated from the
#         K sigmoided logits as follows -

#         \mathcal{L} = \sum_{i=1}^{K} -\mathbb{I}(y = i) \log p(\hat{y}^{(i)} | x)
#         -\mathbb{I} (y \neq i) \log (1 - p(\hat{y}^{(i)} | x))

#         Args:
#           labels: Integer Tensor of dense labels, shape [batch_size].
#           logits: Tensor of shape [batch_size, num_classes].

#         Returns:
#           A scalar containing the mean over the batch for one-vs-all loss.
#         """
#         #eps = tf.keras.backend.epsilon()
#         eps = 1e-6
#         #eps = 1e-10
#         logits = logits * dm_alpha
#         n_classes = tf.cast(logits.shape[1], tf.float32)

#         one_vs_all_probs = tf.math.sigmoid(logits)
#         labels = tf.cast(tf.squeeze(labels), tf.int32)
#         row_ids = tf.range(tf.shape(one_vs_all_probs)[0], dtype=tf.int32)
#         idx = tf.stack([row_ids, labels], axis=1)

#         # Shape of class_probs is [batch_size,].
#         class_probs = tf.gather_nd(one_vs_all_probs, idx)

#         loss = (
#             tf.reduce_mean(tf.math.log(class_probs + eps)) +
#             n_classes * tf.reduce_mean(tf.math.log(1. - one_vs_all_probs + eps)) -
#             tf.reduce_mean(tf.math.log(1. - class_probs + eps)))

#         return -loss

#     return one_vs_all_loss

def _calc_certs(probs: tf.Tensor,
                certainty_variant: str = 'partial') -> tf.Tensor:

    #form Ci's
    #probs = tf.math.sigmoid(logits)
    probs_comp = 1-probs
    K = probs.shape[1]
    cert_list = []

    for i in range(K):
        proj_vec = np.zeros(K)
        proj_vec[i]=1
        proj_mat = np.outer(proj_vec,proj_vec)
        proj_mat_comp = np.identity(K)-np.outer(proj_vec,proj_vec)
        tproj_mat = tf.constant(proj_mat,dtype=tf.float32)
        tproj_mat_comp = tf.constant(proj_mat_comp,dtype=tf.float32)
        out = tf.tensordot(probs,tproj_mat,axes=1) + tf.tensordot(probs_comp,tproj_mat_comp,axes=1)
        cert_list+=[tf.reduce_prod(out,axis=1)]

    if certainty_variant == 'partial':
        certs = tf.stack(cert_list,axis=1,name='certs')

    elif certainty_variant == 'total':
        certs = tf.stack(cert_list,axis=1)
        certs_argmax = tf.one_hot(tf.argmax(certs,axis=1),depth=K)
        certs_reduce = tf.tile(tf.reduce_sum(certs,axis=1,keepdims=True),[1,K])
        certs = tf.math.multiply(certs_argmax,certs_reduce)

    elif certainty_variant == 'normalized':
        certs = tf.stack(cert_list,axis=1)
        certs_norm = tf.tile(tf.reduce_sum(certs,axis=1,keepdims=True),[1,K])
        certs = tf.math.divide(certs,certs_norm)

    else:
        raise ValueError(f'unknown certainty_variant={certainty_variant}')   

    #certs.name = 'certs'
    return certs

def _calc_logits_from_certs(certs: tf.Tensor, 
                            eps: float = 1e-6) -> tf.Tensor:
    #logits_from_certs
    K = certs.shape[1]

    logcerts = tf.math.log(certs+eps)
    rs = tf.tile(logcerts[:,:1],[1,K])-logcerts #set first logit to zero (an arbitrary choice)
    logits_from_certs = -rs    

    return logits_from_certs




def _activ(activation_type: str = 'relu'):
    activation = {'relu': tf.keras.layers.ReLU(), 'sin': tf.keras.backend.sin}
    if activation_type in activation.keys():
        return activation[activation_type]
    else:
        return activation['relu']

class resnetLayer(tf.keras.layers.Layer):
    def __init__(self,
        num_filters: int = 16,
        kernel_size: int = 3,
        strides: int = 1,
        use_activation: bool = True,
        activation_type: str = 'relu', #relu or sin!
        use_norm: bool = True,
        l2_weight: float = 1e-4):
        
        super(resnetLayer,self).__init__()
        
        self.use_activation = use_activation
        self.use_norm = use_norm
        
        self.kernel_regularizer = None
        if l2_weight:
            self.kernel_regularizer = tf.keras.regularizers.l2(l2_weight)    
#         print(f'    resnetLayer num_filters={num_filters}, strides={strides}, kernel_size={kernel_size}')
        self.conv_layer = tf.keras.layers.Conv2D(num_filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 kernel_initializer='he_normal',
                                                 kernel_regularizer=self.kernel_regularizer)
        
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = _activ(activation_type)
        
    def call(self,
             inputs: tf.Tensor) -> tf.Tensor:

        x = self.conv_layer(inputs)
        if self.use_norm:
            x = self.batch_norm(x)
        if self.use_activation:
            x = self.activation(x)
            
        return x       
    
class resnet20Block(tf.keras.layers.Layer):
    
    def __init__(self,
                stack: int,
                res_block: int,
                num_filters: int = 16,
                activation_type: str = 'relu', #relu or sin!
                l2_weight: float = 1e-4):
        
        super(resnet20Block,self).__init__()

        self.stack = stack
        self.res_block = res_block
        self.num_filters = num_filters
        self.activation_type = activation_type
        self.l2_weight = l2_weight
        
#         layers= 3 if self.stack > 0 and self.res_block == 0 else 2
#         print(f'resnetBlock: stack={stack}, res_block={res_block}, filters={num_filters}, number_layers={layers}')
        
        strides = 1
        if self.stack > 0 and self.res_block == 0:
            strides = 2

        self.l_1 = resnetLayer(num_filters=self.num_filters,
                               strides=strides,
                               l2_weight=self.l2_weight,
                               activation_type=self.activation_type)
        
        self.l_2 = resnetLayer(num_filters=self.num_filters,
                               l2_weight=self.l2_weight,
                               use_activation=False)

        self.l_3 = resnetLayer(num_filters=self.num_filters,
                               kernel_size=1,
                               strides=strides,
                               l2_weight=self.l2_weight,
                               use_activation=False,
                               use_norm=False)

        self.l_add = tf.keras.layers.Add()
        self.l_activation = _activ(self.activation_type)
        
    def call(self,inputs: tf.Tensor) -> tf.Tensor:
        y = self.l_1(inputs)
        y = self.l_2(y)
        x = self.l_3(inputs) if self.stack > 0 and self.res_block == 0 else inputs
        x = self.l_add([x, y])
        x = self.l_activation(x)
        return x
    
class DMLayer(tf.keras.layers.Layer):
  def __init__(self, units: int = 10, **kwargs):
    super(DMLayer, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(name='DMLayer_weight',
          shape=(input_shape[-1], self.units),
          initializer="he_normal",
          trainable=True)

  def get_config(self):
    return {"units": self.units}
    
  def call(self, inputs):

    #tf.tile(inputs)
#     w_tiled = tf.tile(tf.reshape(self.w,shape=(1,)+self.w.shape),[inputs.shape[0],1,1])
#     inputs_tiled = tf.tile(tf.reshape(inputs,shape=inputs.shape+(1,)),[1,1,self.units])
#     out = tf.math.sqrt(tf.math.reduce_euclidean_norm(inputs_tiled-w_tiled,axis=1))

#     a = tf.random.normal(shape=(128,64))
#     b = tf.random.normal(shape=(64,10))
    be=tf.expand_dims(self.w,0)
    ae=tf.expand_dims(inputs,-1)
    out = -tf.math.sqrt(tf.math.reduce_euclidean_norm(be-ae,axis=1))
    
    return out



class resnet20(tf.keras.Model):
    def __init__(self,
                 batch_size: int = 128,
                 l2_weight: float = 0.0,
                 activation_type: str = 'relu', #relu or sin
                 certainty_variant: str = 'partial', #partial, total or normalized
                 model_variant: str = '1vsall', #1vsall or vanilla
                 logit_variant: str = 'affine', #affine or dm
                 **params):
        super(resnet20,self).__init__()
        
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        
        if activation_type in ['sin','relu']:
            self.activation_type = activation_type
        else:
            raise ValueError(f'unknown activation_type={activation_type}')
        
        if certainty_variant in ['partial','total','normalized']:
            self.certainty_variant = certainty_variant
        else:
            raise ValueError(f'unknown certainty_variant={certainty_variant}')
        
        if model_variant in ['1vsall','vanilla']:
            self.model_variant = model_variant
        else:
            raise ValueError(f'unknown model_variant={model_variant}')
                        
        if logit_variant in ['affine','dm']:
            self.logit_variant = logit_variant
        else:
            raise ValueError(f'unknown logit_variant={logit_variant}')
        
        self.depth = 20
        self.num_res_blocks = int((self.depth - 2) / 6)
        num_filters = 16
        
        
        self.layer_init_1 = tf.keras.layers.InputLayer(input_shape=(32, 32, 3),
                                                       batch_size=self.batch_size)
        
        self.layer_init_2 = resnetLayer(num_filters=num_filters,
                                        l2_weight=self.l2_weight,
                                        activation_type=self.activation_type)
        
        self.res_blocks = [[0 for stack in range(3)] for res_block in range(self.num_res_blocks)]

        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                self.res_blocks[stack][res_block] = resnet20Block(stack = stack,
                                                                  res_block = res_block,
                                                                  num_filters = num_filters,
                                                                  activation_type = self.activation_type,
                                                                  l2_weight = self.l2_weight)
            num_filters *= 2
        
        self.layer_final_1 = tf.keras.layers.AveragePooling2D(pool_size=8)
        self.layer_final_2 = tf.keras.layers.Flatten()
        
        if self.logit_variant == 'dm':
            self.layer_final_3 = DMLayer(units=10)
        elif self.logit_variant == 'affine':
            self.layer_final_3 = tf.keras.layers.Dense(10, kernel_initializer='he_normal')
        else:
            raise ValueError(f'unknown logit_variant={self.logit_variant}')   
    
    def call(self, 
             inputs: tf.Tensor, 
             trainable: bool = False) -> dict:

        x = self.layer_init_1(inputs)
        x = self.layer_init_2(x)
        
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                x = self.res_blocks[stack][res_block](x)

        x = self.layer_final_1(x)
        x = self.layer_final_2(x)
        
        logits = self.layer_final_3(x)
        
        if self.model_variant == '1vsall':
            probs = tf.math.sigmoid(logits)
            if self.logit_variant == 'dm':
                probs = 2*probs
        elif self.model_variant == 'vanilla':
            probs = tf.math.softmax(logits,axis=-1)
        else:
            raise ValueError(f'unknown model_variant={self.model_variant}')
        
        certs = _calc_certs(probs, certainty_variant = self.certainty_variant)
        logits_from_certs = _calc_logits_from_certs(certs = certs)
        
        return {'logits':logits,'probs':probs,'certs':certs,'logits_from_certs':logits_from_certs}

        
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
class resnet50Block(tf.keras.layers.Layer):
    
    def __init__(self,
                stack: int,
                res_block: int,
                num_filters: int = 16,
                activation_type: str = 'relu', #relu or sin!
                l2_weight: float = 1e-4):
        
        super(resnet50Block,self).__init__()

        self.stack = stack
        self.res_block = res_block
        self.num_filters = num_filters
        self.activation_type = activation_type
        self.l2_weight = l2_weight
        

        
        strides = 1
        if self.stack > 0 and self.res_block == 0:
            strides = 2
#         print(f'resnet50Block: stack={stack}, res_block={res_block}, filters={num_filters}')
        
        self.l_1 = resnetLayer(num_filters=self.num_filters,
                               kernel_size=1,
                               strides=strides,
                               l2_weight=self.l2_weight,
                               activation_type=self.activation_type)

        self.l_2 = resnetLayer(num_filters=self.num_filters,
                               kernel_size=3,
                               l2_weight=self.l2_weight,
                               activation_type=self.activation_type)
        
        self.l_3 = resnetLayer(num_filters=4*self.num_filters,
                               kernel_size=1,
                               l2_weight=self.l2_weight,
                               use_activation=False,
                               use_norm=True)

        self.l_4 = resnetLayer(num_filters=4*self.num_filters,
                               kernel_size=1,
                               strides=strides,
                               l2_weight=self.l2_weight,
                               use_activation=False)

        self.l_add = tf.keras.layers.Add()
        self.l_activation = _activ(self.activation_type)
        
    def call(self,inputs: tf.Tensor) -> tf.Tensor:
        y = self.l_1(inputs)
        y = self.l_2(y)
        y = self.l_3(y)
        if self.res_block == 0:
            x = self.l_4(inputs)
        else:
            x = inputs
            
        x = self.l_add([x, y])
        x = self.l_activation(x)
        return x

#agreed with tf.keras.applications.ResNet50
class resnet50(tf.keras.Model):
    def __init__(self,
                 batch_size: int = 128,
                 l2_weight: float = 0.0,
                 activation_type: str = 'relu', #relu or sin
                 certainty_variant: str = 'partial', #partial, total or normalized
                 model_variant: str = '1vsall', #1vsall or vanilla
                 logit_variant: str = 'affine', #affine or dm
                 **params):
        super(resnet50,self).__init__()
        
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        
        if activation_type in ['sin','relu']:
            self.activation_type = activation_type
        else:
            raise ValueError(f'unknown activation_type={activation_type}')
        
        if certainty_variant in ['partial','total','normalized']:
            self.certainty_variant = certainty_variant
        else:
            raise ValueError(f'unknown certainty_variant={certainty_variant}')
        
        if model_variant in ['1vsall','vanilla']:
            self.model_variant = model_variant
        else:
            raise ValueError(f'unknown model_variant={model_variant}')
                        
        if logit_variant in ['affine','dm']:
            self.logit_variant = logit_variant
        else:
            raise ValueError(f'unknown logit_variant={logit_variant}')
        

#         self.num_res_blocks = int((self.depth - 2) / 6)
        self.num_res_blocks = [3,4,6,3]
        num_filters = 64
        
        self.layer_init_1 = tf.keras.layers.InputLayer(input_shape=(224, 224, 3),
                                                       batch_size=self.batch_size)

        self.layer_init_2 = resnetLayer(num_filters=num_filters,
                                        kernel_size=7,
                                        strides=2,
                                        l2_weight=self.l2_weight,
                                        activation_type=self.activation_type)
        
        self.layer_init_3 = tf.keras.layers.MaxPooling2D(pool_size=3,strides=2)
        
        #self.res_blocks = [[0 for stack in range(4)] for res_block in range(max(self.num_res_blocks))]
        self.res_blocks = [[0 for res_block in range(max(self.num_res_blocks))] for stack in range(4)]        

        for stack in range(4):
            for res_block in range(self.num_res_blocks[stack]):
                self.res_blocks[stack][res_block] = resnet50Block(stack = stack,
                                                                  res_block = res_block,
                                                                  num_filters = num_filters,
                                                                  activation_type = self.activation_type,
                                                                  l2_weight = self.l2_weight)
            num_filters *= 2  
            
        self.layer_final_1 = tf.keras.layers.AveragePooling2D(7)
        self.layer_final_2 = tf.keras.layers.Flatten()
        
        if self.logit_variant == 'dm':
            self.layer_final_3 = DMLayer(units=1000)
        elif self.logit_variant == 'affine':
            self.layer_final_3 = tf.keras.layers.Dense(1000, kernel_initializer='he_normal')
        else:
            raise ValueError(f'unknown logit_variant={self.logit_variant}')   
            
    def call(self, 
             inputs: tf.Tensor, 
             trainable: bool = False) -> dict:

        x = self.layer_init_1(inputs)
        x = self.layer_init_2(x)
        x = self.layer_init_3(x)
        
        for stack in range(4):
            for res_block in range(self.num_res_blocks[stack]):
                x = self.res_blocks[stack][res_block](x)

        x = self.layer_final_1(x)
        x = self.layer_final_2(x)
        
        logits = self.layer_final_3(x)
        
        if self.model_variant == '1vsall':
            probs = tf.math.sigmoid(logits)
            if self.logit_variant == 'dm':
                probs = 2*probs
        elif self.model_variant == 'vanilla':
            probs = tf.math.softmax(logits,axis=-1)
        else:
            raise ValueError(f'unknown model_variant={self.model_variant}')
        
        certs = _calc_certs(probs, certainty_variant = self.certainty_variant)
        logits_from_certs = _calc_logits_from_certs(certs = certs)
        
        return {'logits':logits,'probs':probs,'certs':certs,'logits_from_certs':logits_from_certs}

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        
        #garbage collection
        return {m.name: m.result() for m in self.metrics}    
    
class dummymodel(tf.keras.Model):
    def __init__(self,
                 batch_size:int = 128,
                 **params):
        super(dummymodel,self).__init__()
        self.batch_size = batch_size
        
        self.layer_1 = tf.keras.layers.InputLayer(input_shape=(224, 224, 3),
                                                       batch_size=self.batch_size)
        self.layer_2 = tf.keras.layers.Flatten()
        self.layer_3 = tf.keras.layers.Dense(1000, kernel_initializer='he_normal')

    def call(self, 
             inputs: tf.Tensor, 
             trainable: bool = False) -> dict:

        x = self.layer_1(inputs)
        x = self.layer_2(x)
        logits = self.layer_3(x)
        
        probs = tf.math.sigmoid(logits)
        
        return {'logits':logits,'probs':probs,'certs':probs,'logits_from_certs':logits}
        
        
# def create_model(batch_size: int,
#                  l2_weight: float = 0.0,
#                  activation_type: str = 'relu', #relu or sine
#                  certainty_variant: str = 'partial', # total, partial or normalized
#                  model_variant: str = '1vsall', #1vsall or vanilla
#                  logit_variant: str = 'affine', #affine or dm
#                  **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
    
#     return resnet20(batch_size=batch_size,
#                     l2_weight=l2_weight,
#                     activation_type=activation_type,
#                     certainty_variant=certainty_variant,
#                     model_variant=model_variant,
#                     logit_variant=logit_variant)


# based on um.numpy.plot_diagram, um.numpy.reliability_diagram
def _extract_conf_acc(probs,labels,bins=0,one_hot=False):

    probs = np.array(probs)
    labels = np.array(labels)
    if not one_hot:
        labels_matrix = um.numpy.visualization.one_hot_encode(labels, probs.shape[1])
    else:
        labels_matrix = labels

    # plot_diagram(probs.flatten(), labels_matrix.flatten(), y_axis))

    probs = probs.flatten()
    labels = labels_matrix.flatten()

    probs_labels = [(prob, labels[i]) for i, prob in enumerate(probs)]
    probs_labels = np.array(sorted(probs_labels, key=lambda x: x[0]))
    window_len = int(len(labels)/100.)
    calibration_errors = []
    confidences = []
    accuracies = []
    # More interesting than length of the window (which is specific to this
    # window) is average distance between datapoints. This normalizes by dividing
    # by the window length.
    distances = []
    for i in range(len(probs_labels)-window_len):
        distances.append((
            probs_labels[i+window_len, 0] - probs_labels[i, 0])/float(window_len))
        # It's pretty sketchy to look for the 100 datapoints around this one.
        # They could be anywhere in the probability simplex. This introduces bias.
        mean_confidences = um.numpy.visualization.mean(probs_labels[i:i + window_len, 0])
        confidences.append(mean_confidences)
        class_accuracies = um.numpy.visualization.mean(probs_labels[i:i + window_len, 1])
        accuracies.append(class_accuracies)
        calibration_error = class_accuracies-mean_confidences
        calibration_errors.append(calibration_error)
    if bins>0:
        delta = int((len(probs_labels)-window_len)/bins)
        return confidences[::delta],accuracies[::delta]
    else:
        return confidences, accuracies

    
    
    
    
# nonlinear calibration
class calLayer(tf.keras.layers.Layer):
    def __init__(self,
                 basis_type: str = 'uniform',
                 basis_params: list = [-20,20,20],
                 basis_list: list = [-2,-1,0,1,2],
                 train_basis=True):
        super(calLayer,self).__init__()
        self.basis_type = basis_type
        self.basis_params = basis_params
        self.basis_list = basis_list
        self.train_basis = train_basis
        
    def build(self, input_shape):
#         if input_shape[-1]!=1:
#             raise ValueError('input_shape != 1')

        if self.basis_type=='uniform':
            self.basis_exponents = np.linspace(*self.basis_params)
        else:
            self.basis_exponents = self.basis_list
        
        self.basis_exponents = tf.convert_to_tensor(self.basis_exponents,dtype=tf.float32)
        self.alphas = tf.exp(self.basis_exponents)            
        #self.alphas = tf.cast(self.alphas,dtype=tf.float32)
        self.alphas = tf.Variable(name='calLayer_alphas',
                                  initial_value=self.alphas,
                                  trainable=self.train_basis)
        self.W1 = self.add_weight(name='calLayer_weights',
                                  shape=(len(self.basis_exponents),),
                                  initializer="he_normal",
                                  trainable=True)
    
    def get_config(self):
        return {"basis_type": self.basis_type,
                "basis_params": self.basis_params,
                "basis_list": self.basis_list,
                "train_basis": self.train_basis}

    def call(self,inputs):
        inputs_shape = tf.shape(inputs)
        inputs_r = tf.reshape(inputs,shape=(-1,1))
        self.beta = tf.nn.softmax(self.W1)
        #print(self.alphas)
        eps = 1e-10
        x_alpha = tf.pow(inputs_r+eps,self.alphas)
        out = tf.reduce_sum(self.beta*x_alpha,axis=-1)
        
        return tf.reshape(out,shape=inputs_shape)

def _form_cal_dataset(uncal_model:tf.keras.Model,
                      output_name:str,
                      train_dataset,
                      dataset_bins:int,
                      steps:int,
                      append_random:bool = False,
                      random_frac:float = 0.1):

    cal_dataset = dict()
    #FLAGS = exp1.FLAGS
    #self.cal_dataset = train_dataset
    #dataset = exp1.datasets['cifar10']['val']
    labels = np.empty(0)
    probs = None

    for i,(x,y) in enumerate(train_dataset):
        if i>steps: break

        out = uncal_model(x)[output_name].numpy()
        labels = np.append(labels,y.numpy().astype('int32'))
        probs = out if type(probs)==type(None) else np.concatenate((probs,out))
        
    if append_random:
        print(labels.shape,probs.shape)
        
        random_frac = random_frac
        random_mean = 0.5
        random_std = 0.33
        batch_size = next(iter(train_dataset))[0].shape[0]
        val_examples = steps*batch_size
        random_size = int(val_examples*random_frac)

        #random_x = np.sqrt(random_std)*np.random.randn(random_size,32,32,3) + random_mean
        random_x = np.random.rand(random_size,32,32,3)
        random_probs = uncal_model(random_x)[output_name].numpy()
        #random_labels_onehot = np.zeros(shape=(random_size,random_probs.shape[1]))
        random_labels_onehot = np.ones(shape=(random_size,random_probs.shape[1]))
        #random_labels_onehot = np.random.binomial(1,0.5,size=(random_size,random_probs.shape[1]))
        #random_labels_onehot = np.ones(shape=(random_size,random_probs.shape[1]))
        #random_labels_onehot = np.random.randint(low=0,high=2,size=(random_size,random_probs.shape[1])).astype('float32')
        
        labels_onehot = um.numpy.visualization.one_hot_encode(labels.astype('int32'), probs.shape[1])
        
#         print(labels_onehot.shape)
#         print(random_labels_onehot.shape)
#         print(labels_onehot.dtype)
#         print(random_labels_onehot.dtype)
#         print(random_labels_onehot2.dtype)
#         print(random_labels_onehot[0])
#         print(random_labels_onehot2[0])
        labels_onehot = np.concatenate((labels_onehot,random_labels_onehot))
        probs = np.concatenate((probs,random_probs))
        print(labels_onehot.shape,probs.shape)
        
        confidences, accuracies = _extract_conf_acc(probs=probs,
                                                    labels=labels_onehot,
                                                    bins=dataset_bins,
                                                    one_hot=True)
        
    else:
        confidences, accuracies = _extract_conf_acc(probs=probs,
                                                    labels=labels.astype('int32'),
                                                    bins=dataset_bins,
                                                    one_hot=False)

    cal_dataset['x'] = tf.convert_to_tensor(confidences,dtype=tf.float32)
    cal_dataset['y'] = tf.convert_to_tensor(accuracies,dtype=tf.float32)     

    return cal_dataset
    
class nonlin_calibrator(tf.keras.Model):
    def __init__(self,
                 basis_type: str = 'uniform',
                 basis_params: list = [-20,20,10],
                 basis_list: list = [-2,-1,0,1,2],
                 train_basis: bool = True):
        
        super(nonlin_calibrator,self).__init__()
        self.layer = calLayer(basis_type = basis_type,
                              basis_params = basis_params,
                              basis_list = basis_list,
                              train_basis = train_basis)

    def call(self, 
             inputs: tf.Tensor, 
             training: bool = False) -> tf.Tensor:
        x = self.layer(inputs)
        return x

class cal_model(tf.keras.Model):
    def __init__(self,
                 uncal_model: tf.keras.Model,
                 calibrator: tf.keras.Model,
                 output_name: str):
        super(cal_model,self).__init__()
        
        #self.inp = tf.keras.layers.Input(shape=(32,32,3))
        self.uncal_model = uncal_model
        self.calibrator = calibrator
        self.output_name = output_name
        
    def call(self,
             inputs:tf.Tensor):
        x = self.uncal_model(inputs)[self.output_name]
        x = self.calibrator(x)
        
        return {self.output_name: x}
        
        
def calibrate_model_nonlin(model,
                    dataset,
                    FLAGS,
                    output='certs',
                    epochs=10000,
                    verbose=False,
                     bins=4000,
                     basis_type='uniform', # or list
                     basis_params={-20,20,60},
                     basis_list = [-2,-1,0,1,2]
                    ): 
    
    def feature_create(x,basis_exponents):
        x_feat = tf.tile(tf.reshape(x,shape=(-1,1)),[1,len(basis_exponents)])
        be_tf = tf.convert_to_tensor(basis_exponents,dtype=tf.float32)
        return tf.pow(x_feat,tf.exp(be_tf))
    
    def cal_out(W1,x,basis_exponents):
        
        size = len(basis_exponents)
        x_shape = tf.shape(x)   
#         print(x_shape)
        xr = tf.reshape(x,shape=(-1,))
#         print(xr.shape)
        W1_tile = tf.tile(tf.reshape(tf.nn.softmax(W1),[1,size]),[tf.shape(xr)[0],1])
        x_feat = feature_create(xr,basis_exponents)
    #         print(W1_tile.shape)
    #         print(x_feat.shape)
        out = tf.reduce_sum(W1_tile*x_feat,axis=-1)
        return tf.reshape(out,shape=x_shape)


    def cost(W1, x, y):

        yhats = cal_out(W1,x,basis_exponents)
#         print(yhats.shape)
#         print(y.shape)
        cost_value = tf.keras.losses.MSE(y_true=y,
                                         y_pred=yhats)

        return cost_value

    def grad(W1, x, y):

        with tf.GradientTape() as tape:
            cost_value = cost(W1, x, y)

        return cost_value, tape.gradient(cost_value, W1)


    if basis_type=='uniform':
        basis_exponents = np.linspace(*basis_params)
    else:
        basis_exponents = basis_list
    
    
    W1 = tf.Variable(tf.random.normal(shape=(len(basis_exponents),)))
    
    optimizer = ub.optimizers.get(optimizer_name='adam',
                                  learning_rate_schedule='constant',
                                  learning_rate=0.1,
                                  weight_decay=None)    
    
    #number of classes
    K = FLAGS['no_classes']
    
    labels = np.empty(0)
    probs = np.empty((0,K))
    
    for i,(x,y) in enumerate(dataset):
        if i>FLAGS['validation_steps']: break

        out = model(x)[output].numpy()
        labels = np.append(labels,y.numpy().astype('int32'))
        probs = np.concatenate((probs,out))

    confidences, accuracies = _extract_conf_acc(probs=probs,labels=labels.astype('int32'),bins=bins)
    
    X_train = tf.convert_to_tensor(confidences,dtype=tf.float32)
    y_train = tf.convert_to_tensor(accuracies,dtype=tf.float32)
    
    for i in range(epochs):
        train_cost, grads = grad(W1,X_train,y_train)
        optimizer.apply_gradients(zip([grads], [W1]))
#         if i % 50 == 0: 
#             print(train_cost.numpy())

    def model_return():
        inp = tf.keras.layers.Input(shape=(32,32,3))
        out_model = model(inp)
        out_calibr = cal_out(W1,out_model[output],basis_exponents=basis_exponents)
        out_model[output+'_cal'] = out_calibr
        
        return tf.keras.Model(inputs=inp,outputs=out_model)
    
#     def cal_model(x):
        
#         #out = tf.keras.models.Model(model.layers[0].input, model.output[output])(x)
#         out = model(x)[output]
#         out_shape = out.shape
#         out_calibr = cal_out(W1,out,basis_exponents=basis_exponents)
        
#         return {output+'_cal':out_calibr}
    
    return model_return(),W1