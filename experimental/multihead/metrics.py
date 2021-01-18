import os.path
import json

from absl import app
from absl import flags
from absl import logging

import numpy as np

import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

import dataset_utils #from baselines/cifar

def rel_diag(model,
             dataset,
             FLAGS,
             output='certs',
             savefig=False,
             path='',
            ):
    
    #number of classes
    K = FLAGS.no_classes
    
    labels = np.empty(0)
    probs = np.empty((0,K))
    
    for i,(x,y) in enumerate(dataset):
        if i>FLAGS.steps_per_epoch: break
            
        out = model(x)[output]

        labels = np.append(labels,y.numpy().astype('int32'))
        probs = np.concatenate((probs,out.numpy()))
    
    diagram = um.numpy.reliability_diagram(probs=probs,labels=labels.astype('int32'),img=False)
    if savefig:
        diagram.savefig(path)
 
      
def quartiles(data,**kwargs):
    q25 = list(np.percentile(data,25,**kwargs))
    q50 = list(np.percentile(data,50,**kwargs))
    q75 = list(np.percentile(data,75,**kwargs))
    return q25,q50,q75


class BrierScore(tf.keras.metrics.Mean):
#     positive brier score as defined in 
#     [1]: G.W. Brier.
#     Verification of forecasts expressed in terms of probability.
#     Monthley Weather Review, 1950.
#     for example in class k reads:
#     BS = 1 + sum_i p[i]*p[i] - 2 p[k]
#     um.brier_score defines BS without 1
    def __init__(self,*args,**kwargs):
        super(BrierScore,self).__init__(*args,**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if type(y_true)==np.ndarray:
            y_true = tf.constant(y_true)
        if type(y_pred)==np.ndarray:
            y_pred = tf.constant(y_pred)
            
        sh = tf.shape(y_true)
        if len(sh)>1:
          y_true = tf.reshape(y_true,sh[:1])
        y_true = tf.cast(y_true,dtype=tf.int32)
        
        brier_score = 1 + um.brier_score(labels=y_true, probabilities=y_pred)
        super(BrierScore, self).update_state(brier_score)  


class MMC(tf.keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super(MMC, self).__init__(*args, **kwargs)
        self.mmc = self.add_weight(name='mmc', initializer=tf.zeros_initializer)
        self.n = self.add_weight(name='n', initializer=tf.zeros_initializer,dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mmc.assign_add(tf.reduce_sum(tf.reduce_max(y_pred,axis=-1)))
        self.n.assign_add(tf.shape(y_pred)[0])
        #self.mmc.assign_add(tf.reduce_sum(tf.reduce_max(y_pred,axis=-1))/tf.cast(y_pred.shape[0],dtype=tf.float32))

    def result(self):
        return self.mmc/tf.cast(self.n,dtype=tf.float32)
        
        #return self.mmc
        
    def reset_states(self):
        self.mmc.assign(0)
        self.n.assign(0)

        
class nll(tf.keras.metrics.Metric):
    def __init__(self, name=None,dtype=None):
        super(nll, self).__init__(name,dtype)
        self.nll = self.add_weight(name='nll',initializer=tf.zeros_initializer)
        self.n = self.add_weight(name='n', initializer=tf.zeros_initializer,dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, **kwargs):
        eps=tf.keras.backend.epsilon()
        
        y_true = tf.cast(y_true,dtype=tf.int32)
        y_true = tf.squeeze(y_true)
        y_true0 = tf.one_hot(y_true,depth=y_pred.shape[-1],dtype=tf.float32)
        y_pred0 = tf.cast(y_pred,dtype=tf.float32)
        y_pred0 = tf.clip_by_value(y_pred0,eps,1-eps)
        
        logs = tf.math.log(y_pred0)
        sum1 = -tf.reduce_sum(tf.math.multiply(y_true0, logs),axis=-1)
        nll_sum = tf.reduce_sum(sum1)
        
        self.nll.assign_add(nll_sum)
        self.n.assign_add(tf.shape(y_pred0)[0])

    def result(self):
        #tf.print('calling result')
        return self.nll/tf.cast(self.n,dtype=tf.float32)

    def reset_states(self):
        #tf.print('calling reset_states')
        self.nll.assign(0)
        self.n.assign(0)

        
class acc_th(tf.keras.metrics.Metric):
    def __init__(self, th=0.0,name=None,dtype=None):
        super(acc_th, self).__init__(name,dtype)
        self.n_corr = self.add_weight(name='n_corr',initializer=tf.zeros_initializer,dtype=tf.int32)
        self.n_total = self.add_weight(name='n_total', initializer=tf.zeros_initializer,dtype=tf.int32)
        self.th = self.add_weight(name='th', initializer=tf.zeros_initializer,dtype=tf.float32)
        self.th.assign(th)
        
    def update_state(self, y_true, y_pred, **kwargs):
    
        y_true = tf.cast(y_true,dtype=tf.int32)
        y_true = tf.squeeze(y_true)
        y_pred0 = tf.cast(y_pred,dtype=tf.float32)

        mask = tf.reduce_max(y_pred0,axis=-1)>=self.th
        y_pred_mask = y_pred0[mask]
        y_true_mask = y_true[mask]
        y_pred_max = tf.argmax(y_pred_mask,axis=-1)
        y_pred_max = tf.cast(y_pred_max,dtype=tf.int32)
        
        equals = tf.equal(y_true_mask,y_pred_max)
                
        self.n_corr.assign_add(tf.reduce_sum(tf.cast(equals,dtype=tf.int32)))
        self.n_total.assign_add(tf.shape(y_pred_mask)[0])

    def result(self):
        #tf.print('calling result')
        return tf.cast(self.n_corr,dtype=tf.float32)/tf.cast(self.n_total,dtype=tf.float32)

    def reset_states(self):
        #tf.print('calling reset_states')
        self.n_corr.assign(0)
        self.n_total.assign(0)

class acc_ntotal_th(tf.keras.metrics.Metric):
    def __init__(self, th=0.0,name=None,dtype=None):
        super(acc_ntotal_th, self).__init__(name,dtype)
        self.n_total = self.add_weight(name='n_total', initializer=tf.zeros_initializer,dtype=tf.int32)
        self.th = self.add_weight(name='th', initializer=tf.zeros_initializer,dtype=tf.float32)
        self.th.assign(th)
        
    def update_state(self, y_true, y_pred, **kwargs):
    
#         y_true = tf.cast(y_true,dtype=tf.int32)
#         y_true = tf.squeeze(y_true)
        y_pred0 = tf.cast(y_pred,dtype=tf.float32)
        mask = tf.reduce_max(y_pred0,axis=-1)>=self.th
        y_pred_mask = y_pred0[mask]
#         y_true_mask = y_true[mask]
#         y_pred_max = tf.argmax(y_pred_mask,axis=-1)
#         y_pred_max = tf.cast(y_pred_max,dtype=tf.int32)
        
#         equals = tf.equal(y_true_mask,y_pred_max)
                
#         self.n_corr.assign_add(tf.reduce_sum(tf.cast(equals,dtype=tf.int32)))
        self.n_total.assign_add(tf.shape(y_pred_mask)[0])

    def result(self):
        #tf.print('calling result')
        return tf.cast(self.n_total,dtype=tf.float32)

    def reset_states(self):
        #tf.print('calling reset_states')
        self.n_total.assign(0)

        

def _calc_entropy(probs: tf.Tensor):
    
    eps=tf.keras.backend.epsilon()
    probs = tf.cast(probs,dtype=tf.float32)
    probs = tf.clip_by_value(probs,eps,1-eps)
    logs = tf.math.log(probs)
    result = -tf.reduce_sum(tf.math.multiply(probs, logs),axis=-1)
    
    return result        
        
class mean_entropy(tf.keras.metrics.Metric):
    def __init__(self, name=None,dtype=None):
        super(mean_entropy, self).__init__(name,dtype)
        self.entropy = self.add_weight(name='entropy',initializer=tf.zeros_initializer,dtype=tf.float32)
        self.n = self.add_weight(name='n', initializer=tf.zeros_initializer,dtype=tf.int32)
        
    def update_state(self, y_true, y_pred, **kwargs):
        sum1 = _calc_entropy(y_pred)
        entropy_sum = tf.reduce_sum(sum1)
        
        self.entropy.assign_add(entropy_sum)
        self.n.assign_add(tf.shape(y_pred)[0])

    def result(self):
        #tf.print('calling result')
        return self.entropy/tf.cast(self.n,dtype=tf.float32)

    def reset_states(self):
        #tf.print('calling reset_states')
        self.entropy.assign(0)
        self.n.assign(0)
        



