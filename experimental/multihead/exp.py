import json
import os

from func import AttrDict

import numpy as np
import pandas as pd

import tensorflow as tf
import kerastuner
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20 import one_vs_all_loss_fn
from resnet20 import create_model as resnet20

from resnet20 import nonlin_calibrator,_form_cal_dataset, cal_model

from metrics import nll
from metrics import BrierScore


from func import _load_datasets_basic, _load_datasets_corrupted, _load_datasets_OOD, _reconcile_flags_with_dataset


class experiment:
    
    def __init__(self,exp_name='exp',verbose=False,gpu_memory_fraction=0.31):
        
        self.exp_name = exp_name
        self.verbose = verbose
        self.FLAGS = dict()
        self.model = None
        self.cal_model = None
        self.callbacks = dict()
        self.metrics = None
        self.loss = None
        self.optimizer = None
        self.tuner = None
        
        self.datasets = dict()
        self.data_builders = dict()
        
        self.dataset_loaded = False
        
        # set limits on gpu_memory
        config = tf.compat.v1.ConfigProto()
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,
                                              allow_growth=False)
        sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                               log_device_placement=True)

        sess = tf.compat.v1.Session(config=sess_config)
        tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
        
    def save_flags(self,path):
        with open(path,'w') as fp:
            json.dump(self.FLAGS,fp,indent=4)
        
    def load_flags(self,path=None,load_default=True):
        
        if load_default:
            with open('FLAGS.json','r') as fp:
                FLAGS_default = json.load(fp)
        
        if path is None:
            self.FLAGS = AttrDict(FLAGS_default)
            return None
        
        assert os.path.exists(path), f'conf file {path} does not exist' 
        with open(path,'r') as fp:
            FLAGS = json.load(fp)

        if load_default:
            for name,val in FLAGS_default.items():
                if name in FLAGS.keys(): FLAGS_default[name] = FLAGS[name]
        else:
            FLAGS_default = FLAGS.copy()
                
        self.FLAGS = AttrDict(FLAGS_default)

    def load_data(self,overwrite=True):
        
        if self.verbose: print(f'Loading dataset={self.FLAGS.dataset}...')        
        if overwrite:
            self.datasets = dict()
            self.data_builders = dict()
            
        if self.FLAGS.dataset == 'cifar10':
            dataset_builder,train_dataset,val_dataset,test_dataset = _load_datasets_basic(self.FLAGS)
            self.datasets = {'cifar10': {'train': train_dataset,'val': val_dataset,'test': test_dataset}}
            self.data_builders = {'cifar10': dataset_builder}
            self.FLAGS = _reconcile_flags_with_dataset(self.data_builders['cifar10'],self.FLAGS)
            
        elif self.FLAGS.dataset == 'cifar10-c':
            flags = self.FLAGS.copy()
            flags.dataset = 'cifar10'
            dataset_builder,train_dataset,val_dataset,test_dataset = _load_datasets_basic(self.FLAGS)
            _, test_datasets_corrupt = _load_datasets_corrupted(flags)
            self.data_builders = {'cifar10-c': dataset_builder}

            for name in test_datasets_corrupt.keys():
                if name not in self.datasets.keys(): self.datasets[name] = dict()
                self.datasets[name]['test'] = test_datasets_corrupt[name]  
                
            self.FLAGS = _reconcile_flags_with_dataset(self.data_builders['cifar10-c'],self.FLAGS)
            
        elif self.FLAGS.dataset == 'ood':
            ood_datasets, ood_builders = _load_datasets_OOD(self.FLAGS)
            self.data_builders = ood_builders

            for name in ood_datasets.keys():
                if name not in self.datasets.keys(): self.datasets[name] = dict()
                self.datasets[name]['test'] = ood_datasets[name]         
            
            # TODO: reconcile FLAGS!
            
        else:
            raise ValueError(f'unknown dataset={self.FLAGS.dataset}')
        self.dataset_loaded = True    
        
    def set_output_dir(self):
        dirpath = self.FLAGS['exp']['output_dir']
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        path = os.path.join(dirpath,'FLAGS.json')
        if not os.path.exists(path): self.save_flags(path=path)

            
            
    def load_model(self,path=None):
        if path is None: path = self.FLAGS['model']['model_file']
        load = self.model.load_weights(path).expect_partial()
        if self.verbose: print(f'Loaded model...{path}')

    def save_model(self,path=None):
        if path is None:
            #epochs = self.FLAGS['train_params']['epochs']
            #path = os.path.join(self.FLAGS['exp']['output_dir'], f'model.ckpt-{epochs}')
            path = os.path.join(self.FLAGS['exp']['output_dir'], f'model.ckpt')
        
#         if self.FLAGS['exp']['tune_hyperparams']:
#             models = self.tuner.get_best_models(num_models=1)
#             models[0].save_weights(path)
#         else:
        self.model.save_weights(path)
        if self.verbose: print(f'Saving model to {path}')

    def set_model(self,model_name=None):
        if model_name is None:
            model_name = self.FLAGS['model']['name']
        if self.verbose: print(f'Setting model...{model_name}')
        if model_name == 'resnet20':
            self.model = resnet20(batch_size=self.FLAGS['train_params']['batch_size'],
                                 l2_weight=self.FLAGS['weight_decay'],
                                 activation_type=self.FLAGS['activation'],
                                 certainty_variant=self.FLAGS['certainty_variant'],
                                 model_variant=self.FLAGS['model_variant'],
                                 logit_variant=self.FLAGS['logit_variant'])
        else:
            raise ValueError(f'unknown model_name={model_name}')
        
    def set_seed(self):
        tf.random.set_seed(self.FLAGS['exp']['seed'])
        np.random.seed(self.FLAGS['exp']['seed'])
        
    def set_callbacks(self):
        if self.verbose: 
            use_tb = self.FLAGS['exp']['use_tensorboard']
            use_cp = self.FLAGS['exp']['save_checkpoints']
            use_es = self.FLAGS['exp']['use_early_stopping']
            print(f'Setting train callbacks...TensorBoard={use_tb}')
            print(f'Setting train callbacks...ModelCheckpoint={use_cp}')
            print(f'Setting train callbacks...EarlyStopping={use_es}')
        if use_tb:
            self.callbacks['tensorboard'] = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.FLAGS['exp']['output_dir'],'logs'))
    
        if use_cp:
            self.callbacks['checkpoint'] = tf.keras.callbacks.ModelCheckpoint(self.FLAGS['model']['model_file'], 
                                                                              monitor='val_probs_acc', 
                                                                              verbose=1,
                                                                              save_best_only=True,
                                                                              save_weights_only=True,
                                                                              mode='max')
        if use_es:
            self.callbacks['earlystop'] = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                           mode='min',
                                                                           verbose=1,
                                                                           patience=self.FLAGS.tb_patience)
            
    def _get_callbacks(self):
        return list(self.callbacks.values())
    
    def _basic_metrics(self):
        metrics_basic = {}
        metrics_basic['logits'] = []
        metrics_basic['probs'] = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                                  um.ExpectedCalibrationError(num_bins=self.FLAGS.ece_bins,name='ece'),
                                  nll(name='nll'),
                                  BrierScore(name='brier')]
        metrics_basic['certs'] = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                                  um.ExpectedCalibrationError(num_bins=self.FLAGS.ece_bins,name='ece'),
                                  nll(name='nll'),
                                  BrierScore(name='brier')]
        metrics_basic['logits_from_certs'] = []
        
        return metrics_basic
    
    def set_metrics(self):
        if self.verbose: print(f'Setting metrics...')
        del self.metrics  
        self.metrics = None
        
        if self.FLAGS.model_variant == '1vsall':
            self.metrics = self._basic_metrics()

        elif self.FLAGS.model_variant == 'vanilla':        
            self.metrics = self._basic_metrics()

        else:
            raise ValueError(f'unknown model_variant={self.FLAGS.model_variant}')
    
    def set_loss(self):
        if self.verbose: print(f'Setting losses...')
        del self.loss
        self.loss = None
        
        if self.FLAGS.model_variant=='1vsall':
            self.loss = {'logits':one_vs_all_loss_fn(from_logits=True),
                         'probs':None,
                         'certs':None,
                         'logits_from_certs':None}

        elif self.FLAGS.model_variant=='vanilla':        
            self.loss = {'logits':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         'probs':None,
                         'certs':None,
                         'logits_from_certs':None}

        else:
            raise ValueError(f'unknown model_variant={self.FLAGS.model_variant}')
    
    def set_optimizer(self):
        if self.verbose: print(f'Setting optimizer...')  
        def pick_scheduler(lr):
            lrs = self.FLAGS['optimizer']['lr_scheduler']
            if lrs == 'piecewise_linear':
                bound_1 = int(0.5*self.FLAGS['train_params']['epochs']*self.FLAGS['train_params']['steps_per_epoch'])
                bound_2 = int(0.75*self.FLAGS['train_params']['epochs']*self.FLAGS['train_params']['steps_per_epoch'])
                #bound_1 = 32000
                #bound_2 = 48000
                boundaries = [bound_1, bound_2]
                values = [lr, lr/10, lr/100]
                lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
            elif lrs == 'const':
                lr_scheduler = lr
            else:
                raise ValueError(f'unknown lr_scheduler={lrs}')
            return lr_scheduler

        def hp_params(hp):
            params = {}
            params['learning_rate'] = hp.Float('learning_rate',min_value=1e-3,max_value=1) #extracted from UDL2020-paper-040.pdf
            params['epsilon'] = hp.Float('epsilon',min_value=1e-8,max_value=1e-5)
            params['beta_1'] = hp.Float('beta_1',min_value=0.85,max_value=0.99)
            params['momentum'] = self.FLAGS['optimizer']['momentum']
            return params
        
        nopt = self.FLAGS['optimizer']['name']
        if self.FLAGS['exp']['tune_hyperparams']:
            if nopt == 'sgd':
                def optimizer_sgd_func(hp):
                    params = hp_params(hp)
                    return tf.keras.optimizers.SGD(learning_rate=pick_scheduler(params['learning_rate']),
                                                   momentum=params['momentum'])
                self.optimizer = optimizer_sgd_func

            elif nopt == 'adam':
                def optimizer_adam_func(hp):
                    params = hp_params(hp)           
                    return tf.keras.optimizers.Adam(learning_rate=pick_scheduler(params['learning_rate']),
                                                 epsilon=params['epsilon'],
                                                 beta_1=params['beta_1']) 
                self.optimizer = optimizer_adam_func
            else:
                raise ValueError(f'unknown optimizer={self.FLAGS.optimizer}')
        else:
            lr = self.FLAGS['optimizer']['learning_rate']
            if nopt == 'sgd':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate=pick_scheduler(lr),
                                                    momentum=self.FLAGS['optimizer']['momentum']) 
            elif nopt == 'adam':
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=pick_scheduler(lr),
                                                     epsilon=self.FLAGS['optimizer']['epsilon'],
                                                     beta_1=self.FLAGS['optimizer']['beta_1'])    
            else:
                raise ValueError(f'unknown optimizer={nopt}')                

    def model_train(self,train_dataset=None,val_dataset=None):
        
        if train_dataset is None:
            train_dataset = self.datasets[self.FLAGS['train_dataset']]['train']
        if val_dataset is None:
            val_dataset = self.datasets[self.FLAGS['eval_dataset']]['val']
            
        if self.verbose: print(f'Starting plain training...') 
        history = self.model.fit(train_dataset,
                                 batch_size=self.FLAGS['train_params']['batch_size'],
                                 epochs=self.FLAGS['train_params']['epochs'],
                                 steps_per_epoch=self.FLAGS['train_params']['steps_per_epoch'],
                                 validation_data=val_dataset,
                                 validation_steps=self.FLAGS['train_params']['validation_steps'],
                                 validation_freq=self.FLAGS['train_params']['eval_frequency'],
                                 callbacks=self._get_callbacks(),
                                 shuffle=False)                    
                
    def model_compile(self):
        if self.verbose: print(f'Compiling plain trainer...')          
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)                
                
    def prepare_model(self):
        self.set_model()  
        self.set_metrics()
        self.set_loss()
        self.set_optimizer()
        self.set_callbacks()
        self.model_compile()
        if self.FLAGS['model']['pretrained']:
            self.load_model()    
      

    
    def tuner_compile(self):
        if self.verbose: print(f'Compiling kerastuner trainer...')          
        def tunable_model(hp):

            optimizer = self.optimizer(hp) if self.FLAGS['exp']['tune_hyperparams'] else self.optimizer
            self.model.compile(optimizer=optimizer,
                               loss=self.loss,
                               metrics=self._basic_metrics())
            return self.model  
        
        if self.FLAGS['tune']['by_metric'] == 'acc':
            objective = kerastuner.Objective("val_probs_acc", direction="max")
        elif self.FLAGS['tune']['by_metric'] == 'ece':
            objective = kerastuner.Objective("val_probs_ece", direction="min")
        elif 'val' in self.FLAGS['tune']['by_metric']:
            obj_name = self.FLAGS['tune']['by_metric']
            if 'ece' in obj_name:
                direction = "min"
            objective = kerastuner.Objective(obj_name, direction=direction)
                        
        else:
            bymetric = self.FLAGS['tune']['by_metric']
            raise ValueError(f'unknown by_metric={bymetric}')
        
        self.tuner = kerastuner.tuners.RandomSearch(tunable_model,
                                                    objective=objective,
                                                    max_trials=self.FLAGS['tune']['num_trials'],
                                                    executions_per_trial=1,
                                                    directory=self.FLAGS['tune']['dir'],
                                                    project_name=self.FLAGS['tune']['subdir'])

    def prepare_tuner(self):
        if self.FLAGS['exp']['tune_hyperparams']:
            self.tuner_compile()
                
    def model_tune(self):
        if self.verbose: print(f'Starting kerastuner training...')  
        self.tuner.search(self.datasets[self.FLAGS['train_dataset']]['train'],
                         batch_size=self.FLAGS['tune']['batch_size'],
                         epochs=self.FLAGS['tune']['epochs'],
                         validation_data=self.datasets[self.FLAGS['eval_dataset']]['val'],
                         callbacks=[]) 
        

    def evaluate(self,
                 model=None,
                 datasets=None,
                 save_results=False,
                 postfix=''):
        if model==None: model = self.model
        if datasets is None:
            ds_keys = self.datasets.keys()
            eval_ds = list(ds_keys)[0]
            if self.FLAGS['eval_dataset'] in ds_keys:
                eval_ds = self.FLAGS['eval_dataset']
            datasets = {eval_ds : self.datasets[eval_ds]['test']}
        
        if self.verbose: print(f'Starting evaluation...')
        if save_results:
            model_name = self.FLAGS['exp']['output_dir']+'_'+self.FLAGS['certainty_variant']
            df = pd.DataFrame(columns=['dataset','metric',model_name])
            
        for ds_name,dataset in datasets.items():
            if self.verbose: print('dataset =',ds_name)
            #out_metrics = self.model.evaluate(dataset, return_dict=True)
            out_metrics = model.evaluate(dataset,return_dict=True)
            
            if save_results:
                record = {}
                record['dataset'] = ds_name
#                 metrics_vals = out_metrics.values()
#                 metrics_names = out_metrics.keys()
                for metric,metric_val in out_metrics.items():

                    record['metric'] = metric
                    record[model_name] = metric_val

                    # save record
                    mask_dataset = df['dataset'] == record['dataset']
#                     mask_shift = df['shift_type'] == record['shift_type']
                    mask_metric = df['metric'] == record['metric']
                    row_ix = df[(mask_dataset) & (mask_metric)].index
        #             print('row_ix=',row_ix)
                    if len(row_ix)==0:
        #                 print('new df entry')
                        df = df.append(record,ignore_index=True)
                    elif len(row_ix)==1:
                        df_rec = df.at[row_ix[0],model_name]
                        if np.isnan(df_rec):
        #                     print('new record')
                            df.at[row_ix[0],model_name] = metric_val
                        else:
        #                     print('record exists, appending')
                            df.at[row_ix[0],model_name] = metric_val            
                    else:
                        print('multiple records, somethings wrong')
            else:
                print(out_metrics)
                
                
        if save_results:
            #df.to_csv(os.path.join(self.FLAGS['exp']['output_dir'],model_name+'_'+self.FLAGS['dataset']+postfix+'.csv'))
            return df
        else:
            return 0

        
#     def read_data(self,datasets,builders):
#         self.datasets = datasets
#         self.data_builders = builders
#         self.FLAGS = _reconcile_flags_with_dataset(self.data_builders[self.FLAGS['train_dataset']],self.FLAGS)
#         self.dataset_loaded = True


    def model_eval(self,datasets=None,save_results=False,postfix=''):
        return self.evaluate(model=self.model,
                      datasets=datasets,
                      save_results=save_results,
                      postfix=postfix)
    
    def extract_best_hp(self):
        best_hp_list = self.tuner.get_best_hyperparameters()
        if len(best_hp_list)>0:
            best_hp = best_hp_list[0]
            bhv = best_hp.values
            for key,val in bhv.items():
                self.FLAGS['optimizer'][key] = val

                
    def load_calibrator(self,path=None):
        if path is None: path = self.FLAGS['cal']['model_file']
        load = self.calibrator.load_weights(path).expect_partial()
        if self.verbose: print(f'Loaded calibrator...{path}')

    def save_calibrator(self,path=None):
        if path is None:
            #path = os.path.join(self.FLAGS['exp']['output_dir'], f'calibrator.ckpt')
            path = self.FLAGS['cal']['model_file']
        self.calibrator.save_weights(path)
        if self.verbose: print(f'Saving calibrator to {path}')

    def set_calibrator(self,cal_variant=None):
        if cal_variant is None:
            cal_variant = self.FLAGS['cal']['variant']
        if self.verbose: print(f'Setting calibrator cal_variant={cal_variant}')
        if cal_variant == 'nonlin':
            self.calibrator =  nonlin_calibrator(basis_type = self.FLAGS['cal']['basis_type'],
                                                 basis_params = self.FLAGS['cal']['basis_params'],
                                                 train_basis = self.FLAGS['cal']['train_basis'])
        else:
            raise ValueError(f'unknown cal_variant={cal_variant}') 
        
    def calibrator_compile(self):
        
        loss = tf.keras.losses.MSE
        optimizer = ub.optimizers.get(optimizer_name='adam',
                                      learning_rate_schedule='constant',
                                      learning_rate=0.1,
                                      weight_decay=None)
        metrics = []
        
        self.calibrator.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics)           

    def prepare_calibrator(self):
        if self.verbose: print(f'Preparing calibrator...')
        self.set_calibrator()  
        self.calibrator_compile()
        if self.FLAGS['cal']['pretrained']:
            self.load_calibrator()    


    def calibrator_train(self,cal_dataset=None):
        if cal_dataset is None:
            
            self.cal_dataset = _form_cal_dataset(uncal_model=self.model,
                                     output_name=self.FLAGS['cal']['model_output'],
                                     train_dataset = self.datasets[self.FLAGS['cal']['train_dataset']]['val'],
                                     dataset_bins=self.FLAGS['cal']['dataset_bins'],
                                     steps = self.FLAGS['train_params']['validation_steps'])
            cal_dataset = self.cal_dataset
            
        if self.verbose: print(f'Starting plain calibration...')
        self.calibrator.fit(x=cal_dataset['x'],y=cal_dataset['y'],epochs=self.FLAGS['cal']['epochs'])            
    

    def cal_model_eval(self,datasets=None,save_results=False,postfix=''):
        return self.evaluate(model=self.cal_model,
                      datasets=datasets,
                      save_results=save_results,
                      postfix=postfix)
         
    def prepare_cal_model(self):
        if self.verbose: print(f'Preparing cal_model...')
        self.cal_model = cal_model(uncal_model=self.model,
                                   calibrator=self.calibrator,
                                   output_name=self.FLAGS['cal']['model_output'])  
        self.cal_model.compile(metrics=self.metrics[self.FLAGS['cal']['model_output']])

    def prepare_exp(self,reload_dataset=False):
        
        self.set_seed()
        self.set_output_dir()
        self.prepare_model()

        self.prepare_tuner()
        if self.FLAGS['exp']['load_data']: 
            if not self.dataset_loaded or reload_dataset:
                self.load_data()
                
        self.prepare_calibrator()
        self.prepare_cal_model()
        
    def run_exp(self,phases=None):
        if phases is None: phases = self.FLAGS['exp']['phases']
        if 'tune' in phases:
            if self.FLAGS['exp']['tune_hyperparams']:
                self.model_tune()
                self.extract_best_hp()
                self.prepare_model()
                
        if 'train' in phases:
            self.model_train()
            self.curr_model = self.model    
            if self.FLAGS['exp']['save_model']:
                self.save_model()
                
        if 'cal' in phases:
            self.calibrator_train()
            if self.FLAGS['exp']['save_model']:
                self.save_calibrator()
                
        if 'eval' in phases:
            _ = self.model_eval()
            
        if 'cal_eval' in phases:
            _ = self.cal_model_eval()            
           
 