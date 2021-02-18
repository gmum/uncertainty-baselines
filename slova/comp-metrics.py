import os.path
import argparse

import pandas as pd
import numpy as np

from exp import experiment

parser = argparse.ArgumentParser(description='SLOVA CIFAR-10 ResNet-20 comp-metrics')
parser.add_argument("--config", type=str, help="config json file",default='FLAGS.json')
parser.add_argument("--phase", type=str,help="run evaluation on model: eval or cal_eval",default='eval')
parser.add_argument("--dataset", type=str, help="dataset to compute metrics; cifar10; cifar10-c",default='cifar10-c')
args = parser.parse_args()

exp1 = experiment(exp_name='exp1',verbose=True)
exp1.load_flags(path=args.config)
exp1.FLAGS['dataset'] = args.dataset
exp1.FLAGS['model']['pretrained'] = True
exp1.FLAGS['cal']['pretrained'] = True
exp1.FLAGS['exp']['tune_hyperparams'] = False
exp1.load_data()

dsets = dict()
for name, dset_dict in exp1.datasets.items():
    dsets[name] = dset_dict['test']
    
certs = [exp1.FLAGS['certainty_variant']]
for cert in certs:
    exp1.prepare_exp()
    postfix = '_metrics'
    if args.phase=='eval':
        postfix+='_uncal'
        df = exp1.model_eval(datasets=dsets,save_results=True)

    #   split probs from certs
        model_name_probs = exp1.FLAGS['exp']['output_dir']+'_probs_'+exp1.FLAGS['dataset']+postfix+'.csv'
        model_name_certs = exp1.FLAGS['exp']['output_dir']+'_'+cert+'_'+exp1.FLAGS['dataset']+postfix+'.csv'

        probs_mask = df['metric'].str.contains('probs|loss')
        certs_mask = df['metric'].str.contains('certs|loss')
        dfprobs = df[probs_mask]
        dfcerts = df[certs_mask]
        dfprobs.loc[:,'metric'] = dfprobs[['metric']].applymap(lambda x: x.replace('probs_',''))
        dfcerts.loc[:,'metric'] = dfcerts[['metric']].applymap(lambda x: x.replace('certs_',''))
        dfprobs.to_csv(os.path.join(exp1.FLAGS['exp']['output_dir'],model_name_probs))
        dfcerts.to_csv(os.path.join(exp1.FLAGS['exp']['output_dir'],model_name_certs))

    elif args.phase=='cal_eval':
        postfix+='_cal'
        df = exp1.cal_model_eval(datasets=dsets,save_results=True)
        model_name = exp1.FLAGS['exp']['output_dir']+'_'+exp1.FLAGS['cal']['model_output']+'_'+exp1.FLAGS['dataset']+postfix+'.csv'
        df.to_csv(os.path.join(exp1.FLAGS['exp']['output_dir'],model_name))

    

    