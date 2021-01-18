import os.path
import argparse

import pandas as pd
import numpy as np

from exp import experiment

parser = argparse.ArgumentParser(description='Multihead CIFAR-10 ResNet-20 comp-metrics')
parser.add_argument("--config", type=str, help="config json file",default='FLAGS.json')
parser.add_argument("--dataset", type=str, help="dataset to compute metrics; cifar10; cifar10-c; ood",default='cifar10-c')
args = parser.parse_args()

exp1 = experiment(exp_name='exp1',verbose=True)
exp1.load_flags(path=args.config)
exp1.FLAGS['dataset'] = args.dataset
exp1.FLAGS['exp']['phases'] = ['eval']
exp1.FLAGS['model']['pretrained'] = True
exp1.FLAGS['exp']['tune_hyperparams'] = False
#exp1.FLAGS['model']['model_file'] = '1vsall_dm_relu/model.ckpt-250'
exp1.load_data()

dsets = dict()
for name, dset_dict in exp1.datasets.items():
    dsets[name] = dset_dict['test']
    
certs = ['partial','total','normalized']
for cert in certs:
    exp1.FLAGS['certainty_variant']=cert
    exp1.prepare_exp()
    exp1.model_eval(datasets=dsets,save_result=True)
    
#     #split probs from certs
#     if args.dataset == 'cifar10-c':

#   split probs from certs
    model_name_probs = exp1.FLAGS['model']['output_dir']+'_probs_'+exp1.FLAGS['dataset']+'_metrics.csv'
    model_name_certs = exp1.FLAGS['model']['output_dir']+'_'+cert+'_'+exp1.FLAGS['dataset']+'_metrics.csv'
    df = pd.read_csv(os.path.join(exp1.FLAGS['model']['output_dir'],model_name_certs),index_col=0)
    probs_mask = df['metric'].str.contains('probs|loss')
    certs_mask = df['metric'].str.contains('certs|loss')
    dfprobs = df[probs_mask]
    dfcerts = df[certs_mask]
    dfprobs['metric'] = dfprobs[['metric']].applymap(lambda x: x.replace('probs_',''))
    dfcerts['metric'] = dfcerts[['metric']].applymap(lambda x: x.replace('certs_',''))
    dfprobs.to_csv(os.path.join(exp1.FLAGS['model']['output_dir'],model_name_probs))
    dfcerts.to_csv(os.path.join(exp1.FLAGS['model']['output_dir'],model_name_certs))
    

    