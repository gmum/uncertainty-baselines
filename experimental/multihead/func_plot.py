import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import h5py
import itertools

import tensorflow as tf
from metrics import nll, BrierScore
import uncertainty_metrics as um

from matplotlib.ticker import FormatStrFormatter

def load_df_dict(path='results'):
    df_dict = dict()
    for file in os.listdir(path):
        name,ext = os.path.splitext(file)
        if ext=='.csv':
            #print(os.path.join(path,file))
            df_dict[name] = pd.read_csv(os.path.join(path,file),index_col=0)
            
    return df_dict


def df_dict_to_pivot(df_dict,columns=None):
    '''
    reform df_dict to conform with plot functions
    '''
    
    df_dict_pivot = dict()
    for name,df in df_dict.items():
        if columns==None:
            df_dict_pivot[name] = df.pivot(index='dataset',columns='metric',values=df.columns[-1])
        else:
            df_dict_pivot[name] = df.pivot(index='dataset',columns='metric',values=df.columns[-1]).loc[:,columns]
    return df_dict_pivot


def pivot_to_df_dict(df_dict_pivot):
    df_dict = dict()
    for name,df in df_dict_pivot.items():    
        df_dict[name] = df.reset_index().melt(id_vars=['dataset']).sort_values(by='dataset').reset_index(drop=True)
    return df_dict


def rename_df_dict(df_dict,rename_dict):    
    return {(rename_dict[k] if k in rename_dict.keys() else k):v for k,v in df_dict.items()}


def load_pivot_df_dict_googlefig(hdf5file='metrics_googlefig/cifar_model_predictions.hdf5',clean_key='test',instance=0):
    '''
    load pivoted df_dict from google paper
    '''
    file = h5py.File(hdf5file,'r')
    
    corruptions = ['corrupt-static-brightness-1', 'corrupt-static-brightness-2', 'corrupt-static-brightness-3', 'corrupt-static-brightness-4', 'corrupt-static-brightness-5', 'corrupt-static-contrast-1', 'corrupt-static-contrast-2', 'corrupt-static-contrast-3', 'corrupt-static-contrast-4', 'corrupt-static-contrast-5', 'corrupt-static-defocus_blur-1', 'corrupt-static-defocus_blur-2', 'corrupt-static-defocus_blur-3', 'corrupt-static-defocus_blur-4', 'corrupt-static-defocus_blur-5', 'corrupt-static-elastic_transform-1', 'corrupt-static-elastic_transform-2', 'corrupt-static-elastic_transform-3', 'corrupt-static-elastic_transform-4', 'corrupt-static-elastic_transform-5', 'corrupt-static-fog-1', 'corrupt-static-fog-2', 'corrupt-static-fog-3', 'corrupt-static-fog-4', 'corrupt-static-fog-5', 'corrupt-static-frost-1', 'corrupt-static-frost-2', 'corrupt-static-frost-3', 'corrupt-static-frost-4', 'corrupt-static-frost-5', 'corrupt-static-gaussian_blur-1', 'corrupt-static-gaussian_blur-2', 'corrupt-static-gaussian_blur-3', 'corrupt-static-gaussian_blur-4', 'corrupt-static-gaussian_blur-5', 'corrupt-static-gaussian_noise-1', 'corrupt-static-gaussian_noise-2', 'corrupt-static-gaussian_noise-3', 'corrupt-static-gaussian_noise-4', 'corrupt-static-gaussian_noise-5', 'corrupt-static-glass_blur-1', 'corrupt-static-glass_blur-2', 'corrupt-static-glass_blur-3', 'corrupt-static-glass_blur-4', 'corrupt-static-glass_blur-5', 'corrupt-static-impulse_noise-1', 'corrupt-static-impulse_noise-2', 'corrupt-static-impulse_noise-3', 'corrupt-static-impulse_noise-4', 'corrupt-static-impulse_noise-5', 'corrupt-static-pixelate-1', 'corrupt-static-pixelate-2', 'corrupt-static-pixelate-3', 'corrupt-static-pixelate-4', 'corrupt-static-pixelate-5', 'corrupt-static-saturate-1', 'corrupt-static-saturate-2', 'corrupt-static-saturate-3', 'corrupt-static-saturate-4', 'corrupt-static-saturate-5', 'corrupt-static-shot_noise-1', 'corrupt-static-shot_noise-2', 'corrupt-static-shot_noise-3', 'corrupt-static-shot_noise-4', 'corrupt-static-shot_noise-5', 'corrupt-static-spatter-1', 'corrupt-static-spatter-2', 'corrupt-static-spatter-3', 'corrupt-static-spatter-4', 'corrupt-static-spatter-5', 'corrupt-static-speckle_noise-1', 'corrupt-static-speckle_noise-2', 'corrupt-static-speckle_noise-3', 'corrupt-static-speckle_noise-4', 'corrupt-static-speckle_noise-5', 'corrupt-static-zoom_blur-1', 'corrupt-static-zoom_blur-2', 'corrupt-static-zoom_blur-3', 'corrupt-static-zoom_blur-4', 'corrupt-static-zoom_blur-5']
    clean = [clean_key]
    corruptions += clean
    gfig_dict = dict()

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
     um.ExpectedCalibrationError(num_bins=15,name='ece'),
     nll(name='nll'),
     BrierScore(name='brier')]

    for model_name, model_data in file.items():
        #print(model_name)
        df = pd.DataFrame(columns=['acc','brier','ece','nll'])
        df.columns.name = 'metrics'
        df.index.name = 'dataset'

        for key in model_data.keys():
            if key in corruptions:
                #model_data[key]

                y = model_data[key]['labels']
                p = model_data[key]['probs']
                if 'roll' in key:
                    key = 'roll'
                for metric in metrics:
                    metric.update_state(y[instance,:].astype(np.int32),p[instance,:].astype(np.float32))
                d0 = {'acc':metrics[0].result().numpy(),
                      'ece':metrics[1].result().numpy(),
                      'nll':metrics[2].result().numpy(),
                      'brier':metrics[3].result().numpy()}
                record = pd.Series(d0,index=df.columns,name=key)
                df = df.append(record)
                for metric in metrics:
                    metric.reset_states()

        gfig_dict[model_name] = df.copy()

    return gfig_dict

def cifar10c_shift_plot(df_dict_pivot,metric,model_list,ax):
    '''
    plot shift plot for cifar10c
    '''
#     dfsum = None
#     for key,df in df_dict.items():
#         df_new = df[[metric]]
#         df_new = df_new.rename({metric:key},axis=1)
#         if type(dfsum) == type(None):
#             dfsum = df_new

#         else:
#             dfsum = pd.merge(left=dfsum,right=df_new,on=['dataset'])

#     dfsum['shift_amount'] = 0
#     shifts = [1,2,3,4,5]
#     for shift in shifts:
#         dfsum.loc[(dfsum.index.str.contains(str(shift))),'shift_amount'] = shift
        
#     #model_list = list(dfsum.columns[:-1])
#     dfr_sum_melt = pd.melt(dfsum,id_vars=['shift_amount'],value_vars=model_list,var_name='models')
    
    dfr_sum_melt = pd.DataFrame(columns=['models','shift_amount','value'])
    shifts = [1,2,3,4,5]
    for model in model_list:
        df_dict_pivot[model]['shift_amount'] = 0
        for shift in shifts:
            df_dict_pivot[model].loc[(df_dict_pivot[model].index.str.contains(str(shift))),'shift_amount'] = shift
        df_dict_pivot[model]['models'] = model    
        dfa = df_dict_pivot[model].reset_index().loc[:,['models','shift_amount',metric]].rename({metric:'value'},axis=1)
        dfr_sum_melt = pd.concat([dfr_sum_melt,dfa],axis=0)

    width = float(len(model_list)/(len(model_list)+1))
    sns.boxplot(x='shift_amount',
                y='value',
                data=dfr_sum_melt,
                hue='models',linewidth=1,width=width,showfliers=False,dodge=True,whis=10000.0,ax=ax)

    metric_dict = {'ece':'ECE','acc':'Accuracy','nll':'Negative Log Likelihood','brier':'Brier Score'}
    ax.set_ylabel(metric_dict[metric],fontsize=20)
    ax.set_xlabel('Shift intensity',fontsize=18)
    #ax.set_title('cifar10-c '+metric,fontsize=24)
    plt.setp(ax.get_legend().get_title(), fontsize='18')  
    plt.setp(ax.get_legend().get_texts(), fontsize='14')

def colorize(ax,colors):
    no_methods = len(colors)
    box_size = 5 #each box consists of 5 Line2D's
    no_shifts = 6
    #assert len(colors) == no_methods
    
    #colorize patches
    artists = np.array(ax.artists).reshape(no_shifts,no_methods)
    for artist in artists[:]:
        for color,a in zip(colors,artist):
            a._set_facecolor(color)
    
    linegroups = np.array(ax.get_lines()).reshape(no_shifts,no_methods,box_size)
    
    #colorize test lines
    shift_int = 0
    for color,lines in zip(colors,linegroups[shift_int,:,:]):
        for l in lines:
            l.set_color(color)
            l.set_linewidth(3)
    
    #bold our model
    our_method = 8
    for line in linegroups[:,our_method,:]:
        for color,l in zip(colors,line):
            l.set_linewidth(2)
    
    for a in artists[:,our_method]:
        a.set_linewidth(2)
    
    ax.get_legend().get_patches()[our_method].set_linewidth(2)
    
    #update legend
    for color,patch in zip(colors,ax.get_legend().get_patches()):
        patch.set_facecolor(color)
        
    
    
def cifar10c_shift_plots(df_dict_pivot,metrics,model_list,figsize=(15,10)):
  '''
  plot several cifar10c_shift_plot
  ''' 
  fig,axs = plt.subplots(nrows=len(metrics),ncols=1,figsize=figsize)
  legend_dict = {'vanilla':'Softmax',
                 'temp_scaling':'Temp Scaling',
                 'ensemble':'Ensemble',
                 'dropout_nofirst':'Dropout',
                 'll_dropout':'LL Dropout',
                 'svi':'SVI',
                 'll_svi':'LL SVI'}
  i=0  
  for ax,metric in zip(axs,metrics):
    cifar10c_shift_plot(df_dict_pivot,metric,model_list,ax)
    
    # modify legend
    for t in ax.get_legend().texts:
      newtext = legend_dict[t._text] if t._text in legend_dict.keys() else t._text
      plt.setp(t,text=newtext,fontsize=14)
    plt.setp(ax.get_legend().get_title(),text='Method')
    # modify x-axis labels
    ax.set_xticklabels(['Test',1,2,3,4,5])
    
    #extract colors
    colors=[]
    for patch in axs[0].get_legend().get_patches():
        colors.append(patch._facecolor)
    colors[-1] = 'gold'
    colorize(ax,colors)
  
    ax.set_xlabel('')  
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
  axs[-1].set_xlabel('Shift intensity')
  plt.tight_layout()    

  return fig, axs


# cifar10-c plots

def genplot_c(dfr,metric,ax,ascending=False):

  df_metric = dfr[(dfr['dataset']!='clean') & (dfr['metric']==metric)]
  pos2 = [3,2,1,0]
 
  metrics_dict = {'ece':'ECE','acc':'Accuracy','nll':'Negative Log Likelihood','brier':'Brier Score'}
  #df_metric.boxplot(vert=False,positions=pos2,sym='',ax=ax,grid=(True,False),figsize=(5,5))
  sns.boxplot(data=df_metric,
            linewidth=1,
            width=.8,
            showfliers=False,
            dodge=True,
            orient='h',
            whis=10000.0,
            ax=ax)
  ax.set_title(metrics_dict[metric],fontsize=16)

def cifar10_corrupt_plots(dfr,bbox,legend_fontsize,figsize=(15,20)):

  
  fig, axs = plt.subplots(nrows=2,ncols=2,figsize=figsize)
  
  metrics = ['acc','ece','brier','nll']
  for metric,ax in zip(metrics,axs.reshape(-1)):
    genplot_c(dfr,metric,ax)
    ax.set_yticklabels('')
    ax.xaxis.grid(True)

#   genplot_c(dfr,'ece',axs[1],ascending=False)
#   genplot_c(dfr,'brier',axs[2],ascending=False)
#   genplot_c(dfr,'nll',axs[3],ascending=False)
  #axs[3].set_yticklabels('')
    
  for ax in axs.reshape(-1):
    for line in ax.get_ygridlines():
      line.set_linewidth(0)
    
  fig.legend(axs[0,0].artists,
             dfr.columns[2:],
             bbox_to_anchor=bbox,
             borderaxespad=0,
             fontsize=legend_fontsize,
             ncol=4,
             loc='lower center')
  plt.tight_layout()
  return fig,axs



def pd_merge(dfs,**kwargs):
    dftemp = pd.merge(left=dfs[0],right=dfs[1],**kwargs)
    if len(dfs)>2:
        return pd_merge([dftemp]+dfs[2:],**kwargs)
    else:
        return dftemp