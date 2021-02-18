import argparse

import func_plot

parser = argparse.ArgumentParser(description='SLOVA CIFAR-10 ResNet-20 plot fig-shift')
parser.add_argument('--no_benchmark',default=False,action='store_true')
parser.add_argument("--benchmark_file", type=str, help="path to benchmark hdf5file (https://papers.nips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)",default='metrics_googlefig/cifar_model_predictions.hdf5')
parser.add_argument("--metrics_dir", type=str, help="dir with csv files with metrics",default='metrics_summary')
parser.add_argument("--fig_file",type=str,help="save figure to file", default='fig-shift.eps')
args = parser.parse_args()

# include benchmark
incl_benchmark = not args.no_benchmark

# file with CIFAR10 results of benchmark models taken from https://papers.nips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf
benchmark_hdf5file = args.benchmark_file

# dir with csv files
metrics_dir = args.metrics_dir

# fig_file
fig_file = args.fig_file


#load data
df_dict = func_plot.load_df_dict(metrics_dir)
df_dict_pivot = func_plot.df_dict_to_pivot(df_dict)

if incl_benchmark:
    df_dict_pivot_googlefig = func_plot.load_pivot_df_dict_googlefig(hdf5file=benchmark_hdf5file)
else:
    df_dict_pivot_googlefig = {}
    
#gather all models
rename_dict = {
               'slova-example_metrics' : 'SLOVA',
               'ovadm-example_metrics': 'OVA DM'
              }

df_dict_pivot = func_plot.rename_df_dict(df_dict=df_dict_pivot,rename_dict=rename_dict)

df_summary = {**df_dict_pivot,**df_dict_pivot_googlefig} if incl_benchmark else df_dict_pivot

#plot
metrics = ['acc','ece','nll','brier']
model_list_googlefig = ['vanilla',
              'temp_scaling',
              'ensemble',
              'dropout_nofirst',
              'll_dropout',
              'svi',
              'll_svi',
              ]

model_list_our = ['OVA DM', 'SLOVA']
model_list = model_list_googlefig + model_list_our if incl_benchmark else model_list_our

figsize_y = 4*len(metrics)
fig, axs = func_plot.cifar10c_shift_plots(df_summary,metrics,model_list,figsize=(15,figsize_y))

fig.savefig(fig_file)