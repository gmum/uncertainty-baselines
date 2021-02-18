# SLOVA project
#### EXPERIMENT: SLOVA study on shifted dataset CIFAR10-C
Experiment trains a ResNet20 network on CIFAR10 dataset. Evaluation metrics on CIFAR10-C are Accuracy, ECE, Brier Score and Negative Log Likelihood.
#### Setup
To create miniconda slova-env
> conda env create -f slova-env.yml

#### Training SLOVA on CIFAR10
> python train.py --config cfg/slova.json

Results are saved in dir ``slova``.
#### Metric evaluation on CIFAR10-C
After training, to evaluate on usual metrics
> python comp-metrics.py --config slova/FLAGS.json --phase cal_eval

Results are saved to a ``csv`` file in ``slova`` dir.

#### Plotting Figure 5
* To readily plot part of Fig. 5 containing only our results (stored in ``metrics_summary``)
> python plot_fig-shift.py --no_benchmark

* Complete Fig. 5 with benchmarks taken from ["Can You Trust Your Model’s Uncertainty? Evaluating
Predictive Uncertainty Under Dataset Shift"](https://papers.nips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) 
> python plot_fig-shift.py --benchmark_file path_to_hdf5_file

#### Other models
Other implemented models include OVA DM and softmax found in ``cfg`` dir.
