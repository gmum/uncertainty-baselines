# SLOVA project
#### EXPERIMENT: SLOVA study on shifted dataset CIFAR10-C
Experiment trains a ResNet20 network on CIFAR10 dataset. Evaluation metrics on CIFAR10-C are Accuracy, ECE, Brier Score and Negative Log Likelihood.
#### Training SLOVA on CIFAR10
> python train.py --config cfgs/slova.json

Results are saved in dir ``slova``.
#### Metric evaluation on CIFAR10-C
After training, to evaluate usual metrics run
> python comp-metrics.py --config slova/FLAGS.json --phase cal_eval
Results are saved to a ``csv`` file in ``slova`` dir.
