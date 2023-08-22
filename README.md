# Improving Out-of-Distribution Detection with Margin-Based Prototype Learning

This repository contains the offical PyTorch implementation of paper:
Improving Out-of-Distribution Detection with Margin-Based Prototype Learning. ICONIP 2023

This code is developed based on the code of [CIDER](https://github.com/deeplearning-wisc/cider), and we appreciate their contributions very much. You can download the datasets and find the directory structure there.

## Training

````
sh scripts/train_cider_cifar10.sh
sh scripts/train_cider_cifar100.sh
````

## Evaluation
````
sh scripts/eval_ckpt_cifar10.sh ckpt_c10 #for CIFAR-10
sh scripts/eval_ckpt_cifar100.sh ckpt_c100 # for CIFAR-100
````