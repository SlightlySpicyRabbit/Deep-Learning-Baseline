# Deep-Learning-Baseline

## Overview
During the training of deep neural networks, we usually only need to modify the neural network architecture and the hyperparameters of the optimizer. A large amount of code stored in the main program is not only not conducive to code reuse, but also affects the code reading. This project is part of the deep learning Standard Operating Procedure (SOP). It want to provide a model training and evaluation tool that can be invoked repeatedly.

## What it can do
Train your neural networks easily and supports multiple optimizers:
* SGD
* Momentum
* RMSprop
* Adam

It can be used to check the loss value on validation set during training process (This will help you to observe overfitting)

Visualization shows the change in loss value between the training set and the validation set during training process

Evaluate the model on the specified dataset, and provide various evaluation ways such as loss, accuracy, and confusion matrix

## Runtime Environment
* PyTorch
* sklearn
* matplotlib
