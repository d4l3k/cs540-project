---
title: Comparison of Optimization Techniques for Stateful Black Boxes
date: "May 2018"
author: Jerome Rasky 42386145, Tristan Rice 25886145

header-includes:
   - \usepackage[final]{nips_2017}
   - \usepackage[utf8]{inputenc}
   - \usepackage[T1]{fontenc}
   - \usepackage{hyperref}
   - \usepackage{url}
   - \usepackage{booktabs}
   - \usepackage{amsfonts}
   - \usepackage{nicefrac}
   - \usepackage{microtype}
---

\begin{abstract}
Current research into global black box optimization of expensive functions
focuses on stateless models such as machine learning hyper parameter
optimization. Many real world problems can't be reset back to a clean state
after a single evaluation. In this paper, we evaluate the performance of five
popular black box algorithms on two stateful models.
\end{abstract}

# Introduction

# Related Work

# Description and Justification

## Algorithms

### Random Search

Random search is one of the simplest black box optimization algorithms. It
randomly guesses points within the bounds and then takes the highest value as
the best option. This is used as a baseline for the other algorithms.

### Tree of Parzen Estimators

### Gradient Boosted Decision Trees

### Bayesian Optimization (Gaussian Processes)

### LSTM based Recurrent Neural Network

We implemented the model as described in @bb-rnn. The problem
 of 
black box 
optimization can be formulated as the problem of finding the sequence 
containing the minimum value of a black box function. This formulation can be
used to fit a Long Short Term Memory neural network. This method essentially 
uses an LSTM to learn how to minimize the function, rather than using a 
gradient.

At ever step, the rnn $LSTM$ determines the next step to take.

\begin{align*}
x_n, h_n &= LSTM(x_{n - 1}, y_{n - 1}, h_{n - 1})\\
y_n &= \psi(x_n)
\end{align*}

Where $\psi(x) = E[GP(x)]$, the expected value of the Gaussian process model at
point $x$.

Our implementation of this method uses Tensorflow 
[@tensorflow2015-whitepaper], making use of its LSTM framework. We also 
make use of GPflow [@GPflow2017] for creating our Gaussian processes.

This method is obviously heavily dependent on the exact form of the Gaussian 
process. We started with the simplest case of a "vanilla" Gaussian process using
radial basis functions for the kernels. This approach yielded poor results, 
since with only a few data points to train on, most of the function space is 
predicted to be zero. Using a summed Matern and Linear kernel, performance can
beat Bayesian optimization.

## Models

To evaluate these algorithms we trained them on two stateful models. Each model
has a single function that needs to be optimized with $n$ parameters as well as
bounds for the range of values that can be accepted.

### Death Rate

Many real life models, such as governmental budgeting are nearly impossible to
evaluate so we had to come up with an approximation to it. We created a simple
model by using the World Bank Development Indicators and training a Gradient
Boosted Decision Tree to predict deathrate based off of expenditures in
education, health, R&D and military [@WB]. To add a stateful component to this,
we added momentum, such that changing the parameters produces lag with respect
to the death rate.

While this is a very simple model, it does provide some realistic behaviors.
Many large systems have a high latency between cause and effect. We're also
primarily interested in highlighting the differences between these algorithms.

We bound the inputs to be within two times the maximum existing value for that
expenditure category and greater than zero.

### Airplane

The second model is that of flying an airplane. The Python Flight Mechanics
Engine is a project attempting to model every aspect of flying a plane [@PyFME].
We used it to model flying a plane to a location. The model takes in the
throttle as well as the position of the elevator, aileron and rudder and outputs
the distance the plane has flown towards the target. This model has numerous
stateful variables that need to be modeled including position, rotation,
velocity, direction. There are also many nonlinearities due to air resistance
and gravity.

This model uses the Cessna 172 as a base and bounds the inputs to be match the
actual control range.

# Experiments and Analysis

## Death Rate

## Airplane

# Discussion and Future Work

# References
