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
