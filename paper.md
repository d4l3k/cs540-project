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

Citing some paper [@Jones1998]

# Description and Justification

## Models

To evaluate these algorithms we trained them on two stateful models.

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

### Airplane

The second model is that of flying an airplane. The Python Flight Mechanics
Engine is a project attempting to model every aspect of flying a plane [@PyFME].
We used it to model flying a plane to a location. The model takes in the
throttle as well as the position of the elevator, aileron and rudder and outputs
the distance the plane has flown towards the target. This model has numerous
stateful variables that need to be modeled including position, rotation,
velocity, direction. There are also many nonlinearities due to air resistance
and gravity.

# Experiments and Analysis

# Discussion and Future Work

# References
