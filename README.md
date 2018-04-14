# CPSC 540 Project

By Jerome Rasky and Tristan Rice

# Problem Description

For our project, we want to study blackbox methods on stateful models. This is
particularly applicable to situations such as government budgeting, where model
evaluation is very expensive and stateful. Since we don't have a government that
we can experiment with, we'll do the next best thing: simulate having that
government. We'll start by building a stateful model that we can evaluate, and
then test different blackbox methods against the model we generate. This method
means that we'll be able to test different blackbox methods in a short period of
time, but still produce a valuable survey of methods available in the space.

## Dataset

We'll use World Bank data on deaths per 1000, and expenditure as a percentage of
GDP in education, health, research and development, and the military. To make
data access faster, we'll cache these datasets locally after fetching them once
from the World Bank.

# Generative Model

Our generative model right now is gradient boosted decision trees, as
implemented in the catboost package. We plan to add momentum as the stateful
component, and possibly to explore other more interesting generative approaches.

# Blackbox Methods

## Tree of Parzen Estimators

## Bayesian Optimization

## Random Search

## Gradient-Boosted Decision Trees