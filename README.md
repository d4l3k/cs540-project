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

Parzen estimators extend the Parzen window method in a logical way (1). The simple
method is to pick a single model over the whole data range. The tree structure
means picking one model for all data points where the model has low enough
error, and picking a different model or set of models for the rest. In this way,
TPE is kind of a blend of decision trees and the kernel method.

## Bayesian Optimization

## Random Search

## Gradient-Boosted Decision Trees

(Not too sure about this wording, double check it)

Gradient boosted decision trees is a mixture of shallow decision trees, where
the mixture itself is optimized using the gradient of the loss function (2). This is
like a more informed kernel trick, but an interesting generalization of it to
"weak learners" rather than just some Gaussian basis function.

# Citations

1. Bergstra, James S., et al. "Algorithms for hyper-parameter optimization."
   Advances in neural information processing systems. 2011.

2. Friedman, Jerome H. "Greedy function approximation: a gradient boosting
   machine." Annals of statistics (2001): 1189-1232.