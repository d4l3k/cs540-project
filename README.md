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

Parzen estimators extend the Parzen window method in a logical way (1). The
simple method is to pick a single model over the whole data range. The tree
structure means picking one model for all data points where the model has low
enough error, and picking a different model or set of models for the rest. In
this way, TPE is kind of a blend of decision trees and the kernel method.

## Bayesian Optimization

Bayesian optimization in general is the process of optimizing over the posterior
distribution directly, as opposed to maximizing the likelihood as a way to
approximate optimizing the posterior. The library we're using, in particular,
implements a Student-T process with a spacial correlation model, picking the
next evaluation point on the basis of expected improvement (3).

## Random Search

This is our baseline method, where we pick a bunch of random points. Every other
method should perform better than this method.

## Gradient-Boosted Decision Trees

(Not too sure about this wording, double check it)

Gradient boosted decision trees is a mixture of shallow decision trees, where
the mixture itself is optimized using the gradient of the loss function (2).
This is like a more informed kernel trick, but an interesting generalization of
it to "weak learners" rather than just some Gaussian basis function.

# Citations

1. Bergstra, James S., et al. "Algorithms for hyper-parameter optimization."
   Advances in neural information processing systems. 2011.

2. Friedman, Jerome H. "Greedy function approximation: a gradient boosting
   machine." Annals of statistics (2001): 1189-1232.

3. Martinez-Cantin, Ruben. "Bayesopt: A bayesian optimization library for
   nonlinear optimization, experimental design and bandits." The Journal of
   Machine Learning Research 15.1 (2014): 3735-3739.