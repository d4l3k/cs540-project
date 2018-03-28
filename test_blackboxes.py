import catboost
import bayes_opt
import numpy as np

model = catboost.CatBoostRegressor()
model.load_model('deathrate.cbm')

def report(method, data):
    print('{} - {} dead over {} years', method, np.sum(abs(data)), len(data))

# Bayesian Optimization
bo = bayes_opt.BayesianOptimization(
    lambda a, b, c, d: -model.predict([[a, b, c, d]])[0],
    {'a': (0, 28), 'b': (0, 34), 'c': (0, 9), 'd': (0, 30)},
)

bo.maximize(init_points=1, n_iter=25)

report('BayesianOptimization', bo.Y)

import ipdb; ipdb.set_trace()
