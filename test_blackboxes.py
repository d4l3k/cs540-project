import catboost
import bayes_opt
import numpy as np

model = catboost.CatBoostRegressor()
model.load_model('deathrate.cbm')

num_years = 10

def report(method, data):
    print('{} - {} dead over {} years'.format(method, np.sum(abs(data)), len(data)))

bounds = np.array([(0, 14.06), (2.23, 17.14), (0.02, 4.41), (0.13, 14.81)])

# Bayesian Optimization
bo = bayes_opt.BayesianOptimization(
    lambda a, b, c, d: -model.predict([[a, b, c, d]])[0],
    {'a': bounds[0], 'b': bounds[1], 'c': bounds[2], 'd': bounds[3]},
)

bo.maximize(init_points=1, n_iter=num_years)

report('BayesianOptimization', bo.Y[1:])


random_state = np.random.RandomState()
x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(num_years, bounds.shape[0]))
y = model.predict(x_tries)

report('RandomSearch', y)



import ipdb; ipdb.set_trace()
