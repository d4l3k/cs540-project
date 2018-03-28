import catboost
import bayes_opt
import numpy as np
import scipy.optimize

model = catboost.CatBoostRegressor()
model.load_model('deathrate.cbm')

init_years = 2
num_years = 10

results = []
def report(method, data):
    global results
    results += [(method, data)]

def print_results():
    global results
    for method, data in results:
        print('{} - {} dead over {} years'.format(method, np.sum(abs(data)), len(data)))

bounds = np.array([(0, 14.06), (2.23, 17.14), (0.02, 4.41), (0.13, 14.81)])

# TODO: give a realistic initialization point

# Bayesian Optimization
bo = bayes_opt.BayesianOptimization(
    lambda a, b, c, d: -model.predict([[a, b, c, d]])[0],
    {'a': bounds[0], 'b': bounds[1], 'c': bounds[2], 'd': bounds[3]},
)
bo.maximize(init_points=init_years, n_iter=num_years)
report('BayesianOptimization', bo.Y[1:])

# RandomSearch
random_state = np.random.RandomState()
x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(num_years, bounds.shape[0]))
y = model.predict(x_tries)
report('RandomSearch', y)

# GBDT
x = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(init_years, bounds.shape[0]))
y = model.predict(x)
gbdt_model = catboost.CatBoostRegressor()
for i in range(num_years):
    print('x = {}, y = {}'.format(x, y))
    gbdt_model.fit(x, y)
    res = scipy.optimize.minimize(lambda x: gbdt_model.predict([x])[0], x[len(x)-1], bounds=bounds)
    if not res.success:
        continue
    x += [res.x]
    y += model.predict([res.x])

report('GBDT', y)

print_results()

import ipdb; ipdb.set_trace()
