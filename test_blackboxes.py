import catboost
import bayes_opt
import numpy as np
import scipy.optimize
import hyperopt
from sparklines import sparklines
from tabulate import tabulate
import time

model = catboost.CatBoostRegressor()
model.load_model('deathrate.cbm')

init_years = 5
num_years = 20
sample_points = 100

results = []

def report(*args):
    global results
    results += [args]
    print_result(*args)


def format_result(method, data, time_taken):
    return [method, np.sum(data), data[len(data)-1], len(data),
            '\n'.join(sparklines(data)), time_taken]


def print_result_table(table):
    print(tabulate(table, headers=['Method', 'Dead per 1000', 'Final', 'Years',
        'Distribution', 'Time (s)']))


def print_result(*args):
    print_result_table([format_result(*args)])


def print_results():
    print_result_table([format_result(*res) for res in results])


bounds = np.array([(0, 14.06), (2.23, 17.14), (0.02, 4.41), (0.13, 14.81)])
random_state = np.random.RandomState()

# TODO: give a realistic initialization point

# Tree of Parzen Estimators
start = time.time()
x = []


def tpe_objective(args):
    global x
    x += [args]
    return model.predict([args])[0]


hyperopt.fmin(tpe_objective, space=[
    hyperopt.hp.uniform(str(i), bound[0], bound[1])
    for i, bound in enumerate(bounds)
], algo=hyperopt.tpe.suggest, max_evals=(num_years+init_years))

report('Tree of Parzen Estimators', model.predict(x)[init_years:], time.time() - start)

# Bayesian Optimization
start = time.time()
bo = bayes_opt.BayesianOptimization(
    lambda a, b, c, d: -model.predict([[a, b, c, d]])[0],
    {'a': bounds[0], 'b': bounds[1], 'c': bounds[2], 'd': bounds[3]},
)
bo.maximize(init_points=init_years, n_iter=num_years)
report('Bayesian Optimization', -bo.Y[init_years:], time.time() - start)

# RandomSearch (baseline, every model should do better than this)
start = time.time()
x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(num_years, bounds.shape[0]))
y = model.predict(x_tries)
report('Random Search', y, time.time() - start)

# GBDT
start = time.time()
def random_points(n):
    return random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))

x = random_points(init_years)
y = model.predict(x)
gbdt_model = catboost.CatBoostRegressor()
for i in range(num_years):
    gbdt_model.fit(x, y)
    x_tmps = []
    for test_x in random_points(sample_points):
        res = scipy.optimize.minimize(lambda x: gbdt_model.predict([x])[0], test_x, bounds=bounds)
        if not res.success:
            continue
        x_tmps += [res.x]
    best_i = np.argmin(gbdt_model.predict(x_tmps))
    best_x = x_tmps[best_i]
    x = np.append(x, [best_x], axis=0)
    y_tmp = model.predict([best_x])
    y = np.append(y, y_tmp)

report('GBDT', y[init_years:], time.time() - start)

print_results()

import ipdb; ipdb.set_trace()
