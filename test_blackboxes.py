import catboost
import bayes_opt
import numpy as np
import scipy.optimize
import hyperopt

model = catboost.CatBoostRegressor()
model.load_model('deathrate.cbm')

init_years = 2
num_years = 10
sample_points = 100

results = []
def report(method, data):
    global results
    results += [(method, data)]
    print_result(method, data)

def print_result(method, data):
    print('{} - {} dead over {} years'.format(method, np.sum(abs(data)), len(data)))

def print_results():
    global results
    for method, data in results:
        print_result(method, data)

bounds = np.array([(0, 14.06), (2.23, 17.14), (0.02, 4.41), (0.13, 14.81)])
random_state = np.random.RandomState()

# TODO: give a realistic initialization point

# Tree of Parzen Estimators
x = []
def tpe_objective(args):
    global x
    x += [args]
    return model.predict([args])[0]

hyperopt.fmin(tpe_objective, space=[
    hyperopt.hp.uniform(str(i), bound[0], bound[1]) for i, bound in enumerate(bounds)
], algo=hyperopt.tpe.suggest, max_evals=(num_years+init_years))

report('Tree of Parzen Estimators', model.predict(x)[init_years:])

# Bayesian Optimization
bo = bayes_opt.BayesianOptimization(
    lambda a, b, c, d: -model.predict([[a, b, c, d]])[0],
    {'a': bounds[0], 'b': bounds[1], 'c': bounds[2], 'd': bounds[3]},
)
bo.maximize(init_points=init_years, n_iter=num_years)
report('Bayesian Optimization', bo.Y[init_years:])

# RandomSearch
x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(num_years, bounds.shape[0]))
y = model.predict(x_tries)
report('Random Search', y)

# GBDT
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

report('GBDT', y[init_years:])

print_results()

import ipdb; ipdb.set_trace()
