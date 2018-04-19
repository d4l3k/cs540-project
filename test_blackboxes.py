#!/usr/bin/env python3
import catboost
import bayes_opt
import numpy as np
import scipy.optimize
import scipy.linalg
import hyperopt
from sparklines import sparklines
from tabulate import tabulate
import time
import argparse

model = catboost.CatBoostRegressor(verbose=False)
model.load_model('deathrate.cbm')

init_years = 5
num_years = 20
sample_points = 100
momentum = .10

results = []


def report(*args):
    global results
    results += [args]
    print_result(*args)


def format_result(method, data, time_taken, predict_calls):
    return [method, np.sum(data), data[len(data) - 1], len(data),
            '\n'.join(sparklines(data)), time_taken, predict_calls]


def print_result_table(table):
    print(tabulate(table, headers=['Method', 'Dead per 1000', 'Final', 'Years',
                                   'Distribution', 'Time (s)', 'Calls to Predict']))


def print_result(*args):
    print_result_table([format_result(*args)])


def print_results():
    print_result_table([format_result(*res) for res in results])


total_predict_calls = 0
last_y = None


def predict(x):
    global total_predict_calls, last_y
    total_predict_calls += 1

    y = model.predict(x)

    if last_y is None:
        last_y = y
    else:
        # Add some momentum year-on-year
        y += momentum * last_y
        last_y = y

    return y


def clear_predict():
    global total_predict_calls, last_y
    total_predict_calls = 0
    last_y = None


bounds = np.array([(0, 14.06), (2.23, 17.14), (0.02, 4.41), (0.13, 14.81)])
random_state = np.random.RandomState()

# TODO better initialization points
tpe_x = []


def tpe():
    """Tree of Parzen Estimators"""
    clear_predict()

    start = time.time()
    global tpe_x
    tpe_x = []

    def tpe_objective(args):
        global tpe_x
        tpe_x += [args]
        return predict([args])[0]

    hyperopt.fmin(tpe_objective, space=[
        hyperopt.hp.uniform(str(i), bound[0], bound[1])
        for i, bound in enumerate(bounds)
    ], algo=hyperopt.tpe.suggest, max_evals=(num_years + init_years))

    report('Tree of Parzen Estimators', predict(tpe_x)[init_years:], time.time() - start, total_predict_calls)


def bayesian_opt():
    """Bayesian Optimization"""
    clear_predict()

    # Bayesian Optimization
    start = time.time()
    bo = bayes_opt.BayesianOptimization(
        lambda a, b, c, d: -predict([[a, b, c, d]])[0],
        {'a': bounds[0], 'b': bounds[1], 'c': bounds[2], 'd': bounds[3]},
    )

    bo.maximize(init_points=init_years, n_iter=num_years)
    report('Bayesian Optimization', -bo.space.Y[init_years:], time.time() - start, total_predict_calls)


def random_search():
    """RandomSearch (baseline, every model should do better than this)"""
    clear_predict()
    start = time.time()
    y = np.zeros(num_years)

    for i in range(num_years):
        # Evaluate each year separately
        x = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(1, bounds.shape[0]))
        y[i] = predict(x)

    report('Random Search', y, time.time() - start, total_predict_calls)


def gbdt():
    """GBDT"""
    clear_predict()
    start = time.time()

    def random_points(n):
        return random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))

    x = random_points(init_years)
    y = np.zeros(init_years)

    # generate each initial year separately
    for i in range(init_years):
        y[i] = predict(np.reshape(x[i, :], (1, x.shape[1])))

    gbdt_model = catboost.CatBoostRegressor(verbose=False)
    for i in range(num_years):
        gbdt_model.fit(x, y)
        x_tmps = []
        for test_x in random_points(sample_points):
            res = scipy.optimize.minimize(lambda p_x: gbdt_model.predict([p_x])[0], test_x, bounds=bounds)
            if not res.success:
                continue
            x_tmps += [res.x]
        best_i = int(np.argmin(gbdt_model.predict(x_tmps)))
        best_x = x_tmps[best_i]
        x = np.append(x, [best_x], axis=0)
        y_tmp = predict([best_x])
        y = np.append(y, y_tmp)

    report('GBDT', y[init_years:], time.time() - start, total_predict_calls)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tpe", help="run tree of parzen estimators", action="store_true")
    parser.add_argument("--bayesian", help="run bayesian optimization", action="store_true")
    parser.add_argument("--random-search", help="run random search", action="store_true")
    parser.add_argument("--gbdt", help="run gradient boosted decision trees", action="store_true")

    args = parser.parse_args()

    run_all = not (args.tpe or args.bayesian or args.random_search or args.gbdt)

    if run_all or args.tpe:
        tpe()

    if run_all or args.bayesian:
        bayesian_opt()

    if run_all or args.random_search:
        random_search()

    if run_all or args.gbdt:
        gbdt()

    print_results()


if __name__ == '__main__':
    main()
