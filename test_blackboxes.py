#!/usr/bin/env python3
print("loading")

import argparse
import time

import bayes_opt
import catboost
import hyperopt
import numpy as np
import scipy.linalg
import scipy.optimize
import gpflow
import tensorflow as tf

from sparklines import sparklines
from tabulate import tabulate
from tensorflow.contrib import rnn

from models import deathrate

init_years = 5
num_years = 20
sample_points = 100

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


def predict(x):
    global total_predict_calls, last_y
    total_predict_calls += 1

    return model.predict(x)


def clear_predict():
    global total_predict_calls, last_y
    total_predict_calls = 0
    model.reset()

random_state = np.random.RandomState()

def random_points(n):
    return random_state.uniform(model.bounds[:, 0], model.bounds[:, 1], size=(n, model.bounds.shape[0]))

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
        for i, bound in enumerate(model.bounds)
    ], algo=hyperopt.tpe.suggest, max_evals=(num_years + init_years))

    report('Tree of Parzen Estimators', predict(tpe_x)[init_years:], time.time() - start, total_predict_calls)


def bayesian_opt():
    """Bayesian Optimization"""
    clear_predict()

    # Bayesian Optimization
    start = time.time()
    bo = bayes_opt.BayesianOptimization(
        lambda a, b, c, d: -predict([[a, b, c, d]])[0],
        {'a': model.bounds[0], 'b': model.bounds[1], 'c': model.bounds[2], 'd': model.bounds[3]},
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
        x = random_state.uniform(model.bounds[:, 0], model.bounds[:, 1], size=(1, model.bounds.shape[0]))
        y[i] = predict(x)

    report('Random Search', y, time.time() - start, total_predict_calls)


def gbdt():
    """GBDT"""
    clear_predict()
    start = time.time()

    x = random_points(init_years)
    y = np.zeros(init_years)

    # generate each initial year separately
    for i in range(init_years):
        y[i] = predict(np.reshape(x[i, :], (1, x.shape[1])))

    gbdt_model = catboost.CatBoostRegressor(verbose=True)
    for i in range(num_years):
        gbdt_model.fit(x, y)
        x_tmps = []
        for test_x in random_points(sample_points):
            res = scipy.optimize.minimize(lambda p_x:
                    gbdt_model.predict([p_x])[0], test_x, bounds=model.bounds)
            if not res.success:
                continue
            x_tmps += [res.x]
        best_i = int(np.argmin(gbdt_model.predict(x_tmps)))
        best_x = x_tmps[best_i]
        x = np.append(x, [best_x], axis=0)
        y_tmp = predict([best_x])
        y = np.append(y, y_tmp)

    report('GBDT', y[init_years:], time.time() - start, total_predict_calls)


def lstm():
    """LSTM RNN Strategy"""
    clear_predict()

    g_functions = []

    opt = gpflow.train.ScipyOptimizer()

    sess = tf.Session()

    init = tf.global_variables_initializer()

    sess.run(init)

    print("sampling GPs")

    # First create a sample of functions
    for i in range(init_years):
        # Create a gaussian process
        x = random_points(init_years)
        y = np.zeros((init_years, 1))

        # Generate our random points
        for i in range(init_years):
            y[i, 0] = predict(np.reshape(x[i, :], (1, x.shape[1])))

        with gpflow.defer_build():
            # Initialize our model
            f = gpflow.models.GPR(x, y, kern=gpflow.kernels.RBF(x.shape[1]))

        # Compile and optimize it
        f.compile(session=sess)

        opt.minimize(f)

        # Save it
        g_functions.append(f)

    print("creating LSTM")

    # Size of our input, plus one for the corresponding y value
    d_input = model.bounds.shape[0]

    # How many initial years we're training on
    batch_size = init_years

    # Output size of the LSTM, which we want to have predict x values, so match that
    num_units = d_input

    learning_rate = 0.001

    # How long we want to predict for
    time_steps = num_years

    # Create a placeholder for our starting point
    start = tf.placeholder(tf.float64, [batch_size, d_input])

    cell = rnn.BasicLSTMCell(num_units, forget_bias=1)

    def batch_predict(position):
        preds = []

        for i in range(batch_size):
            # TODO this is producing pred_mean with shape (4,)??? Shouldn't it be (1,)?
            pred_mean, pred_var = g_functions[i]._build_predict(tf.reshape(position[i], (position[i].shape[0], 1)))
            pred_mean, _ = g_functions[i].likelihood.predict_mean_and_var(pred_mean, pred_var)
            preds.append(pred_mean)

        return tf.stack(preds)

    def lstm_loop(time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:
            next_cell_state = cell.zero_state(batch_size, tf.float64)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= time_steps)
        elements_first = (time == 0)
        finished = tf.reduce_all(elements_finished)
        first = tf.reduce_all(elements_first)

        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, d_input + 1], dtype=tf.float64),
            lambda: tf.cond(
                first,
                lambda: tf.concat(start, batch_predict(start)),
                lambda: tf.concat(cell_output, batch_predict(cell_output))
            )
        )

        next_loop_state = None

        return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    # these outputs will be in the shape [time_step, batch_size, num_units]
    outputs, _, _ = tf.nn.raw_rnn(cell, lstm_loop)

    predictions = [batch_predict(outputs[t]) for t in range(time_steps)]

    # Try to make the best point better
    loss = tf.reduce_min(tf.stack(predictions))

    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    print("training")

    # Train
    for iter in range(10):
        # Sample a bunch of random starting points
        batch_start = random_points(batch_size)

        sess.run(opt, feed_dict={start: batch_start})

    # Test
    test_start = random_points(1)
    print("Loss:", sess.run(loss, feed_dict={start: test_start}))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tpe", help="run tree of parzen estimators", action="store_true")
    parser.add_argument("--bayesian", help="run bayesian optimization", action="store_true")
    parser.add_argument("--random-search", help="run random search", action="store_true")
    parser.add_argument("--gbdt", help="run gradient boosted decision trees", action="store_true")
    parser.add_argument("--lstm", help="run LSTM method", action="store_true")

    args = parser.parse_args()

    global model
    model = deathrate.Model()

    run_all = not (args.tpe or args.bayesian or args.random_search or args.gbdt or args.lstm)

    if run_all or args.tpe:
        print("running TPE")
        tpe()

    if run_all or args.bayesian:
        print("running bayesian optimization")
        bayesian_opt()

    if run_all or args.random_search:
        print("running random search")
        random_search()

    if run_all or args.gbdt:
        print("running gbdt")
        gbdt()

    if run_all or args.lstm:
        print("running lstm")
        lstm()

    print_results()


if __name__ == '__main__':
    main()
