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
import matplotlib.pyplot as plt

from sparklines import sparklines
from tabulate import tabulate
from tensorflow.contrib import rnn

from models import deathrate
from models import airplane

init_years = 5
num_years = 20
sample_points = 100
epoch = 0

results = []


def report(*args):
    global results
    args = args + (epoch,)
    results.append(args)
    print_result(*args)


def format_result(method, data, time_taken, predict_calls, epoch):
    return [method, np.sum(data), data[len(data) - 1], len(data),
            '\n'.join(sparklines(data)), time_taken, predict_calls, epoch]


def print_result_table(table):
    print(tabulate(table, headers=['Method', model.label, 'Final', 'Years',
                                   'Distribution', 'Time (s)', 'Calls to Predict', 'Epoch']))


def print_result(*args):
    print_result_table([format_result(*args)])


def print_results():
    print_result_table([format_result(*res) for res in results])

def graph_results():
    labels = []
    for result in results:
        method = result[0]
        data = result[1]
        epoch = result[4]

        if epoch > 0:
            method += ' (Epoch {})'.format(epoch)

        labels.append(method)
        plt.plot(range(0,len(data)), data)
    plt.legend(labels)
    plt.title('Algorithm Performance (Smaller is better)')
    plt.xlabel('Iteration')
    plt.ylabel(model.label)
    plt.show()



total_predict_calls = 0


def predict(x):
    global total_predict_calls
    total_predict_calls += 1

    clamped = np.clip(x.copy(), model.bounds[:, 0], model.bounds[:, 1])

    if not np.array_equal(clamped, x):
        print('predict: Clamped {} to {}'.format(x, clamped))

    return model.predict(clamped)

def clear_predict():
    global total_predict_calls
    total_predict_calls = 0
    model.reset()

random_state = np.random.RandomState()

def random_points(n):
    return random_state.uniform(model.bounds[:, 0], model.bounds[:, 1], size=(n, model.bounds.shape[0]))


def tpe():
    """Tree of Parzen Estimators"""
    clear_predict()

    start = time.time()
    tpe_x = []
    tpe_values = []

    def tpe_objective(args):
        tpe_x.append(args)
        val = predict([args])[0]
        tpe_values.append(val)
        return val

    hyperopt.fmin(tpe_objective, space=[
        hyperopt.hp.uniform(str(i), bound[0], bound[1])
        for i, bound in enumerate(model.bounds)
    ], algo=hyperopt.tpe.suggest, max_evals=(num_years + init_years))

    report('Tree of Parzen Estimators', tpe_values[init_years:], time.time() - start, total_predict_calls)


def bayesian_opt(iter_feature):
    """Bayesian Optimization"""
    clear_predict()

    bounds = {
        'a': model.bounds[0],
        'b': model.bounds[1],
        'c': model.bounds[2],
        'd': model.bounds[3],
        'iteration': [0, 0]
    }

    def update_iteration():
        bounds['iteration'] = [total_predict_calls, total_predict_calls]
        bo.set_bounds(bounds)


    def bo_predict(a, b, c, d, iteration):
        val = -predict([[a, b, c, d]])[0]
        if iter_feature:
            update_iteration()
        return val


    # Bayesian Optimization
    start = time.time()
    bo = bayes_opt.BayesianOptimization(
        bo_predict,
        bounds,
    )

    bo.maximize(init_points=1, n_iter=(init_years-1+num_years))
    method = 'Bayesian Optimization'
    if iter_feature:
        method += ' (Iteration Feature)'
    report(method, -bo.space.Y[init_years:], time.time() - start, total_predict_calls)


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

    gbdt_model = catboost.CatBoostRegressor(verbose=False)
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
    start_time = time.time()

    g_functions = []

    # Size of our input, plus one for the corresponding y value
    d_input = model.bounds.shape[0]

    # How many initial years we're training on
    batch_size = 1

    # Output size of the LSTM, which we want to have predict x values, so match that
    num_units = d_input

    # How long we want to predict for
    # TODO add init years to this to make things a bit better at the beginning
    time_steps = num_years

    # Create a placeholder for our starting point
    # Use an undecided batch size
    start = tf.placeholder(tf.float64, [batch_size, d_input])

    cell = rnn.BasicLSTMCell(num_units, forget_bias=1)

    opt = gpflow.train.ScipyOptimizer()

    sess = tf.Session()

    feeds = {}

    print("sampling GPs")

    position_bias = model.bounds[:, 0]
    position_weight = model.bounds[:, 1] - model.bounds[:, 0]

    def normalize_position(x):
        return 2. * np.divide((x - position_bias), position_weight) - 1.

    def denormalize_position(x):
        return np.multiply((x + 1.) / 2., position_weight) + position_bias

    # First create a sample of functions
    for i in range(batch_size):
        # Create a gaussian process
        x_sample = random_points(init_years)
        x = np.zeros(x_sample.shape)
        y = np.zeros((init_years, 1))

        # Generate our random points
        for i in range(init_years):
            y[i, 0] = predict(np.reshape(x_sample[i, :], (1, x.shape[1])))
            # normalize the position
            x[i, :] = normalize_position(x_sample[i, :])

        with gpflow.defer_build():
            # Initialize our model
            kern = gpflow.kernels.Matern32(x.shape[1]) + gpflow.kernels.Linear(x.shape[1])
            f = gpflow.models.GPR(x, y, kern=kern)

        # Compile it
        f.compile(session=sess)

        # Include this function's feeds into ours
        feeds.update(f.initializable_feeds)

        opt.minimize(f)

        # Save it
        g_functions.append(f)

    print("creating LSTM")

    def step_constraint_violation(step):
        return tf.reduce_sum(tf.maximum(1. - tf.abs(step), tf.ones(step.shape, tf.float64)) - 1.)

    def batch_predict(position):
        preds = []

        for i in range(batch_size):
            this_position = tf.reshape(position[i], (1, position[i].shape[0]))

            pred_mean, pred_var = g_functions[i]._build_predict(this_position)
            pred_mean, _ = g_functions[i].likelihood.predict_mean_and_var(pred_mean, pred_var)

            # We know the dimensionality here, but GPFlow loses track of it
            # As well, turn these into one-dimensional tensors
            preds.append(tf.reshape(pred_mean, (1,)))

        return tf.stack(preds)

    def lstm_loop(time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:
            cell_output = start
            next_cell_state = cell.zero_state(batch_size, tf.float64)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= time_steps)
        finished = tf.reduce_all(elements_finished)

        next_input = tf.cond(
            finished,
            lambda: tf.zeros((batch_size, d_input + 1), dtype=tf.float64),
            lambda: tf.concat([cell_output, batch_predict(cell_output)], 1)
        )

        next_loop_state = None

        return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    # these outputs will be in the shape [time_step, batch_size, d_input + 1]
    outputs, _, _ = tf.nn.raw_rnn(cell, lstm_loop)

    # Get a list of all our steps
    steps = tf.unstack(outputs.stack(), time_steps)

    # Figure out the predicted y values there
    predictions = [batch_predict(step) for step in steps]

    # Sum constraint violations across all steps
    violations = [step_constraint_violation(step) for step in steps]

    # Try to reduce the total number of deaths
    loss = tf.reduce_sum(tf.stack(predictions)) + 1000 * tf.reduce_sum(tf.stack(violations))

    opt = tf.train.AdamOptimizer().minimize(loss, var_list=cell.trainable_variables)

    print("training")

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer(), feed_dict=feeds)

    # Train
    for i in range(200):
        # Sample a bunch of random starting points
        batch_start = random_points(batch_size)

        # normalize those positions
        for b in range(batch_size):
            batch_start[b, :] = normalize_position(batch_start[b, :])

        sess.run(opt, feed_dict={start: batch_start})

        if i % 10 == 0:
            print("Loss:", sess.run(loss, feed_dict={start: batch_start}))

    # Test
    print("testing")
    test_start = random_points(1)

    # Predict all these steps using our actual function
    results = []
    state = cell.zero_state(1, dtype=tf.float64)

    def np_step_constraint_violation(step):
        l = np.maximum(model.bounds[:, 0] - step, np.zeros(step.shape))
        h = np.maximum(step - model.bounds[:, 1], np.zeros(step.shape))
        return np.sum(l + h)

    step = test_start

    for i in range(num_years):
        if np_step_constraint_violation(step) != 0:
            print("Step is outside bounds!")
        actual = predict(step)
        results.append(float(actual))
        cell_input = np.concatenate((normalize_position(step), np.reshape(actual, (1, 1))), axis=1)
        step_ta, state = cell(tf.convert_to_tensor(cell_input), state)
        step = denormalize_position(sess.run(step_ta))
        print(step)

    report('LSTM', results, time.time() - start_time, total_predict_calls)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tpe", help="run tree of parzen estimators", action="store_true")
    parser.add_argument("--bayesian", help="run bayesian optimization", action="store_true")
    parser.add_argument("--random-search", help="run random search", action="store_true")
    parser.add_argument("--gbdt", help="run gradient boosted decision trees", action="store_true")
    parser.add_argument("--lstm", help="run LSTM method", action="store_true")
    parser.add_argument("--airplane", help="use Airplane model", action="store_true")
    parser.add_argument("--repl", help="Drop into a REPL before running the models", action="store_true")
    parser.add_argument("--verbose", help="verbose logging", action="store_true")
    parser.add_argument("--iter-feature", help="use iteration number as feature", action="store_true")
    parser.add_argument("--epochs", type=int, help="number of epochs to run", default=1)

    args = parser.parse_args()

    global model
    if args.airplane:
        model = airplane.Model(args.verbose)
    else:
        model = deathrate.Model()

    run_all = not (args.tpe or args.bayesian or args.random_search or args.gbdt or args.lstm)

    if args.repl:
        import ipdb; ipdb.set_trace()

    global epoch
    for e in range(0, args.epochs):
        epoch = e

        if run_all or args.tpe:
            print("running TPE")
            tpe()

        if run_all or args.bayesian:
            print("running bayesian optimization")
            bayesian_opt(False)
            if args.iter_feature:
                bayesian_opt(True)

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
    graph_results()


if __name__ == '__main__':
    main()
