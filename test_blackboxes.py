#!/usr/bin/env python3
print("loading")

import argparse
import time
import math

import bayes_opt
import catboost
import hyperopt
import numpy as np
import scipy.linalg
import scipy.optimize
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from sparklines import sparklines
from tabulate import tabulate
from tensorflow.contrib import rnn

from models import deathrate
from models import airplane

init_years = 5
sample_points = 100
epoch = 0

results = []


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def report(*args):
    global results
    args = args + (epoch, model.metadata())
    results.append(args)
    print_result(*args)
    with open("results.txt", "a") as myfile:
        myfile.write(json.dumps(args, cls=NumpyEncoder))
        myfile.write('\n')


def format_result(method, data, time_taken, predict_calls, epoch, metadata):
    return [method, np.sum(data), data[len(data) - 1], len(data),
            '\n'.join(sparklines(data)), time_taken, predict_calls, epoch]


def print_result_table(table):
    print(tabulate(table, headers=['Method', model.label, 'Final', 'Years',
                                   'Distribution', 'Time (s)', 'Calls to Predict', 'Epoch']))


def print_result(*args):
    print_result_table([format_result(*args)])


def print_results():
    print_result_table([format_result(*res) for res in results])

def merge(vals):
    if isinstance(vals[0], (list, tuple, np.ndarray)):
        out = []

        columns = len(vals[0])
        for i in range(columns):
            intermediates = []
            for row in vals:
                intermediates.append(row[i])

            out.append(merge(intermediates))

        return out

    elif isinstance(vals[0], str):
        return vals[0]

    elif isinstance(vals[0], (int, float, complex)):
        return np.mean(vals)

    else:
        raise Exception('unknown type for {}'.format(vals))


def average_results():
    global results

    print('averaging...')

    grouped = {}
    for result in results:
        method = result[0]
        grouped[method] = grouped.get(method, []) + [result]

    merged = []
    for key, results in grouped.items():
        merged.append(merge(results))

    results = merged

def graph_results():
    labels = []

    metadata = []

    plt.figure()

    for result in results:
        method = result[0]
        data = result[1]
        epoch = result[4]

        if epoch > 0 and not args.mean:
            method += ' (Epoch {})'.format(epoch)

        labels.append(method)
        plt.plot(range(0,len(data)), data)

        metadata.append((method, result[5]))

    plt.legend(labels)
    plt.title('Algorithm Performance (Smaller is better)')
    plt.xlabel('Iteration')
    plt.ylabel(model.label)
    plt.show()

    model.plot(metadata)



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

def random_points(n, iter_feature=False):
    data = random_state.uniform(model.bounds[:, 0], model.bounds[:, 1], size=(n, model.bounds.shape[0]))
    if iter_feature:
        data = np.hstack((data, np.ones((n, 1)) * total_predict_calls))
    return data


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


def gbdt(iter_feature):
    """GBDT"""
    clear_predict()
    start = time.time()

    def apply_iter(x):
        if iter_feature:
            x[len(x)-1] = total_predict_calls
        return x

    init_points = 2
    x = random_points(init_points, iter_feature)
    y = np.zeros(init_points)

    num_features = len(model.bounds)

    # generate each initial year separately
    for i in range(init_points):
        y[i] = predict(np.reshape(x[i, :num_features], (1, num_features)))

    gbdt_model = catboost.CatBoostRegressor(verbose=False)
    for i in range(init_years-init_points+num_years):
        gbdt_model.fit(x, y)
        x_tmps = []
        for test_x in random_points(sample_points, iter_feature):
            bounds = model.bounds
            if iter_feature:
                bounds = np.vstack((bounds, [[total_predict_calls, total_predict_calls]]))
            res = scipy.optimize.minimize(lambda p_x:
                    gbdt_model.predict([apply_iter(p_x)])[0], test_x,
                    bounds=bounds)
            if not res.success:
                continue
            x_tmps.append(apply_iter(res.x))

        best_i = int(np.argmin(gbdt_model.predict(x_tmps)))
        best_x = x_tmps[best_i]
        x = np.append(x, [best_x], axis=0)
        y_tmp = predict([best_x[:num_features]])
        y = np.append(y, y_tmp)

    method = 'GBDT'
    if iter_feature:
        method += ' (Iteration Feature)'

    report(method, y[init_years:], time.time() - start, total_predict_calls)


class LSTMModel:
    """LSTM RNN Model"""
    BATCH_SIZE = 1
    TRAINING_ITERATIONS = 200

    def __init__(self):
        # Some constants depend on model/flags
        self.d_input = model.bounds.shape[0]
        self.num_units = self.d_input
        self.time_steps = num_years
        self.init_samples = init_years

        tf.reset_default_graph()
        gpflow.reset_default_session()

        self.g_functions = []
        self.samples = []
        self.cell = rnn.BasicLSTMCell(self.num_units, forget_bias=1)
        self.gp_opt = gpflow.train.ScipyOptimizer()
        self.opt = tf.train.AdamOptimizer()
        self.start = tf.placeholder(tf.float64, [self.BATCH_SIZE, self.d_input])
        self.feeds = {}
        self.sess = tf.Session()
        self.position_bias = model.bounds[:, 0]
        self.position_weight = model.bounds[:, 1] - model.bounds[:, 0]

    def _normalize_position(self, x):
        return 2. * np.divide((x - self.position_bias), self.position_weight) - 1.

    def _denormalize_position(self, x):
        return np.multiply((x + 1.) / 2., self.position_weight) + self.position_bias

    @staticmethod
    def _step_constraint_violation(step):
        return tf.reduce_sum(tf.maximum(1. - tf.abs(step), tf.ones(step.shape, tf.float64)) - 1.)

    @staticmethod
    def _output_step_constraint_violation(step):
        l = np.maximum(model.bounds[:, 0] - step, np.zeros(step.shape))
        h = np.maximum(step - model.bounds[:, 1], np.zeros(step.shape))

        return np.sum(l + h)

    def _batch_predict(self, position):
        preds = []

        for i in range(self.BATCH_SIZE):
            this_position = tf.reshape(position[i], (1, position[i].shape[0]))

            pred_mean, pred_var = self.g_functions[i]._build_predict(this_position)
            pred_mean, _ = self.g_functions[i].likelihood.predict_mean_and_var(pred_mean, pred_var)

            # We know the dimensionality here, but GPFlow loses track of it
            # As well, turn these into one-dimensional tensors
            preds.append(tf.reshape(pred_mean, (1,)))

        return tf.stack(preds)

    def _lstm_loop(self, time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:
            cell_output = self.start
            next_cell_state = self.cell.zero_state(self.BATCH_SIZE, tf.float64)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= self.time_steps)
        finished = tf.reduce_all(elements_finished)

        next_input = tf.cond(
            finished,
            lambda: tf.zeros((self.BATCH_SIZE, self.d_input + 1), dtype=tf.float64),
            lambda: tf.concat([cell_output, self._batch_predict(cell_output)], 1)
        )

        next_loop_state = None

        return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

    def _build_one_gp(self, x_sample, y_sample):
        x = np.zeros(x_sample.shape)
        y = y_sample

        # Generate our random points
        for i in range(self.init_samples):
            # normalize the position
            x[i, :] = self._normalize_position(x_sample[i, :])

        with gpflow.defer_build():
            # Initialize our model
            kern = gpflow.kernels.Matern32(x.shape[1]) + gpflow.kernels.Linear(x.shape[1])
            f = gpflow.models.GPR(x, y, kern=kern)

        # Compile it
        f.compile(session=self.sess)

        # Include this function's feeds into ours
        self.feeds.update(f.initializable_feeds)

        self.gp_opt.minimize(f)

        return f

    def _build_rnn(self):
        # these outputs will be in the shape [time_step, batch_size, d_input + 1]
        outputs, _, _ = tf.nn.raw_rnn(self.cell, self._lstm_loop)

        # Get a list of all our steps
        steps = tf.unstack(outputs.stack(), self.time_steps)

        # Figure out the predicted y values there
        predictions = [self._batch_predict(step) for step in steps]

        # Sum constraint violations across all steps
        violations = [self._step_constraint_violation(step) for step in steps]

        # Try to reduce the total number of deaths
        loss = tf.reduce_sum(tf.stack(predictions)) + 1000 * tf.reduce_sum(tf.stack(violations))

        return loss, steps

    def _train(self, loss):
        train_loss = self.opt.minimize(loss, var_list=self.cell.trainable_variables)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer(), feed_dict=self.feeds)

        # Train
        for i in range(self.TRAINING_ITERATIONS):
            # Sample a bunch of random starting points
            batch_start = random_points(self.BATCH_SIZE)

            # normalize those positions
            for b in range(self.BATCH_SIZE):
                batch_start[b, :] = self._normalize_position(batch_start[b, :])

            self.sess.run(train_loss, feed_dict={self.start: batch_start})

            if i % 10 == 0:
                print("Loss:", self.sess.run(loss, feed_dict={self.start: batch_start}))

    def _sample_batches(self):
        batch_samples = int(math.ceil(len(self.samples) / self.BATCH_SIZE))

        for i in range(0, len(self.samples), batch_samples):
            yield self.samples[i:i + batch_samples]

    def _build_gps(self):
        # Clear whatever's already there
        self.g_functions.clear()

        for batch in self._sample_batches():
            # list of tuples -> tuple of lists
            x_sample, y_sample = zip(*batch)

            # Create a gaussian process
            f = self._build_one_gp(np.stack(x_sample), np.stack(y_sample))

            # Save it
            self.g_functions.append(f)

    def evaluate(self):
        clear_predict()
        start_time = time.time()

        print("sampling GPs")

        # Create our initial samples
        for i in range(self.init_samples):
            point = random_points(1)
            self.samples.append((np.reshape(point, (model.bounds.shape[0],)), predict(point)))

        # Create our gaussians
        self._build_gps()

        print("creating LSTM")

        loss, steps = self._build_rnn()

        print("training")

        self._train(loss)

        # Test
        print("testing")

        # Start at a minimum found by the network

        test_start = random_points(self.BATCH_SIZE)

        test_steps = self.sess.run(steps, feed_dict={self.start: test_start})

        # Predict all these steps using our actual function
        test_actuals = []
        state = self.cell.zero_state(1, dtype=tf.float64)

        step = np.stack([self._denormalize_position(test_steps[-1][0, :])])

        for i in range(num_years):
            if self._output_step_constraint_violation(step) != 0:
                print("Step is outside bounds!")

            print(step)
            actual = predict(step)
            test_actuals.append(float(actual))

            cell_input = np.concatenate((self._normalize_position(step), np.reshape(actual, (1, 1))), axis=1)

            step_ta, state = self.cell(tf.convert_to_tensor(cell_input), state)
            step = self._denormalize_position(self.sess.run(step_ta))

        report('LSTM', test_actuals, time.time() - start_time, total_predict_calls)


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
    parser.add_argument("--mean", help="take the mean of the values", action="store_true")
    parser.add_argument("--num-years", type=int, help="number of years to run for", default=20)
    parser.add_argument("--load", type=str, help="load results file")

    global args
    args = parser.parse_args()

    global num_years
    num_years = args.num_years

    global model
    if args.airplane:
        model = airplane.Model(args.verbose)
    else:
        model = deathrate.Model()

    if args.load is not None:
        print('Loading file')
        with open(args.load) as f:
            for line in f.readlines():
                results.append(json.loads(line.strip()))

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
            gbdt(False)
            if args.iter_feature:
                gbdt(True)

        if run_all or args.lstm:
            print("running lstm")
            LSTMModel().evaluate()

    print_results()

    if args.mean:
        average_results()
        print_results()

    graph_results()


if __name__ == '__main__':
    main()
