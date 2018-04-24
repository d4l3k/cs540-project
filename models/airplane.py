from pyfme.aircrafts import Cessna172
from pyfme.environment.atmosphere import ISA1976
from pyfme.environment.wind import NoWind
from pyfme.environment.gravity import VerticalConstant
from pyfme.environment import Environment
from pyfme.models import EulerFlatEarth
from pyfme.models.state.position import EarthPosition
from pyfme.utils.trimmer import steady_state_trim
from pyfme.utils.input_generator import Constant
from pyfme.simulator import Simulation

import matplotlib.pyplot as plt

import numpy as np

start = EarthPosition(x=0, y=0, height=10000)
target = EarthPosition(x=10000, y=10000, height=11000)
step_size = 1 # seconds

def dist(a, b):
    return np.linalg.norm(a - b.earth_coordinates)

class Model:
    def __init__(self, verbose):
        self.label = 'Distance'
        self.reset()
        self.verbose = verbose

    def predict(self, xs):
        traveled = []

        for x in xs:
            start = self.sim.system.full_state.position.earth_coordinates.copy()
            self.x.append(start[0])
            self.height.append(-start[2])

            if self.verbose:
                print(x, self.sim.system.full_state)

            for i, v in enumerate(x):
                self.controls[i].offset = x[i]

            self.time += step_size
            self.sim.propagate(self.time)

            end = self.sim.system.full_state.position.earth_coordinates.copy()

            traveled.append(dist(end, target) - dist(start, target))

        return np.array(traveled)

    def reset(self):
        self.time = 0
        self.aircraft = Cessna172()
        bounds = []
        for control, limits in self.aircraft.control_limits.items():
            bounds.append(limits)
        self.bounds = np.array(bounds)
        self.x = []
        self.height = []

        self.atmosphere = ISA1976()
        self.gravity = VerticalConstant()
        self.wind = NoWind()
        self.environment = Environment(self.atmosphere, self.gravity, self.wind)
        pos = start
        psi = 0.5  # rad
        TAS = 45  # m/s
        controls0 = {'delta_elevator': 0, 'delta_aileron': 0, 'delta_rudder': 0, 'delta_t': 0.5}
        trimmed_state, trimmed_controls = steady_state_trim(
                self.aircraft,
                self.environment,
                pos,
                psi,
                TAS,
                controls0
                )
        self.environment.update(trimmed_state)
        self.system = EulerFlatEarth(t0=0, full_state=trimmed_state)
        controls = {
            'delta_elevator': Constant(trimmed_controls['delta_elevator']),
            'delta_aileron': Constant(trimmed_controls['delta_aileron']),
            'delta_rudder': Constant(trimmed_controls['delta_rudder']),
            'delta_t': Constant(trimmed_controls['delta_t'])
        }
        self.controls = []
        for k, v in controls.items():
            self.controls.append(v)
        self.sim = Simulation(self.aircraft, self.system, self.environment, controls)

    def metadata(self):
        return (self.x, self.height)

    def plot(self, metadata):
        plt.figure()
        labels = []
        for method, meta in metadata:
            labels.append(method)
            plt.plot(meta[0], meta[1])

        plt.title('Airplane Path')
        plt.xlabel('X position')
        plt.ylabel('Height')
        plt.legend(labels)
        plt.show()

