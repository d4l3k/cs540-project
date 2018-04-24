import catboost
import numpy as np

momentum = .10

class Model:
    def __init__(self):
        self.model = catboost.CatBoostRegressor(verbose=False)
        self.model.load_model('deathrate.cbm')
        self.bounds = np.array([(0, 14.06), (2.23, 17.14), (0.02, 4.41), (0.13, 14.81)])
        self.reset()
        self.label = 'Dead per 1000'

    def predict(self, x):
        y = self.model.predict(x)

        if self.last_y is None:
            self.last_y = y
        else:
            # Add some momentum year-on-year
            y += momentum * self.last_y
            self.last_y = y

        return y

    def reset(self):
        self.last_y = None

    def metadata(self):
        return []

    def plot(self, meta):
        pass
