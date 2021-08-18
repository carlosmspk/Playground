import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegressor:
    def __init__(self, x, y):
        """
        This class is used to fit a set of x, y values.
        It can be run "all at once", or stepwise in which case the intermediate results can be inspected with higher detail.
        """
        self.x = x
        self.y = y
        self.params = self.init_params()
        self.err = np.inf
        self.err_updated = False
        self.hist = {
            'param1' : [],
            'param2' : [],
            'error' : []
        }

    def init_params(self, param1_range=None, param2_range=None):
        """
        Randomly initialize the parameters to some reasonable values for given range.
        """
        param1_range = param1_range or [0, 1]
        param2_range = param2_range or [0, 1]
        return np.random.uniform(param1_range[0], param1_range[1], 1), np.random.uniform(param2_range[0], param2_range[1], 1)

    def get_error(self):
        """
        gets current error if it has been updated.
        If it hasn't been updated, updates error for new parameters and returns new error.
        """
        if self.err_updated:
            return self.err

        self.err = np.sum((self.y - self.predict(self.x))**2)
        self.hist['error'].append(self.err)
        self.err_updated = True
        return self.err

    def update_params(self):
        """
        updates a new step of parameters based on cost function (error)
        """
        #TODO: implement gradient descent

    def predict(self, x):
        return self.params[0] + self.params[1] * x

    def run(self, steps=1):
        """
        Runs the linear regression for a given number of steps.
        """
        #TODO: implement full algorithm

    
if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = (x + np.random.uniform(-1, 1, 10)) * 2
    plt.plot(x, y, 'o')
    plt.show()
