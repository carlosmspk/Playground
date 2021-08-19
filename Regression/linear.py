import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import warn
from matplotlib.gridspec import GridSpec
import pandas as pd

class ErrorUpdateException(Exception):
    pass

class ParameterUpdateException(Exception):
    pass

class SimpleLinearRegressor:
    def __init__(self, x, y, cost_function = None, lr=0.01, lr_decay=0.5, lr_decay_2=1, param1_range=None, param2_range=None):
        """
        This class is used to fit a set of x, y values.
        It can be run "all at once", or stepwise in which case the intermediate results can be inspected with higher detail.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        if x.shape != y.shape:
            raise Exception("x and y must have the same shape. If attempting multivariate (x with higher dimension) use MultipleLinearRegressor instead.")
        self.cost_function = cost_function or self._default_cf
        try:
            result = self.cost_function(self.y, self.y)
            assert result == 0
        except:
            raise Exception("Cost function must follow the signature: 'cost_func(iterable <float>, iterable <float>) -> float' and also return zero for two identical series. The arguments are assumed to be the predicted y and the true y values, respectively.")
        self.lr = lr
        self.lr_decay = np.clip(lr_decay, 0, 1)
        self.lr_decay_2 = np.clip(lr_decay_2, 0, 1)
        self.params = self.init_params(param1_range, param2_range)
        self.last_params = np.zeros(2)
        self.err, self.last_err = 0.0 ,0.0 
        self.err_updated = False
        self.hist = {
            'param1' : [],
            'param2' : [],
            'd_param1' : [],
            'd_param2' : [],
            'lr' : [],
            'error' : []
        }
        self.hist['lr'].append(self.lr)

    def init_params(self, param1_range : tuple = None, param2_range : tuple = None, dist_from_zero : float = None) -> np.ndarray:
        """
        Randomly initialize the parameters for given range.
        Ranges are given as a tuple of (min, max) values.
        dist_from_zero is minimum distance from zero that is assured for both parameters.
        """
        param_ranges = []
        param_ranges.append(param1_range or [-1, 1])
        param_ranges.append(param2_range or [-1, 1])
        params = []
        for i in range(2):
            value_to_append = np.random.uniform(param_ranges[i][0], param_ranges[i][1])
            if dist_from_zero:
                if abs(value_to_append) < dist_from_zero:
                    if value_to_append < 0:
                        value_to_append -= dist_from_zero
                    else:
                        value_to_append += dist_from_zero
            params.append(value_to_append)
        return np.array(params)

    def get_error(self):
        """
        gets current error if it has been updated.
        If it hasn't been updated, updates error for new parameters and returns new error.
        """
        if self.err_updated:
            return self.err

        self.last_err = self.err
        self.err = self.cost_function(self.predict(self.x), self.y)
        self.err_updated = True
        return self.err

    def get_err_derivative(self):
        """
        Returns the derivative of the error function with respect to the parameters.
        Only applies if both last_err and err have non-infinite values.
        Returns a derivative of 1 for both params if either of the error values is infinite. Else returns the two derivatives as an np.array: [d_theta1, d_theta2]
        """
        if (self.params - self.last_params == 0).any():
            raise ParameterUpdateException("Can't calculate derivative of error function with respect to parameters if the parameters are the same as the last parameters.")
        result = (self.err - self.last_err) * (self.params - self.last_params)
        return result

    def update_params(self):
        """
        updates a new step of parameters based on cost function (error).
        This method assumes error has been updated, and raises an Exception if it isn't.
        Calling this method will lead to a new error, and thus, the error will be considered as not updated after execution.
        """
        if not self.err_updated:
            raise ErrorUpdateException("Error has not been updated. Can't update parameters on the same error values.")
        self.err_updated = False
        param_derivative = self.get_err_derivative()
        self.hist['d_param1'].append(param_derivative[0])
        self.hist['d_param2'].append(param_derivative[1])
        self.last_params = np.copy(self.params)
        # note that this updates both parameters based on previous value of parameters and only then updated the parameters.
        param_deltas = np.clip(self.lr * param_derivative, -1, 1)
        self.params -= param_deltas
        
        if len(self.hist['d_param1']) > 1:
            # if any of the derivatives changed signal, decrease learning rate
            if np.sign(self.hist['d_param1'][-1]) != np.sign(self.hist['d_param1'][-2]) or np.sign(self.hist['d_param2'][-1]) != np.sign(self.hist['d_param2'][-2]):
                self.lr *= self.lr_decay
                self.lr_decay *= self.lr_decay_2
                self.hist['lr'].append(self.lr)




    def predict(self, x):
        return self.params[0] + self.params[1] * x

    def step(self):
        self.hist['error'].append(self.get_error())
        self.update_params()
        self.hist['param1'].append(self.params[0])
        self.hist['param2'].append(self.params[1])

    def run(self, steps=100):
        """
        Runs the linear regression for a given number of steps.
        if steps = 0, runs until parameters don't change.
        """
        if steps == 0:
            while True:
                try:
                    self.step()
                except ParameterUpdateException:
                    warn("Params haven't changed from last iteration, interrupting learning before assigned number of steps.")
                    break
                
        for i in tqdm(range(steps)):
            try:
                self.step()
            except ParameterUpdateException:
                warn("Params haven't changed from last iteration, interrupting learning before assigned number of steps.")
                break
            

    def plot_results(self, target_params=None):
        """
        Plots the error, and paramater values over the number of steps.
        """
        fig = plt.figure()
        gs = GridSpec(3, 2, figure=fig)
        if target_params is None:
            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, :])
            ax2 = fig.add_subplot(gs[2, :])
        else:
        
            df_params = pd.DataFrame({'Result': self.params, 'Target': target_params, 'Difference': target_params-self.params}, index=['offset', 'slope'])
            print(df_params)

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[2, 0])
            ax5 = fig.add_subplot(gs[:, 1])
            ax5.scatter(self.x, self.y, label='Original Data', c='blue')
            for i,(param1,param2) in enumerate(zip (self.hist['param1'], self.hist['param2'])):
                ax5.plot(self.x, self.x*param2 + param1, alpha=(i+1)/len(self.hist['param1']), c='orange')
            ax5.plot(self.x, self.predict(self.x), label='Predictions over time', c='orange')
            ax5.plot(self.x, self.predict(self.x), label='Final Predicted Data', c='red')
            ax5.plot(self.x, self.x*target_params[1] + target_params[0], label='Target Fit', lw=3, c='green')
            ax5.legend()
            ax5.set_title('Linear Regression Predictions over time')
        ax0.plot([i for i in range(len(self.hist['error']))], self.hist['error'])
        ax0.set_ylabel("Error")
        ax1.plot([i for i in range(len(self.hist['param1']))], self.hist['param1'])
        ax1.set_ylabel("Offset")
        ax2.plot([i for i in range(len(self.hist['param2']))], self.hist['param2'])
        ax2.set_ylabel("Slope")
        plt.tight_layout()
        plt.show()

    def reset(self, x=None, y=None, cost_function = None, lr=0.01, param1_range=None, param2_range=None):
        x = x or self.x
        y = y or self.y
        self.__init__(x, y, cost_function, lr, param1_range, param2_range)


    def _default_cf(self, y_predicted, y_true):
        return np.sum((y_predicted - y_true)**2)



class MultipleLinearRegressor:
    #TODO: implement multivariate linear regression
    def __init__(self) -> None:
        raise NotImplementedError

    
if __name__ == '__main__':
    n = 100

    target_offset, target_slope, noise = 100, -100, 10

    x = np.array([i for i in range(n)])
    y = (x + np.random.uniform(-noise, noise, n)) * target_slope + target_offset
    
    regressor = SimpleLinearRegressor(x, y, lr_decay=0.99)
    regressor.run(steps=100)
    regressor.plot_results(target_params=(target_offset, target_slope))