"""
Error functions describe mathematical procedures to translate KPI's in to a single-valued error
measure for minimization.

Note: since error functions extend Value, and are defined based on Value input, such error
functions can be used to combine other error functions.

@author: wjschakel
"""
from math import sqrt

from ots_calib.parameters import Parameters, Value

# TODO: implement Bayesian error functions


class Seed(object):
    """
    Simple container for a seed value.

    Attributes
    ----------
    seed : int
        The seed value.
    """

    def __init__(self):
        self.seed = 0


class SeededErrorFunction(Value):
    """
    Error function that runs multiple seeds for the same error function and returns the mean error.

    Attributes
    ----------
    _error_function : Value
        Error function that depends on parameters and a seed value.

    _seed : Seed
        Seed object in to which the seed value will be set, which should be shared with the error
        function.

    _seeds : list[int]
        List of seed values.
    """

    def __init__(self, error_function: Value, seed: Seed, seeds: int | list[int]):
        """
        Constructor.
        """
        self._error_function = error_function
        self._seed = seed
        self._seeds = seeds if isinstance(seeds, list) else list(range(seeds))

    def get_value(self, parameters: Parameters) -> float:
        """
        Returns the average error over all seeds.
        """
        sum_error = 0
        for seed in self._seeds:
            self._seed.seed = seed
            sum_error += self._error_function.get_value(parameters)
        return sum_error / len(self._seeds)


class SumOfSquares(Value):
    """
    Error function that returns the sum of squares between pairwise values.

    Attributes
    ----------
    _sim_values : list[Value]
        Value functions of simulated data.

    _data_values : list[Value]
        Value functions of data.
    """

    def __init__(self, sim_values: list[Value], data_values: list[Value]):
        """
        Constructor.
        """
        if len(sim_values) != len(data_values):
            raise ValueError('Kpis for simulation and unfiltered_data are not of equal length.')
        self._sim_values = sim_values.copy()
        self._data_values = data_values.copy()

    def get_value(self, parameters: Parameters) -> float:
        """
        Returns the sum of squares.
        """
        error_value = 0.0
        for i in range(len(self._sim_values)):
            sim_value = self._sim_values[i].get_value(parameters)
            data_value = self._data_values[i].get_value(parameters)
            delta_value = sim_value - data_value
            error_value += (delta_value ** 2)
        return error_value


class RootMeanSquareError(SumOfSquares):
    """
    Error function that returns the root mean square error (RMSE) between pairwise values.

    Attributes
    ----------
    _sim_values : list[Value]
        Value functions of simulated data.

    _data_values : list[Value]
        Value functions of data.
    """

    def __init__(self, sim_values: list[Value], data_values: list[Value]):
        """
        Constructor.
        """
        super().__init__(sim_values, data_values)

    def get_value(self, parameters: Parameters) -> float:
        """
        Returns the root mean squared error.
        """
        return sqrt(super().get_value(parameters) / len(self._sim_values))


class WeightedSum(Value):
    """
    Error function that returns a weighted sum.

    Attributes
    ----------
    _value_functions : dict[Value, float]
        Value functions and their weight.

    _sum_weights : float
        Sum of the value weights.
    """

    def __init__(self, value_functions: dict[Value, float]):
        """
        Constructor.
        """
        self._value_functions = value_functions.copy()
        self._sum_weights = 0.0
        for weight in value_functions.values():
            if weight < 0.0:
                raise ValueError(f'Weight {weight: %.4f} must be above or equal to 0.')
            self._sum_weights += weight
        if self._sum_weights <= 0.0:
                raise ValueError('Sum of weights must be above 0.')

    def get_value(self, parameters: Parameters) -> float:
        """
        Returns weighted sum.
        """
        error_value = 0.0
        for value_function, weight in self._value_functions.items():
            error_value += (value_function.get_value(parameters) * weight)
        return error_value / self._sum_weights
