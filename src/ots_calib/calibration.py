"""
Calibration template with default implementations.

@author: wjschakel
"""
from abc import ABC, abstractmethod

from ots_calib.console import Console
from ots_calib.parameters import Parameters, Value, Parameter

# TODO: include SciPy support, see: https://docs.scipy.org/doc/scipy/reference/optimize.html


class Calibration(ABC):
    """
    Attributes
    ----------
    _error_function : Value
        Error function.

    _console : Console
        Console to show progress.
    """

    def __init__(self, error_function: Value, console: Console):
        """
        Constructor.
        """
        self._error_function = error_function
        self._console = console

    @abstractmethod
    def calibrate(self, parameters: Parameters) -> list[float]:
        """
        Performs calibration. Returns the fit over iterations. Optimized parameter values are
        set to the input parameters.
        """
        pass

    def get_console(self) -> Console:
        """
        Returns the console.
        """
        return self._console


class AbstractCalibration(Calibration):
    """
    Attributes
    ----------
    _covered : dict[tuple, float]
        Covered parameter sets and their error value.
    """

    def __init__(self, error_function: Value, console: Console):
        """
        Constructor.
        """
        super().__init__(error_function, console)
        self._covered: dict[list, float] = dict()

    def get_error_value(self, parameters: Parameters) -> float:
        """
        Returns the error value for the given parameters. This method remembers parameter sets for
        which the error is determined, and will return the stored value rather than recalculating
        the error if the same parameter values are given.
        """
        parameter_values = tuple(parameters.get_values())
        if parameter_values in self._covered:
            error_value = self._covered[parameter_values]
        else:
            error_value = self._error_function.get_value(parameters)
            self._covered[parameter_values] = error_value
        return error_value


class CrossSearch(AbstractCalibration):
    """
    Simple optimization based on Schakel et al. (2012). In each iteration, each parameter is
    checked at a factor smaller and larger. The parameter change with lowest error is applied and
    the algorithm runs again. If no better solution is found, the search space is reduced by
    lowering the factor. If the factor is sufficiently small, calibration stops. Optimized
    parameters will never change sign relative to the initial value.

    Schakel, W.J., Knoop, V.L., and Van Arem, B. (2012), LMRS: Integrated Lane Change Model
    with Relaxation and Synchronization, Transportation Research Records: Journal of the
    Transportation Research Board, No. 2316, pp. 47-57.

    Attributes
    ----------
    _factor : float
        Current factor applied on parameters in search space.

    _shrink : float
        Shrink factor on the search factor when no better solution was found.

    _stop_criterion : float
        Stop criterion. The algorithm stops when the factor has become smaller than this.
    """

    def __init__(self, error_function: Value, console: Console, init_factor: float=1.25,
                 shrink: float=(2.0 / 3.0), stop_criterion: float=1.001):
        """
        Constructor.
        """
        super().__init__(error_function, console)
        if init_factor <= 1.0:
            raise ValueError(f'Initial factor {init_factor} needs to be above 1.0.')
        if shrink >= 1.0:
            raise ValueError(f'Shrink {shrink} needs to be below 1.0.')
        if stop_criterion <= 1.0:
            raise ValueError(f'Stop criterion {stop_criterion} needs to be above 1.0.')
        self._factor = init_factor
        self._shrink = shrink
        self._stop_criterion = stop_criterion

    def calibrate(self, parameters: Parameters) -> list[float]:
        """
        Performs calibration. Returns the fit over iterations. Optimized parameter values are
        set in to the input parameters.
        """
        best_fit = self.get_error_value(parameters)
        fit = [best_fit]
        while self._factor > self._stop_criterion:
            found_better = True
            while found_better:
                found_better, better_fit, better_parameter_id, better_parameter_value = \
                    self._cross_step(parameters, best_fit)
                if found_better:
                    self.get_console().log(
                        f'Set parameter {better_parameter_id} to {better_parameter_value}.')
                    parameters.set_value(better_parameter_id, better_parameter_value)
                    best_fit = better_fit
                    fit.append(better_fit)
            # Shrink
            self._factor = 1.0 + self._shrink * (self._factor - 1.0)
            self.get_console().log(f'Shrunk to factor {self._factor}.')
            self.get_console().show_value('Factor', self._factor)
        self.get_console().log('Finished optimization.')
        self.get_console().remove_value('Factor')
        return fit

    def _cross_step(self, parameters: Parameters, best_fit: float) -> tuple:
        """
        Performs a single iteration, applying a cross search by looking at a smaller and larger
        value for each parameter.
        """
        found_better = False
        better_fit = best_fit
        better_parameter_id = None
        better_parameter_value = None
        for parameter_id in parameters.get_ids():
            parameter: Parameter = parameters.get_parameter(parameter_id)
            base_value = parameters.get_value(parameter_id)
            for f in (1.0 / self._factor, self._factor):
                new_value = f * base_value
                if (parameter.get_minimum() <= new_value and
                        new_value <= parameter.get_maximum()):
                    parameters.set_value(parameter_id, new_value)
                    error_value = self.get_error_value(parameters)
                    if error_value < better_fit:
                        found_better = True
                        better_fit = error_value
                        better_parameter_id = parameter_id
                        better_parameter_value = new_value
                    parameters.reset_value(parameter_id)
        return (found_better, better_fit, better_parameter_id, better_parameter_value)


class GridSearch(AbstractCalibration):
    """
    Simple grid search.

    Attributes
    ----------
    _steps : int | dict[str, int]
        Steps (per parameter).
    """

    def __init__(self, error_function: Value, console: Console, steps: int | dict[str, int]=10):
        """
        Constructor.
        """
        super().__init__(error_function, console)
        self._steps = steps

    def calibrate(self, parameters: Parameters) -> list[float]:
        """
        Performs calibration. Returns the fit over iterations, which is a single value in grid
        search. Optimized parameter values are set in to the input parameters. Initial values may
        serve as bounds on other parameters in the grid search, if parameters are relative.
        """
        process_id = self.get_console().create_process('Grid search')
        self.get_console().create_value('Runs', 0)
        min_vals = {parameter.get_id(): parameter.get_minimum() for parameter in parameters}
        max_vals = {parameter.get_id(): parameter.get_maximum() for parameter in parameters}
        best_fit = float('inf')
        parameter_ids = parameters.get_ids()
        parameter_index = 0
        current_steps = [0] * len(parameter_ids)
        total_runs = 1
        run = 0
        # Set lower bound values and count runs
        for parameter_id in parameters.get_ids():
            parameters.set_value(parameter_id, min_vals[parameter_id])
            total_runs *= (self._get_steps(parameter_id) + 1)
        while True:
            run += 1
            self.get_console().show_value('Runs', run)
            self.get_console().show_progress(process_id, run / total_runs)
            # Set current parameter step and run
            parameter_id = parameter_ids[parameter_index]
            n = self._get_steps(parameter_id)
            step_size = (max_vals[parameter_id] - min_vals[parameter_id]) / n
            new_value = min_vals[parameter_id] + current_steps[parameter_index] * step_size
            parameters.set_value(parameter_id, new_value)
            error_value = self.get_error_value(parameters)
            self.get_console().log(f'{parameters} gave error {error_value}')
            if error_value < best_fit:
                best_fit = error_value
                best_parameter_values = parameters.get_values()
            # Increase first parameter that allows it, set earlier to their minimum value
            parameter_index = 0
            while parameter_index < len(current_steps) and current_steps[parameter_index] == \
                    self._get_steps(parameter_ids[parameter_index]):
                current_steps[parameter_index] = 0
                parameters.set_value(parameter_id, min_vals[parameter_id])
                parameter_index += 1
            if parameter_index < len(current_steps):
                current_steps[parameter_index] += 1
            else:
                # No parameter allows an increase, we're done
                for i in range(len(parameter_ids)):
                    parameters.set_value(parameter_ids[i], best_parameter_values[i])
                self.get_console().remove_process(process_id)
                self.get_console().remove_value('Runs')
                return [best_fit]

    def _get_steps(self, parameter_id: str) -> int:
        """
        Returns the number of steps for the parameter.
        """
        if isinstance(self._steps, int):
            return self._steps
        return self._steps[parameter_id]
    
class Genetic(AbstractCalibration):
    
    
    
    def __init__(self, error_function: Value, console: Console, population_size: int=30):
        """
        Constructor.
        """
        super().__init__(error_function, console)
        self._population_size = population_size
