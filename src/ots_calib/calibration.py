"""
Calibration template with default implementations.

@author: wjschakel
"""
import random
import copy
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


class GeneticSearch(AbstractCalibration):
    """
    Calibrate Parameters with a simple Genetic Algorithm.

    :param pop_size:   individuals per generation
    :param generations: number of generations
    :param crossover_rate: probability of crossover
    :param mutation_rate:  probability of per-gene mutation
    :param console:   ots_calib Console for progress
    """

    def __init__(
        self,
        error_function,
        console=None,
        pop_size: int=50,
        generations: int=40,
        crossover_rate: float=0.9,
        mutation_rate: float=0.05,
        seed: int | None=None,
    ):
        super().__init__(error_function, console)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self._rng = random.Random(seed)

    # ------------------------------------------------ helpers
    def _encode(self, pars: Parameters) -> list[float]:
        """Parameter → genome (0-1 scaling)."""
        return [
            (p.get_value() - p.get_minimum()) /
            (p.get_maximum() - p.get_minimum())
            for p in pars
        ]

    def _decode(self, genome: list[float], template: Parameters) -> Parameters:
        """Genome → fresh Parameters object."""
        new_pars: Parameters = copy.deepcopy(template)
        for g, p in zip(genome, new_pars):
            lo = p.get_minimum()
            hi = p.get_maximum()
            p.set_value(lo + g * (hi - lo))
        return new_pars

    def calibrate(self, parameters: Parameters) -> float:
        """Run GA; updates *parameters* to best found & returns best error."""

        # 1  create a progress bar and get its id
        proc_id = self._console.create_process("Genetic Search")

        # -------- initial population  ----------------------------------

        gene_ids = parameters.get_ids()  # tuple of parameter names
        n_genes = len(gene_ids)  # 2 for the demo (a, b)

        population: list[list[float]] = [
            [self._rng.random() for _ in range(n_genes)]  # one genome
            for _ in range(self.pop_size)  # whole population
        ]
        best_err = float("inf")
        best_genome: list[float] | None = None

        # -------- evolutionary loop  -----------------------------------
        for gen in range(self.generations):
            fitness: list[float] = []
            for genome in population:
                pars_i = self._decode(genome, parameters)
                err = self._error_function.get_value(pars_i)
                # fitness.append(-err)
                fitness.append(1.0 / (1.0 + err))  # scales to (0 , 1]
                if err < best_err:
                    best_err, best_genome = err, genome

            # 2  update progress bar  (value between 0.0 and 1.0)
            self._console.show_progress(
                proc_id,
                progress=(gen + 1) / self.generations,
                label=f"GA gen {gen + 1}/{self.generations}",
            )

            # -------- selection / variation (unchanged) -----------------
            total_fit = sum(fitness)
            pick = lambda: population[self._roulette(fitness, total_fit)]

            new_pop: list[list[float]] = []
            while len(new_pop) < self.pop_size:
                p1, p2 = pick(), pick()
                c1, c2 = p1[:], p2[:]
                if self._rng.random() < self.crossover_rate:
                    self._sbx(c1, c2)
                self._mutate(c1)
                self._mutate(c2)
                new_pop.extend([c1, c2])
            # ---------- elitism (keep the very best genome) --------------
            elite = population[fitness.index(max(fitness))]  # best of old pop
            new_pop[0] = elite                               # overwrite slot 0
            
            population = new_pop[: self.pop_size]

        # 3  close the progress bar
        self._console.remove_process(proc_id)

        # -------- copy the best genome into *parameters* -----------------
        best_pars = self._decode(best_genome, parameters)  # fresh clone
        for p_src, p_dst in zip(best_pars, parameters):  # same order
            p_dst.set_value(p_src.get_value())  # overwrite

        return best_err

    # ---------------------------------------------------- GA operators
    def _roulette(self, fitness: list[float], total: float) -> int:
        t = self._rng.random() * total
        s = 0.0
        for idx, f in enumerate(fitness):
            s += f
            if s >= t:
                return idx
        return len(fitness) - 1

    def _sbx(self, g1: list[float], g2: list[float], eta: float=2.0):
        """Simulated-Binary Crossover (modifies genomes in place)."""
        for i in range(len(g1)):
            if self._rng.random() < 0.5:
                if abs(g1[i] - g2[i]) > 1e-12:
                    x1, x2 = sorted([g1[i], g2[i]])
                    beta = 1.0 + (2.0 * (x1) / (x2 - x1))
                    beta **= -(eta + 1)
                    c1 = 0.5 * (x1 + x2 - beta * (x2 - x1))
                    c2 = 0.5 * (x1 + x2 + beta * (x2 - x1))
                    g1[i], g2[i] = min(max(c1, 0.0), 1.0), min(max(c2, 0.0), 1.0)

    def _mutate(self, g: list[float], eta: float=5.0):
        for i in range(len(g)):
            if self._rng.random() < self.mutation_rate:
                delta = (self._rng.random() - 0.5) / (eta)
                g[i] = min(max(g[i] + delta, 0.0), 1.0)

