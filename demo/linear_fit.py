"""
Demo of calibration tool.

@author: wjschakel
"""
import random
import time

from pandas.core.frame import DataFrame

from ots_calib.calibration import GridSearch, CrossSearch, GeneticSearch
from ots_calib.console import Console
from ots_calib.data import File, SimulationFile
from ots_calib.error_function import SumOfSquares, SeededErrorFunction, Seed
from ots_calib.kpi import DataValue
from ots_calib.parameters import Parameters, Parameter
from ots_calib.simulation import Simulation

# TODO: linear car-following model to data fit
# TODO: OTS trajectory data fit

if __name__ == '__main__':

    # Parameters
    param1 = Parameter('a', 5.0, 0.0, 10.0)
    param2 = Parameter('b', 5.0, 0.0, 10.0)
    parameters = Parameters((param1, param2))

    # Seed object shared by SeededErrorFunction and TestSimulation
    seed = Seed()

    # Simulation that stores linear values with a bit of noise
    class TestSimulation(Simulation):

        def __init__(self, parameters: Parameters, seed: Seed):
            super().__init__(parameters, seed)

        def run(self):
            a = self.get_parameters().get_value('a')
            b = self.get_parameters().get_value('b')
            random_generator = random.Random(self.get_seed())
            x = []
            y = []
            for i in range(10):
                x.append(float(i + 1))
                r1 = random_generator.gauss(mu=1.0, sigma=0.01)
                r2 = random_generator.gauss(mu=1.0, sigma=0.01)
                y.append(a * r1 + b * r2 * x[i])
            df = DataFrame({'X': x, 'Y': y})
            df.to_csv('./data_files/linear_simulated_data.csv', sep=',', decimal='.', index=False)

    simulation = TestSimulation(parameters, seed)

    # Values from data files
    empirical_file = File('./data_files/linear_empirical_data.csv', pre_processing=None, sep=',')
    simulated_file = SimulationFile('./data_files/linear_simulated_data.csv', simulation, sep=',')
    data_values = []
    sim_values = []
    for i in range(10):
        data_values.append(DataValue(empirical_file, i, 'y'))
        sim_values.append(DataValue(simulated_file, i, 'Y'))

    # Error function
    sum_of_squares = SumOfSquares(sim_values, data_values)
    error_function = SeededErrorFunction(sum_of_squares, seed, 5)

    # Console
    console = Console(do_log=False, do_gui=True)

    # Calibration by GridSearch
    t0 = time.time()
    fit = GridSearch(error_function, console, steps=10).calibrate(parameters)
    t1 = time.time() - t0
    print(f'fit: {fit}')
    print(f'a={parameters.get_value("a")}, b={parameters.get_value("b")} in {t1}s')

    # Calibration by CrossSearch
    parameters.set_initial()
    t0 = time.time()
    fit = CrossSearch(error_function, console).calibrate(parameters)
    t1 = time.time() - t0
    print(f'fit: {fit}')
    print(f'a={parameters.get_value("a")}, b={parameters.get_value("b")} in {t1}s')

    # Calibration with Genetic Algorithm
    parameters.set_initial()  # reset to starting values
    ga = GeneticSearch(# you can tweak these numbers
            error_function,
            console,
            pop_size=20,
            generations=20,
            crossover_rate=0.9,
            mutation_rate=0.05,
            seed=42
    )
    t0 = time.time()
    fit = ga.calibrate(parameters)
    t1 = time.time() - t0
    print(f'GA fit: {fit}')
    print(f'a={parameters.get_value("a")}, b={parameters.get_value("b")} in {t1}s')
