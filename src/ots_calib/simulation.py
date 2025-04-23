"""
A simulation can run based on parameter values to supply simulation data files.

@author: wjschakel
"""
from abc import abstractmethod, ABC

from ots_calib.error_function import Seed
from ots_calib.parameters import Parameters


class Simulation(ABC):
    """
    Simulation template.

    Attributes
    ----------
    _parameters : Parameters
        Parameters.

    _seed : Seed
        Seed object.

    _last_seed : int
       Last seed value used.

    _last_values : list[float]
       Last parameter values used.
    """

    def __init__(self, parameters: Parameters, seed: Seed):
        """
        Constructor.
        """
        self._parameters = parameters
        self._seed = seed
        self._last_seed = None
        self._last_values = None

    def get_parameters(self) -> Parameters:
        """
        Returns parameters.
        """
        return self._parameters

    def get_seed(self) -> int:
        """
        Returns the seed.
        """
        return self._seed.seed

    def run_or_skip(self):
        """
        Runs the simulation only if either one or more parameters or the seed has a different value
        from the last time the model ran.
        """
        if not self._last_seed == self._seed.seed or \
                not self._last_values == self._parameters.get_values():
            self._last_seed = self._seed.seed
            self._last_values = self._parameters.get_values()
            self.run()

    @abstractmethod
    def run(self):
        """
        Runs the simulation. Should not return until simulation data files are stored.
        """
        pass
