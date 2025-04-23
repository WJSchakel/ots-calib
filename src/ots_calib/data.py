"""
Data objects return a DataFrame. The data can come from various sources, depending on the data
implementation.

@author: wjschakel
"""
from abc import ABC, abstractmethod
import os

from ots_calib.simulation import Simulation
import pandas as pd

# TODO: support zip file


class Data(ABC):
    """
    Data template.
    """

    def __init__(self):
        """
        Constructor.
        """

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Returns the data.
        """
        pass


class PreProcessing(ABC):
    """
    Pre-processing template.
    """

    def __init__(self):
        """
        Constructor.
        """

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the pre-processes input data.
        """
        pass


class File(Data):
    """
    Data file with columns.

    Attributes
    ----------
    _file : str
        Path and name pointing to a file.

    _pre_processing : PreProcessing
        Optional pre-processing, which may be None.

    _sep : str
        Column separator in data file.

    _reload : bool
        Whether to reload the file after its timestamp has changed.

    _modifiction_time : float
        Internal time stamp to check whether the underlying data file was changed.
    """

    def __init__(self, file: str, pre_processing: PreProcessing=None, sep: str=',',
                 reload: bool=False):
        self._file = file
        self._pre_processing = pre_processing
        self._sep = sep
        self._reload = reload
        self._modification_time = None
        if not self._reload:
            self._load()

    def get_data(self) -> pd.DataFrame:
        if self._reload:
            self._load()
        return self._data_frame

    def _load(self):
        """
        Loads the file.
        """
        if not os.path.exists(self._file):
            raise IOError(f'File {self._file} does not exist.')
        if self._modification_time != os.path.getmtime(self._file):
            if self._pre_processing:
                self._data_frame = self._pre_processing(pd.read_csv(self._file, sep=self._sep))
            else:
                self._data_frame = pd.read_csv(self._file, sep=self._sep)


class SimulationFile(File):
    """
    File from simulation.

    Attributes
    ----------
    _simulation : Simulation
        Simulation that is run when new parameter values are encountered.
    """

    def __init__(self, file: str, simulation: Simulation, pre_processing: PreProcessing=None,
                 sep: str=','):
        super().__init__(file, pre_processing, sep=sep, reload=True)
        self._simulation = simulation

    def get_data(self) -> pd.DataFrame:
        """
        Runs the simulation (if parameters or seed has changed) and returns processed (if files are
        new) output.
        """
        self._simulation.run_or_skip()
        return super().get_data()
