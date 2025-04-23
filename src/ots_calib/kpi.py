"""
Key Performance Indicators.

@author: wjschakel
"""
from _collections_abc import Callable

from ots_calib.data import Data
from ots_calib.parameters import Parameters, Value


class Kpi(Value):
    """
    Defines any object that can return a value dependent on parameters.
    """

    def __init__(self):
        """
        Constructor.
        """


class DataValue(Kpi):
    """
    Returns a single value from data.

    Attributes
    ----------
    _data : Data
        Data provider.

    _index : int
        Index in column of value to return.

    _column : str
        Column name in data.
    """

    def __init__(self, data: Data, index: int, column: str):
        """
        Constructor.
        """
        self._data = data
        self._index = index
        self._column = column

    def get_value(self, _parameters: Parameters) -> float:
        return float(self._data.get_data().loc[self._index, self._column])


class ColumnStatistic(Kpi):
    """
    Returns statistic of a column of data.

    Attributes
    ----------
    _data : Data
        Data provider.

    _column : str
        Column name in data.

    _fun : Callable
        Function to call on column. For example. Series.mean, Series.median or Series.max.
    """

    def __init__(self, data: Data, column: str, fun: Callable):
        """
        Constructor.
        """
        self._data = data
        self._column = column
        self._fun = fun

    def get_value(self, _parameters: Parameters) -> float:
        return float(self._fun(self._data.get_data().loc[:, self._column]))
