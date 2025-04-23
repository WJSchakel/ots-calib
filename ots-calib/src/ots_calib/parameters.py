"""
Parameters is a set of parameter objects, with parameter value and their bounds.

@author: wjschakel
"""
from abc import ABC, abstractmethod
from builtins import StopIteration
from collections.abc import Iterable


class Parameters(Iterable):
    """
    Attributes
    ----------
    _parameters : dict[str, Parameter]

    _id_list: list[str]

    _last_values : dict[str, float]
    """

    def __init__(self, parameters: Iterable):
        """
        Constructor which attaches itself to the parameters.
        """
        self._parameters = dict()
        self._id_list = []
        for param in parameters:
            param._parameters = self
            self._parameters[param.get_id()] = param
            self._id_list.append(param.get_id())
        self._last_values = dict()

    def get_value(self, parameter_id: str) -> float:
        """
        Returns the current value of the parameter with given parameter id.
        """
        return self._parameters[parameter_id].get_value()

    def set_value(self, parameter_id: str, value: float):
        """
        Sets the parameter and remembers the previous value for resettings.
        """
        self._last_values[parameter_id] = self.get_value(parameter_id)
        self._parameters[parameter_id].set_value(value)

    def reset_value(self, parameter_id: str):
        """
        Reset value of parameter to previous value.
        """
        if parameter_id in self._last_values:
            self._parameters[parameter_id].set_value(self._last_values[parameter_id])
            self._last_values.pop(parameter_id)

    def set_initial(self):
        """
        Set all initial parameter values.
        """
        for parameter in self._parameters.values():
            parameter.set_value(parameter.get_initial_value())

    def get_ids(self) -> list[str]:
        """
        Returns list with values.
        """
        return self._id_list.copy()

    def get_values(self) -> list[float]:
        """
        Returns list with values.
        """
        values = []
        for parameter_id in self._id_list:
            values.append(self.get_value(parameter_id))
        return values

    def get_parameter(self, parameter_id: str):
        """
        Returns parameter.
        """
        return self._parameters[parameter_id]

    def __iter__(self):
        """
        Returns iterator over parameters.
        """
        self._it_index = 0
        return self

    def __next__(self):
        """
        Returns the next parameter.
        """
        if self._it_index >= len(self._id_list):
            raise StopIteration
        parameter = self.get_parameter(self._id_list[self._it_index])
        self._it_index += 1
        return parameter

    def __str__(self):
        """
        """
        result = 'Parameters ['
        sep = ''
        for parameter in self._parameters.values():
            result += (sep + f'{parameter.get_id()}: {parameter.get_value():0.2f}')  # noqa
            sep = ', '
        return result + ']'


class Value(ABC):
    """
    Defines any object that can return a value dependent on parameters.
    """

    def __init__(self):
        """
        Constructor.
        """

    @abstractmethod
    def get_value(self, parameters: Parameters):
        """
        Returns a value which may depend on current parameter values.
        """
        raise NotImplementedError('Method get_value not implemented by sub-class.')


class Bound(Value):
    """
    Attributes
    ----------
    _bound_function : Callable[Parameters]
        Function that returns the value of this bound. This is either a fixed value (float) or the
        current value of another parameters in Parameters (str).
    """

    def __init__(self, bound: float | str):
        """
        Constructor
        """
        if isinstance(bound, float):
            self._bound_function = lambda _parameters: bound
        elif isinstance(bound, str):
            self._bound_function = lambda parameters: parameters.get_value(bound)
        else:
            raise TypeError(f'input bound ({bound}) is not float or str')

    def get_value(self, parameters: Parameters) -> float:
        """
        Returns the value of the bound, which is either a fixed value, or the value of another
        parameter.
        """
        return self._bound_function(parameters)


class Parameter:
    """
    Attributes
    ----------
    _id : str
        Parameter id.

    _initial_value : float
        Initial value for parameter.

    _min : Value
        Minimum parameter value.

    _max : Value
        Maximum parameter value.

    _value : float
        Parameter value.

    _parameters : Parameters
        Parameters to obtain dynamic bound value for. Set by Parameters.
    """

    def __init__(self, param_id: str, initial_value: float, min_val: [Value, float, str],
                 max_val: [Value, float, str]):
        """
        Constructor.
        """
        self._id = param_id
        self._initial_value = initial_value
        self._min = min_val if isinstance(min_val, Value) else Bound(min_val)
        self._max = max_val if isinstance(max_val, Value) else Bound(max_val)
        self._value = initial_value

    def get_id(self) -> str:
        """
        Returns the parameter id.
        """
        return self._id

    def get_initial_value(self) -> float:
        """
        Returns the initial value.
        """
        return self._initial_value

    def get_minimum(self) -> float:
        """
        Returns the minimum value.
        """
        return self._min.get_value(self._parameters)

    def get_maximum(self) -> float:
        """
        Returns the maximum value.
        """
        return self._max.get_value(self._parameters)

    def get_value(self) -> float:
        """
        Returns the current value.
        """
        return self._value

    def set_value(self, value: float):
        """
        Sets the current value.
        """
        if value < self.get_minimum() or value > self.get_maximum():
            raise ValueError(f'Value {value} for parameter {self.get_id()} is not in bounds.')
        self._value = value
