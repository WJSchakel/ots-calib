"""
Console to display progress and state of the running algorithms.

@author: wjschakel
"""
from collections import OrderedDict
from math import ceil
import time

# TODO: plot of fit over iterations


class Console(object):
    """
    The console is what components can use to show their progress. This can either be in the form
    of labeled values, or of a labeled progress bar.

    Attributes
    ----------
    _values : dict[str, object]
        Value labels and their current value. Values can have any type and will by printed as str.

    _processes : dict[int, tuple[str, float]]
        Processes mapped by their internal id, and having both a label and progress value.

    _process_id : int
        Process id counter to increase process ids for created processes.

    _last_show : float
        System time of last time information was shown.

    _update_time : float
        Time [s] between showing information.

    _process_width : int
        Character width of progress bars indicating process progress.

    _value_width : int
        Character width of value display.

    _do_log : bool
        Whether to show logging.

    _do_gui : bool
        Whether to show GUI.
    """

    def __init__(self, update_time: float=1.0, process_width: int=50, value_width: int=30,
                 do_log: bool=True, do_gui: bool=True):
        """
        Constructor
        """
        self._values: dict[str, object] = OrderedDict()
        self._processes: dict[int, tuple[str, float]] = OrderedDict()
        self._process_id = 0
        self._last_show = 0
        self._update_time = update_time
        self._process_width = process_width
        self._value_width = value_width
        self._do_log = do_log
        self._do_gui = do_gui

    def log(self, text: str):
        """
        Prints line to console.
        """
        if self._do_log:
            print(text)

    def create_value(self, label, init_value=None):
        """
        Creates value but shows no update.
        """
        self._values[label] = init_value

    def show_value(self, label: str, value):
        """
        Show value that should ideally stay visible and that may be updated.
        """
        self._values[label] = value
        self._show_state()

    def remove_value(self, label: str):
        """
        Removes value and label.
        """
        if label in self._values:
            self._values.pop(label)

    def create_process(self, label: str) -> int:
        """
        Creates a process for which progress can be shown. Returns an identifier for the process.
        """
        process_id = self._process_id
        self._process_id += 1
        self._processes[process_id] = (label, 0.0)
        return process_id

    def show_progress(self, process_id: int, progress: float, label: str=None):
        """
        Show progress of a process.
        """
        progress = min(max(0.0, progress), 1.0)
        process = self._processes[process_id]
        if label is None:
            label = process[0]
        self._processes[process_id] = (label, progress)
        self._show_state(force=progress >= 1.0)

    def remove_process(self, process_id: int):
        """
        Removes process.
        """
        if process_id in self._processes:
            self._processes.pop(process_id)

    def _show_state(self, force: bool=False):
        """
        Shows the current state, if this is forced or was not already shown recently.
        """
        now = time.time()
        if not self._do_gui or (not force and now < self._last_show + self._update_time):
            return
        self._last_show = now

        # Copy to prevent concurrent modifications
        processes = self._processes.copy()
        values = self._values.copy()
        n_process = len(processes)
        n_value = len(values)
        it_process = iter(processes)
        it_value = iter(values)

        if n_process and n_value:
            hor_line = ('+' + '-' * (self._process_width + 9) + '+' +
                        '-' * (self._value_width + 2) + '+')
            sep = ' | '
        elif n_process:
            hor_line = '+' + '-' * (self._process_width + 9) + '+'
            sep = ''
        elif n_value:
            hor_line = '+' + '-' * (self._value_width + 2) + '+'
            sep = ''
        else:
            return  # no processes, no values, nothing to show

        print(hor_line)
        for i in range(max(n_process, n_value)):
            # Progress bar
            if i < n_process:
                process = processes[next(it_process)]
                process_str1 = process[0].ljust(self._process_width + 7, ' ')
                n_solid = ceil(process[1] * self._process_width)
                bar = ('■' * n_solid) + ('□' * (self._process_width - n_solid))
                p = 100 * process[1]
                process_str2 = f'{bar}{p: 6.1f}%'
            elif n_process:
                process_str1 = ' ' * (self._process_width + 7)
                process_str2 = process_str1
            else:
                process_str1 = ''
                process_str2 = ''
            # Value
            if i < n_value:
                label = next(it_value)
                value_str1 = label + ':'
                value_str1 = value_str1.ljust(self._value_width, ' ')
                value_str2 = ' ' + str(self._values[label])
                value_str2 = value_str2.ljust(self._value_width, ' ')
            elif n_value:
                value_str1 = ' ' * self._value_width
                value_str2 = value_str1
            else:
                value_str1 = ''
                value_str2 = ''
            # Print label lines and value lines
            print('| ' + process_str1 + sep + value_str1 + ' |')
            print('| ' + process_str2 + sep + value_str2 + ' |')
        print(hor_line)
