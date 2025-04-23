Introduction
===============
Welcome to **ots-calib**, a calibration tool with both general optimization features, and features
specific to data processing in the context of `OpenTrafficSim <https://opentrafficsim.org/>`_
(OTS).

This codebase is a PyDev project developed in Eclipse. To use it, please install Eclipse with
PyDev, and import the git repository. The project uses standard PEP8 formatting with lines of 100
characters.

Structure
===============
The overall structure of the project code is as follows:

- Module **parameters** includes ``Parameters`` which holds one or more ``Parameter`` objects. Each
  parameters is defined with an initial value and applicable range. Each parameter holds a value 
  that is optimized.

- Module **calibration** holds various optimization algorithms, which optionally derive from
  ``Calibration`` (link to ``error_function`` and ``console``) or ``AbstractCalibration``
  (skips evaluation of same parameters).

- Module **console** holds a simple console that prints on the environment console. It can show
  status values and progress of processes, depending on what other components are requesting to be
  shown.

- Module **error_function** contains generic functions based on underlying empirical and simulated
  values. It also contains ``SeededErrorFunction`` using a ``Seed`` object and underlying error
  function.

- Module **kpi** contains kpi's that transform data into values that feed the error functions.
  The simplest type is ``DataValue``, which takes a single value from data.

- Module **data** contains data classes that provide data in the form of a pandas ``DataFrame``.
  This data can come from a file, which may or may not be pre-processed, and may or may not come 
  from an underlying simulation.

- Module **filter** contains methods to filter data. Similarly to ``data``, these classes provide a
  ``DataFrame``, and can hence be used as the basis for kpi's. Of particular interest is
  ``SpaceTimeRegion`` which contains methods to provide theoretically sensible traffic statistics
  over a space-time region.

- Module **path** contains a combined pre-processing and filtering to supply data relevant to a
  path, defined by a set of lanes in simulation. This too can be directly used as data supplying a
  ``DataFrame``.

- Module **simulation** contains a class that combines a seed and parameters, and invokes a
  simulation run when new simulation files are required. Implementations of this class will be
  use-case specific.

Both ``error_function``'s and ``kpi``'s extend ``Value``, which ``Calibration`` expects as error
function. Any supporting structure of error functions and kpi's can be used to feed the error value
to the calibration algorithm.

Demos
===============
Demo's can be found in the demo folder.

- **linear_fit** is a simple optimization comparing a simulated linear equation with a little noise
to a data file, using both a grid-search, and a cross-search.

Acknowledgement
===============
The initial version of this calibration pipeline was built as part of two projects: i4Driving (TU 
Delft, PANTEIA, UNINA, VTI, CTAG, ZF, RDW, TUM, TH AB, CNR-ISTC, DNDE, TJU, UI, UQ, SWISS RE LTD, 
OUW) and Fosim-OTS (TU Delft, Enigmatry, Rijkswaterstaat).

..
    Edited on https://rsted.info.ucl.ac.be/