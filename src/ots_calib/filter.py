"""
Filters supply a filtered representation of data, made relevant for a KPI or error function.

@author: wjschakel
"""
from ots_calib.data import Data
import pandas as pd


class Filter(Data):
    """
    Filter template.

    Attributes
    ----------
    _underlying_data : Data
        Underlying data that will be filtered.

    _underlying_data_frame : DataFrame
        Pointer to underlying data to check whether the underlying data has changed and cached
        values should be cleared.
    """

    def __init__(self, underlying_data: Data):
        """
        Constructor.
        """
        self._underlying_data = underlying_data
        self._underlying_data_frame = None

    def get_underlying_data(self) -> pd.DataFrame:
        """
        Returns the underlying data.
        """
        self._underlying_data_frame = self._underlying_data.get_data()
        return self._underlying_data_frame

    def has_new_underlying_data(self) -> bool:
        """
        Returns whether get_underlying_data() will return new underlying data.
        """
        return self._underlying_data.get_data() is not self._underlying_data_frame


class Range(Filter):
    """
    Returns a range of the underlying data.

    Attributes
    ----------
    _column : str
        Name of column in the data.

    _min : float
        Minimum column value (inclusive).

    _max : float
        Maximum column value (exclusive).

    _filtered_data : DataFrame
        Filtered data.
    """

    def __init__(self, underlying_data: Data, column: str, minimum: float, maximum: float):
        """
        Constructor.
        """
        super().__init__(underlying_data)
        self._column = column
        self._min = minimum
        self._max = maximum

    def get_data(self) -> pd.DataFrame:
        """
        Returns the time-filtered data.
        """
        if self.has_new_underlying_data():
            d = self.get_underlying_data()
            self._filtered_data = d[d[self._column] >= self._min & d[self._column] < self._max]
        return self._filtered_data


class SpaceTimeRegion(Filter):
    """
    Space-time region which supplies traffic performance indicators as defined by Edie (1965) and
    the data contained within the region.

    Edie, L. (1965) Discussion of traffic stream measurements and definitions, in Proceedings of
    the 2nd International Symposium on the Theory of Traffic Flow, Paris: OECD, pp. 139–154.

    Attributes
    ----------
    _underlying_data : Data
        Underlying data that should be pre-processed and pre-filtered to contain data along a path
        for which the distance column has increasing values along the path, data contains only
        relevant vehicles, and the series column indicates parts of each vehicle trajectory where
        it was on the path. So for example, when a vehicle leaves the path by a lane change, and
        enters it again later, the series column should have a different value after entering.

    _vehicle_id_column : str
        Name of the column containing a unique vehicle id for each vehicle.

    _series_column : str
        Name of the column containing a unique series value for each part during which the vehicle
        is in the path.

    _col_t : str
        Name of the time column.

    _min_t : float
        Minimum time value of the space-time region.

    _max_t : float
        Maximum time value of the space-time region.

    _col_x : str
        Name of the distance column.

    _min_x : float
        Minimum distance value of the space-time region.

    _max_x : float
        Maximum distance value of the space-time region.

    _path_data_frame : DataFrame
        DataFrame containing the data from the path that is within the space-time region.

    _total_time_spent : float
        Cached value of total time spent.

    _production : float
        Cached value of production (total distance covered).
    """

    def __init__(self, underlying_data: Data, vehicle_id_column: str, series_column: str,
                 col_t: str, min_t: float, max_t: float,
                 col_x: str, min_x: float, max_x: float):
        """
        Constructor.
        """
        if min_t >= max_t or min_x >= max_x:
            raise ValueError(f'SpaceTimeRegion has improper bounds t=[{min_t}, {max_t}], ' +
                             f'x=[{min_x} {max_x}]')
        super().__init__(underlying_data)
        self._vehicle_id_column = vehicle_id_column
        self._series_column = series_column
        self._col_t = col_t
        self._min_t = min_t
        self._max_t = max_t
        self._col_x = col_x
        self._min_x = min_x
        self._max_x = max_x

        self._path_data_frame = None
        self._total_time_spent = None
        self._production = None

    def get_data(self) -> pd.DataFrame:
        """
        Returns the data contained within the space-time region.
        """
        self._reset_on_new_underlying_data()
        if not self._path_data_frame:
            df = self._underlying_data_frame
            in_t = df[self._col_t] >= self._min_t & df[self._col_t] <= self._max
            in_x = df[self._col_x] >= self._min_x & df[self._col_x] <= self._max_x
            self._path_data_frame = df[in_t & in_x]
        return self._path_data_frame

    def get_area(self) -> float:
        """
        Returns the area of the region (time x space).
        """
        return (self._min_t - self._min_t) * (self._max_x - self._min_x)

    def get_total_time_spent(self) -> float:
        """
        Returns the total time spent within the region.
        """
        self._compute_total_time_spent_and_production()
        return self._total_time_spent

    def get_production(self) -> float:
        """
        Returns the production (total distance covered) within the region.
        """
        self._compute_total_time_spent_and_production()
        return self._production

    def get_flow(self) -> float:
        """
        Returns the flow within the region.
        """
        self._reset_on_new_underlying_data()
        return self.get_production() / self.get_area()

    def get_density(self) -> float:
        """
        Returns the density within the region.
        """
        self._reset_on_new_underlying_data()
        return self.get_total_time_spent() / self.get_area()

    def get_mean_speed(self) -> float:
        """
        Returns the mean speed within the region.
        """
        return self.get_production() / self.get_total_time_spent()

    def _reset_on_new_underlying_data(self) -> bool:
        """
        Clears cached values when the underlying data has changed.
        """
        if self.has_new_underlying_data():
            self._path_data_frame = None
            self._total_time_spent = None
            self._production = None

    def _compute_total_time_spent_and_production(self):
        """
        Computes the total time spent and production interpolating data at the edges of the
        space-time region.
        """
        self._reset_on_new_underlying_data()

        if not self._total_time_spent:
            self._total_time_spent = 0
            self._production = 0

            # Loop all vehicles contained in the space-time region
            all_data = self.get_underlying_data()
            region_data = self.get_data()
            for vehicle_id in set(region_data[self._vehicle_id_column]):
                all_veh_data = all_data.loc[all_data[self._vehicle_id_column] == vehicle_id]
                region_veh_data = region_data.loc[region_data[self._vehicle_id_column]
                                                  == vehicle_id]

                # First vehicle data in space-time region
                index_first = region_veh_data[self._col_t].idxmin()
                t_in = region_veh_data[self._col_t].iloc[index_first]
                x_in = region_veh_data[self._col_x].iloc[index_first]
                series_in = all_veh_data[self._series_column].iloc[index_first]
                t_first = t_in

                # If the vehicle existed before, interpolate
                is_before_t = all_veh_data[self._col_t] < self._min_t
                is_before_x = all_veh_data[self._col_x] < self._min_x
                is_same_series_before = all_veh_data[self._series_column] == series_in
                before = all_veh_data.loc[(is_before_t | is_before_x) & is_same_series_before]
                if before:
                    index_before = before[self._col_t].idxmax()
                    t_before = before[self._col_t].iloc[index_before]
                    x_before = before[self._col_x].iloc[index_before]
                    v = (x_in - x_before) / (t_in - t_before)
                    t_x = t_before + (self._min_x - x_before) / v
                    if t_x < self._min_t:
                        t_in = self._min_t
                        x_in = x_before + (self._min_t - t_before) * v
                    else:
                        t_in = t_x
                        x_in = self._min_x

                # Last vehicle data in space-time region
                index_last = region_veh_data[self._col_t].idxmax()
                t_out = region_veh_data[self._col_t].iloc[index_last]
                x_out = region_veh_data[self._col_x].iloc[index_last]
                series_out = all_veh_data[self._series_column].iloc[index_last]
                t_last = t_out

                # If the vehicle existed after, interpolate
                is_after_t = all_veh_data[self._col_t] > self._max_t
                is_after_x = all_veh_data[self._col_x] > self._max_x
                is_same_series_after = all_veh_data[self._series_column] == series_out
                after = all_veh_data.loc[(is_after_t | is_after_x) & is_same_series_after]
                if after:
                    index_after = after[self._col_t].idxmin()
                    t_after = after[self._col_t].iloc[index_after]
                    x_after = after[self._col_x].iloc[index_after]
                    v = (x_after - x_out) / (t_after - t_out)
                    t_x = t_out + (self._max_x - x_out) / v
                    if t_x < self._max_t:
                        t_out = t_x
                        x_out = self._max_x
                    else:
                        t_out = self._max_t
                        x_out = x_out + (self._min_t - t_out) * v

                # Loop series
                for series in set(region_veh_data[self._series_column]):
                    series_data = region_veh_data.loc[region_veh_data[self._series_column]
                                                      == series]
                    index_min = series_data[self._col_t].idxmin()
                    t0 = series_data[self._col_t].iloc[index_min]
                    x0 = series_data[self._col_x].iloc[index_min]
                    if t0 == t_first:
                        t0 = t_in
                        x0 = x_in

                    index_max = series_data[self._col_t].idxmax()
                    t1 = series_data[self._col_t].iloc[index_max]
                    x1 = series_data[self._col_x].iloc[index_max]
                    if t1 == t_last:
                        t1 = t_out
                        x1 = x_out

                    self._total_time_spent += (t1 - t0)
                    self._production += (x1 - x0)
