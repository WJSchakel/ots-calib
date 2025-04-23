"""
Filters and pre-processes OTS trajectory data to describe a path.

@author: wjschakel
"""
from ots_calib.data import Data
import pandas as pd


class OtsLane(object):
    """
    Describes lane as part of a link within a path.

    Attributes
    ----------
    _link_id : str
        Link id.

    _lane_id : str
        Lane id.

    _start_distance : float
        Start distance along the path of the lane.
    """

    def __init__(self, link_id: str, lane_id: str, start_distance: float):
        """
        Constructor.
        """
        self._link_id = link_id
        self._lane_id = lane_id
        self._start_distance = start_distance

    def get_link_id(self) -> str:
        """
        Returns the link id.
        """
        return self._link_id

    def get_lane_id(self) -> str:
        """
        Returns the lane id.
        """
        return self._lane_id

    def get_start_distance(self) -> float:
        """
        Returns the start distance along the path of the lane.
        """
        return self._start_distance


class OtsPath(Data):
    """
    Returns data relevant to a path. The underlying data should already be filtered, or should be
    filtered later, regarding other filtering such as by vehicle type.

    Attributes
    ----------
    _underlying_data : Data
        Underlying data.

    _lanes : set[OtsLane]
        Set of lanes in the path.

    _series_column : str
        Name of the column that represents series. A serie is a part of a trajectory that will or
        will not be on the path. Output data will have one value for the serie for each part of a
        vehicle trajectory during which the vehicle is consecutively on the path.

    _link_column : str
        Name of the link column in the data.

    _lane_column : str
        Name of the lane column in the data.

    _vehicle_id_column : str
        Name of the vehicle id column in the data.

    _distance_column : str
        Name of the distance column in the data.

    _underlying_data_frame : DataFrame
        Use to check whether new data is underlying, and hence filtering should be redone.

    _path_data_frame : DataFrame
        Cached filtered data for the path.
    """

    def __init__(self, underlying_data: Data, lanes: set[OtsLane],
                 series_column: str='traj#', link_column: str='linkId', lane_column: str='laneId',
                 vehicle_id_column: str='gtuId', distance_column: str='x'):
        """
        Constructor.
        """
        self._underlying_data = underlying_data
        self._lanes = lanes
        self._series_column = series_column
        self._link_column = link_column
        self._lane_column = lane_column
        self._vehicle_id_column = vehicle_id_column
        self._distance_column = distance_column

        self._underlying_data_frame = None
        self._path_data_frame = None

    def get_data(self) -> pd.DataFrame:
        """
        Filters the underlying data such that the returned data only contains the lanes of the
        path. The distance column is updated to increase along the path. The series column is
        updated to contain unique values per part where a vehicle is within the path consecutively.
        """
        all_df = self._underlying_data.get_data()
        if self._underlying_data_frame is not all_df:
            self._underlying_data_frame = all_df

            # Filter for lanes
            these = [False] * len(all_df)
            for lane in self._lanes:
                these = these | (all_df[self._link_column] == lane.get_link_id() &
                                 all_df[self._lane_column] == lane.get_lane_id())
            path_df = all_df.loc[these]

            # Increase position per lane
            for lane in self._lanes:
                these = (path_df[self._link_column] == lane.get_link_id() &
                         path_df[self._lane_column] == lane.get_lane_id())
                dx = lane.get_start_distance()
                path_df.loc[these, self._distance_column] += dx

            # Assign series numbers
            new_serie_num = 0
            for vehicle_id in set(path_df[self._vehicle_id_column]):
                start_positions = list()
                in_paths = list()
                serie_ids = list()
                of_vehicle = all_df[self._vehicle_id_column] == vehicle_id
                for serie_id in set(all_df.loc[of_vehicle, self._series_column]):
                    start_positions.append(all_df.loc[all_df[self._series_column] ==
                                           serie_id, self._distance_column].min())
                    in_paths.append(len(path_df.loc[path_df[self._series_column] == serie_id]) > 0)
                    serie_ids.append(serie_id)
                prev_in_path = None
                for _, in_path, serie_id in sorted(zip(start_positions, in_paths, serie_ids)):
                    if in_path:
                        if not prev_in_path:
                            new_serie_num += 1
                        path_df.loc[path_df[self._series_column] ==
                                    serie_id, self._series_column] = new_serie_num
                    prev_in_path = in_path

            self._path_data_frame = path_df
        return self._path_data_frame
