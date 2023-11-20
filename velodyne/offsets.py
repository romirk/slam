from typing import Literal

import numpy as np

from .definitions import FIRING_CYCLE_DURATION, FIRING_SINGLE_DURATION

TimingTable = np.ndarray[tuple[Literal[32], Literal[12]], np.dtype[np.float_]]


def time_offset(sequence_index, data_point_index):
    return (FIRING_CYCLE_DURATION * sequence_index) + (FIRING_SINGLE_DURATION * data_point_index)


def make_timing_table(dual_mode: bool) -> TimingTable:
    table_indices = np.indices((32, 12))
    y_q_16 = table_indices[0] // 16
    if dual_mode:
        data_block_indices = (table_indices[1] - (table_indices[1] % 2)) + y_q_16
    else:
        data_block_indices = (table_indices[1] * 2) + y_q_16
    data_point_indices = table_indices[0] % 16
    timing_table: TimingTable = time_offset(data_block_indices, data_point_indices)

    return timing_table


def interpolate_azimuth(timing_table: TimingTable, azimuths: np.ndarray[Literal[12]]):
    """
    determine the azimuth of each data point in the packet
    :param timing_table: timing offset of each data point
    :param azimuths: azimuth of each data block (recorded by the first laser)
    :return: azimuth of each data point
    """

    # make sure azimuths are monotonically increasing
    for i in range(1, len(azimuths)):
        if azimuths[i] < azimuths[i - 1]:
            azimuths[i] += 36000

    azs = np.interp(timing_table, timing_table[0], azimuths)

    # deal with last column

    azs[:, -1] = np.interp(timing_table[:, -1], [timing_table[0, -1], timing_table[-1, -1]],
                           [azimuths[-1], azimuths[-1] * 2 - azimuths[-2]])

    # deal with wrap around
    azs = np.mod(azs, 36000)

    return azs
