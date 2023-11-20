from enum import Enum
from typing import Literal, Type, TypeAlias

import numpy as np

SphericalCoordinates: Type = np.ndarray[Literal[3], np.dtype[np.float_]]  # [r, omega, alpha]
"""Spherical coordinates. [radius, elevation, azimuth]"""
CartesianCoordinates: Type = np.ndarray[Literal[3], np.dtype[np.float_]]  # [x, y, z]
"""Cartesian coordinates. [x, y, z]"""

FIRING_CYCLE_DURATION: float = 55.296  # [µs]
"""Duration of the firing sequence in microseconds."""

FIRING_SINGLE_DURATION: float = 2.304  # [µs]
"""The cycle time between firings in microseconds."""

FIRING_RECHARGE_DURATION: float = 18.43  # [µs]
"""The time required to recharge the laser after firing in microseconds."""

LASER_ID_TO_VERTICAL_ANGLE: np.ndarray = np.array(
    [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
)
"""Vertical angle corresponding to each laser ID (degrees)."""

LASER_ID_TO_VERTICAL_CORRECTION: np.ndarray = np.array(
    [11.2, -0.7, 9.7, -2.2, 8.1, -3.7, 6.6, -5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, 0.7, -11.2]
)
"""Vertical correction corresponding to each laser ID (mm)."""

LaserDataPoint: np.dtype = np.dtype(
    [
        ("range", np.uint16),
        ("reflectivity", np.uint8)
    ]
)
"""
Data type of a single laser data point.

A data point is represented in the packet by three bytes - two bytes of distance and one byte of calibrated 
reflectivity. The distance is an unsigned integer. It has 2 mm granularity. Hence, a reported value of 51,154 
represents 102,308 mm or 102.308 m. Calibrated reflectivity is reported on a scale of 0 to 255. The
elevation angle (ω) is inferred based on the position of the data point within a data block.

A distance of 0 indicates a non-measurement. The laser is either off or a measurable reflection was not 
returned in time.
"""

LaserAzimuth: Type = np.uint16
"""
A two-byte azimuth value (α) appears after the flag bytes at the beginning of each data block. The azimuth is an 
unsigned integer. It represents an angle in hundredths of a degree. Therefore, a raw value of 27742 should be 
interpreted as 277.42°.
Valid values for azimuth range from 0 to 35999. Only one azimuth value is reported per data block.
"""

LaserDataBlock = np.dtype(
    [
        ("flag", np.uint16),
        ("azimuth", LaserAzimuth),
        ("data", LaserDataPoint, 32)
    ]
)
"""
The information from two firing sequences of 16 lasers is contained in each data block. Each packet contains 
the data from 24 firing sequences in 12 data blocks.
Only one Azimuth is returned per data block.
"""

TimeStamp: Type = np.uint32
"""
The four-byte time stamp is a 32-bit unsigned integer marking the moment of the first data point in the first firing 
sequence of the first data block. The time stamp’s value is the number of microseconds elapsed since the top of the 
hour. The number ranges from 0 to 3,599,999,999, the number of microseconds in one hour.
"""


class ReturnMode(Enum):
    """The Return Mode byte indicates how the packet’s azimuth and data points are organized."""
    STRONGEST = 0x37
    LAST_RETURN = 0x38
    DUAL_RETURN = 0x39


ProductID: TypeAlias = np.uint8
"""
Every sensor model line has its lasers arrayed vertically at slightly different angles. Use the Product ID byte to 
identify the correct set of vertical(or elevation) angles. Product IDs are not unique and may be shared by different 
sensors.
"""

MODEL_TO_PRODUCT_ID: dict[str, ProductID] = {
    "HDL-32E": np.uint8(0x21),
    "VLP-16": np.uint8(0x22),
    "Puck LITE": np.uint8(0x22),
    "Puck Hi-Res": np.uint8(0x24),
    "VLP-32C": np.uint8(0x28),
    "Velarray": np.uint8(0x31),
    "Vls-128": np.uint8(0xA1),
}

LaserDataBlockArray: Type = np.ndarray[Literal[12], LaserDataBlock]  # type: ignore
"""Array of 12 data blocks."""

LaserDataPacket: np.dtype = np.dtype(
    [
        ("header", np.uint8, 42),
        ("blocks", LaserDataBlock, 12),
        ("timestamp", TimeStamp),
        ("factory", np.uint8, 2)
    ]
)

LaserPositionPacket: np.dtype = np.dtype(
    [
        ("HEAD", np.uint8, 187),  # Reserved for future use (null)
        ("T_top", np.uint8),  # Temperature of the top board (Celsius) { 0 <= T_top <= 150 }
        ("T_bot", np.uint8),  # Temperature of the bottom board (Celsius) { 0 <= T_bot <= 150 }
        ("T_adc", np.uint8),  # Temperature when ADC calibration last ran (Celsius) { 0 <= T_adc <= 150 }
        ("dT_adc", np.int16),  # Change in temperature since last ADC calibration (Celsius) { -150 <= dT_adc <= 150 }
        ("dt_adc", np.uint32),  # Time since last ADC calibration (seconds) { 0 <= dt_adc <= 2^32 - 1 }
        ("R_adc", np.uint8),  # Reason for last ADC calibration { 0 <= R_adc <= 4 }
        ("S_adc", np.uint8),  # Status of current ADC calibration (bitmask)
        ("TOH", TimeStamp),  # microseconds since top of the hour
        ("PPS", np.uint8),  # Pulse per second status; 0: Absent, 1: Synchronizing, 2: Locked, 3: Error
        ("TS", np.uint8),  # Thermal status; 0: OK, 1: Thermal shutdown
        ("T_ts", np.uint8),  # Temperature of unit when thermal shutdown occurred (Celsius) { 0 <= LTST <= 150 }
        ("T_pow", np.uint8),  # Temperature of unit (bottom board) at power up
        ("NMEA", np.uint8, 128),  # NMEA sentence (GPRMC or GPGGA)
        ("TAIL", np.uint8, 178)  # Unused (null)
    ]
)
