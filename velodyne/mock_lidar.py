import asyncio
import socket
from datetime import datetime

import numpy as np

from .definitions import FIRING_RECHARGE_DURATION, LASER_ID_TO_VERTICAL_ANGLE, LASER_ID_TO_VERTICAL_CORRECTION, \
    LaserDataBlock, LaserDataPacket
from .offsets import interpolate_azimuth, make_timing_table

_RNG = np.random.default_rng()
__FACTORY_BYTES = np.uint16(0x37) << 8 | np.uint8(0x22)
__AZIMUTHS = np.linspace(0, 33000, 12)  # + __RNG.normal(0, 100, 12)


def cross(a, b):
    return np.cross(a, b)


class Surface:
    def __init__(self, points: np.ndarray, reflectivity: np.uint8):
        self.points = points[:, :3]
        self.reflectivity = reflectivity
        self.normal = self.__compute_normal()
        self.area = np.linalg.norm(self.normal)

    def __compute_normal(self):
        """
        Compute the normal of the surface
        :return: normal vector
        """
        if len(self.points) < 3:
            raise ValueError("Surface must have at least 3 points")
        # compute the normal
        v1 = self.points[1] - self.points[0]
        v2 = self.points[2] - self.points[0]
        return np.cross(v1, v2)

    async def intersect(self, laser_id: int, laser_azimuth: float):
        """
        Determine if a laser intersects with the surface. LiDAR origin is assumed to be at (0, 0, 0).
        We apply a vertical correction to the laser based on the laser ID.
        :param laser_azimuth: laser azimuth
        :param laser_id: laser ID
        :return: None if no intersection, otherwise the distance to the intersection
        """
        laser_elevation = LASER_ID_TO_VERTICAL_ANGLE[laser_id]  # degrees
        laser_elevation = np.radians(laser_elevation)  # radians
        laser_azimuth = np.radians(laser_azimuth)  # radians
        # compute the direction vector of the laser
        laser_dir = np.array([np.cos(laser_azimuth) * np.cos(laser_elevation),
                              np.sin(laser_azimuth) * np.cos(laser_elevation),
                              np.sin(laser_elevation)])
        laser_origin = np.array([0, 0, LASER_ID_TO_VERTICAL_CORRECTION[laser_id] / 1000])
        # compute the distance to the surface
        d = np.dot(self.normal, self.points[0])
        # compute the distance from the laser to the surface
        t = (d - np.dot(self.normal, laser_dir)) / np.dot(self.normal, laser_dir)
        # compute the intersection point
        intersection = laser_dir * t + laser_origin
        # check if the intersection point is within the triangle
        if self.__point_in_triangle(intersection):
            # add some noise to the distance
            distance = np.linalg.norm(intersection) + _RNG.normal(0, 0.01)
            # print(f"distance: {distance}")
            # add some noise to the reflectivity
            reflectivity = self.reflectivity + _RNG.normal(0, 1)
            return distance, reflectivity
        else:
            return None, None

    def __point_in_triangle(self, intersection: np.ndarray):
        """
        Determine if a point is inside a triangle
        :param intersection: point to test
        :return: True if the point is inside the triangle, False otherwise
        """
        # compute vectors
        v0 = self.points[2] - self.points[0]
        v1 = self.points[1] - self.points[0]
        v2 = self.points[2] - self.points[1]

        c0 = intersection - self.points[0]
        c1 = intersection - self.points[1]
        c2 = intersection - self.points[2]

        d02 = np.dot(self.normal, cross(v2, c2))
        d00 = np.dot(self.normal, cross(v0, c0))
        d01 = np.dot(self.normal, cross(v1, c1))

        return d00 > 0 and d01 > 0 and d02 > 0


async def packet_generator():
    """
    Generate a packet of data
    :return: a packet of data
    """

    surface = Surface(np.array([[0, 1, 1], [1, 1, -1], [-1, 1, -1]]), np.uint8(75))
    packet = np.zeros((), dtype=LaserDataPacket)

    tt = make_timing_table(False)
    azs = interpolate_azimuth(tt, __AZIMUTHS)
    # packet["header"] = np.uint8(0x00)

    while True:
        # get microseconds since the top of the hour
        toh = datetime.now().replace(minute=0, second=0, microsecond=0)
        packet["timestamp"] = np.uint32((datetime.now() - toh).microseconds)

        # packet["factory"] = __FACTORY_BYTES

        blocks = np.ndarray((12,), dtype=LaserDataBlock)
        blocks["azimuth"] = np.uint16(__AZIMUTHS)

        for i in range(24):
            block = i // 2
            sector = i % 2

            for laser_id, el in enumerate(LASER_ID_TO_VERTICAL_ANGLE):
                # compute the distance to the surface
                n = block
                k = laser_id + 16 * sector
                # # add some noise to the azimuth
                a = azs[k, n] + _RNG.normal(0, 0.01)
                distance, reflectivity = await surface.intersect(laser_id, a)
                if distance is not None:
                    blocks[n]["data"][k]["range"] = np.uint16(distance)
                    blocks[n]["data"][k]["reflectivity"] = np.uint8(reflectivity)
                else:
                    blocks[n]["data"][k]["range"] = np.uint16(0)
                    blocks[n]["data"][k]["reflectivity"] = np.uint8(0)

        packet["blocks"] = blocks
        yield packet


shutdown = asyncio.Event()


async def packet_writer(sock: socket):
    """
    Write packets to the socket
    :param sock: socket to write to
    :return: None
    """
    try:
        async for packet in packet_generator():
            if shutdown.is_set():
                break
            sock.send(packet.tobytes())
            # print(">")
            await asyncio.sleep(FIRING_RECHARGE_DURATION / 100)
            # print(">")
    except asyncio.CancelledError:
        pass
    finally:
        sock.close()


SCREEN_HEIGHT = 24
SCREEN_WIDTH = 64


async def draw_env(packets: list[LaserDataPacket], azs: np.ndarray):
    """
    create a top-down view of the environment in text
    :param packets: list of packets to draw
    :return: None
    """

    # get the maximum range

    # create a grid
    grid = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=np.uint32)
    max_y = SCREEN_HEIGHT // 2 - 1
    max_x = SCREEN_WIDTH // 2 - 1
    origin = np.array([max_y, max_x])

    # add two reference points
    L = len(packets)
    grid_pts = []
    # draw the environment
    for packet in packets:
        for n, block in enumerate(packet["blocks"]):
            points = block[0]["data"]
            for k, point in enumerate(points):
                if point["range"] > 0:
                    az = azs[k, n]
                    # el = LASER_ID_TO_VERTICAL_ANGLE[k % 16]
                    r = point["range"]
                    x = int(np.cos(np.radians(az)) * r)
                    y = int(np.sin(np.radians(az)) * r)
                    p = np.array([y, x]) + origin
                    # print(f"{p[0]} {p[1]} {point['reflectivity']}")
                    grid_pts.append(np.array([p[0], p[1], point["reflectivity"] / L]))

    grid_pts = np.array(grid_pts)
    max_p_x = abs(max(grid_pts, key=lambda _p: abs(_p[0]))[0]) * 2
    max_p_y = abs(max(grid_pts, key=lambda _p: abs(_p[1]))[1]) * 2
    grid_pts[:, 0] = max_x * grid_pts[:, 0] / max_p_x
    grid_pts[:, 1] = max_y * grid_pts[:, 1] / max_p_y
    for p in grid_pts:
        grid[int(p[0]), int(p[1])] += p[2]

    print(f"max range: {max_p_y} {max_p_x}")
    # construct the string
    s = ""
    for y in range(SCREEN_HEIGHT):
        for x in range(SCREEN_WIDTH):
            c = grid[x, y] / 255
            if c < 0.1:
                if y == max_y:
                    s += "+" if x == max_x else "-"
                    continue
                if x == max_x:
                    s += "|"
                    continue
                s += " "
            elif c < 0.2:
                s += "."
            elif c < 0.3:
                s += ":"
            elif c < 0.4:
                s += "-"
            elif c < 0.5:
                s += "="
            elif c < 0.6:
                s += "+"
            elif c < 0.7:
                s += "*"
            elif c < 0.8:
                s += "#"
            elif c < 0.9:
                s += "%"
            else:
                s += "@"
        s += "\n"
    s += "---"
    print(s)


async def packet_reader(sock: socket):
    """
    Read packets from the socket
    :param sock: socket to read from
    :return: None
    """
    sock.setblocking(False)
    sock.settimeout(0.1)
    try:
        tt = make_timing_table(False)
        azs = interpolate_azimuth(tt, __AZIMUTHS)

        packets = []
        c = 0
        buffer = bytearray()
        while shutdown.is_set() is False:
            data = sock.recv(LaserDataPacket.itemsize)
            buffer.extend(data)
            if len(buffer) < LaserDataPacket.itemsize:
                await asyncio.sleep(0.001)
                continue
            data = buffer[:LaserDataPacket.itemsize]
            buffer = buffer[LaserDataPacket.itemsize:]
            packet = np.frombuffer(data, dtype=LaserDataPacket)
            packets.append(packet)
            c += 1
            if c == 10:
                c = 0
                await draw_env(packets, azs)

            await asyncio.sleep(FIRING_RECHARGE_DURATION / 100)
    except asyncio.CancelledError:
        pass
    finally:
        sock.close()


async def main():
    rsock, wsock = socket.socketpair(socket.AF_UNIX)
    try:
        await asyncio.gather(packet_writer(wsock), packet_reader(rsock))
    except asyncio.CancelledError:
        shutdown.set()


if __name__ == '__main__':
    asyncio.run(main())
