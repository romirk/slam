import asyncio
import socket
from datetime import datetime

import cv2
import numpy as np

from .definitions import FIRING_RECHARGE_DURATION, LASER_ID_TO_VERTICAL_ANGLE, LASER_ID_TO_VERTICAL_CORRECTION, \
    LaserDataPacket
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
        v2 = self.points[2] - self.points[1]
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
            print(f"intersection: {intersection}")
            # add some noise to the distance
            distance = np.linalg.norm(intersection) + _RNG.normal(0, 0.01)
            assert distance > 0
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


class Wall:
    def __init__(self, points: np.ndarray, reflectivity: np.uint8):
        self.p1 = points[0]
        self.p2 = points[1]
        self.reflectivity = reflectivity
        self.normal = self.__compute_normal()
        print(f"normal: {self.normal}")

    def __compute_normal(self):
        """
        Compute the normal of the surface
        :return: normal vector
        """
        # compute the normal
        x1, y1 = self.p1[0], self.p1[1]
        x2, y2 = self.p2[0], self.p2[1]
        return np.array([y2 - y1, x1 - x2, 0])

    async def intersect(self, laser_id: int, laser_azimuth: float):
        """
        Determine if a laser intersects with the plane. Plane is parallel to the z axis.
        LiDAR origin is assumed to be at (0, 0, 0). We apply a vertical correction to the laser based on the laser ID.
        :param laser_azimuth: laser azimuth
        :param laser_id: laser ID
        :return: None if no intersection, otherwise the distance to the intersection
        """
        laser_elevation = LASER_ID_TO_VERTICAL_ANGLE[laser_id]
        laser_elevation = np.radians(laser_elevation)
        laser_azimuth = np.radians(laser_azimuth)
        # compute the direction vector of the laser
        laser_dir = np.array([np.cos(laser_azimuth) * np.cos(laser_elevation),
                              np.sin(laser_azimuth) * np.cos(laser_elevation),
                              np.sin(laser_elevation)])
        laser_origin = np.array([0, 0, LASER_ID_TO_VERTICAL_CORRECTION[laser_id] / 1000])
        # compute the distance to the parallel plane
        if np.dot(self.normal, laser_dir) == 0:
            return None, None
        d = np.dot(self.normal, laser_origin - self.p1) / np.dot(self.normal, laser_dir)
        p = laser_dir * d + laser_origin
        if d < 0 or d > 100:
            return None, None
        distance = d + _RNG.normal(0, 0.01)
        reflectivity = self.reflectivity + _RNG.normal(0, 1)
        return distance, reflectivity


async def packet_generator():
    """
    Generate a packet of data
    :return: a packet of data
    """

    # surface = Surface(np.array([[0, 1, 1], [1, 1, -1], [-1, 1, -1]]), np.uint8(75))
    surface = Wall(np.array([[-1, 1, 0], [1, 1, 0]]), np.uint8(75))
    packet = np.zeros((), dtype=LaserDataPacket)

    tt = make_timing_table(False)
    azs = interpolate_azimuth(tt, __AZIMUTHS)
    # packet["header"] = np.uint8(0x00)

    while True:
        # get microseconds since the top of the hour
        toh = datetime.now().replace(minute=0, second=0, microsecond=0)
        packet["timestamp"] = np.uint32((datetime.now() - toh).microseconds)

        # packet["factory"] = __FACTORY_BYTES

        for i in range(24):
            block = i // 2
            sector = i % 2

            for laser_id, el in enumerate(LASER_ID_TO_VERTICAL_ANGLE):
                # compute the distance to the surface
                n = block
                k = laser_id + 16 * sector
                # # add some noise to the azimuth
                a = azs[k, n] + _RNG.normal(0, 0.01)
                packet["blocks"][n]["azimuth"] = a
                distance, reflectivity = await surface.intersect(laser_id, a / 100)
                if distance is not None:
                    # print(f"laser: {laser_id} azimuth: {a} distance: {distance} reflectivity: {reflectivity}")
                    packet["blocks"][n]["data"][k]["range"] = np.uint16(distance)
                    packet["blocks"][n]["data"][k]["reflectivity"] = np.uint8(reflectivity)
                else:
                    packet["blocks"][n]["data"][k]["range"] = np.uint16(0)
                    packet["blocks"][n]["data"][k]["reflectivity"] = np.uint8(0)

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


SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200


async def draw_env(packets: list[LaserDataPacket], azs: np.ndarray):
    """
    create a top-down view of the environment in text
    :param packets: list of packets to draw
    :return: None
    """

    if not len(packets):
        return

    # create a grid
    grid = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=np.uint8)
    max_y = SCREEN_HEIGHT // 2 - 1
    max_x = SCREEN_WIDTH // 2 - 1
    origin = np.array([max_y, max_x])

    # add two reference points
    L = len(packets)
    grid_pts = []
    # draw the environment

    for i, packet in enumerate(packets):
        for n, block in enumerate(packet["blocks"][0]):
            data = block["data"]
            for k, data_point in enumerate(data):
                if data_point["range"] == 0:
                    continue
                # compute the distance to the surface
                laser_id = k % 16
                sector = k // 16
                block = n
                # compute the azimuth
                a = azs[k, n] / 100
                # compute the distance to the surface
                distance = data_point["range"]
                # print(f"laser: {laser_id} azimuth: {a} distance: {distance} reflectivity: {data['reflectivity']}")
                # compute the elevation angle
                elevation = LASER_ID_TO_VERTICAL_ANGLE[laser_id]
                # compute the x and y coordinates
                x = distance * np.cos(np.radians(elevation)) * np.cos(np.radians(a))
                y = distance * np.cos(np.radians(elevation)) * np.sin(np.radians(a))
                # print(f"laser: {laser_id} azimuth: {a} distance: {distance} reflectivity: {reflectivity}")
                grid[int(y + max_y), int(x + max_x)] += data_point["reflectivity"] / L

    text_mode = False
    if text_mode:
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
        return
    thr = 0.5
    grid[grid < thr] = 0
    grid[grid >= thr] = 255
    img = cv2.resize(grid, (SCREEN_WIDTH * 4, SCREEN_HEIGHT * 4), grid, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("lidar", img)
    cv2.waitKey(1)


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
                packets = []
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
