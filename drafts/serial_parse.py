import os
import struct

import serial
import numpy as np


def main():
    ser = serial.Serial(port="COM7", baudrate=115200, dsrdtr=os.name != "nt")

    def recv():
        ser.read_until(b"\x7E")
        buf = ser.read(16)
        assert ser.read(1) == b"\x7D"

        return struct.unpack("<L6h", buf)

    data = recv()
    dt, ax, ay, az, wx, wy, wz = data

    a = np.array([ax, ay, az], dtype=np.float64) / 256 * 9.81
    w = np.array([wx, wy, wz], dtype=np.float64) / 14.375

    print(dt)
    print(a)
    print(w)

    ser.close()


if __name__ == "__main__":
    main()
