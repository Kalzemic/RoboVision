import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import serial
import struct
import numpy as np
import time


IMAGE_FACTOR = 2

PORT = '/dev/ttyACM0'
BAUD = 115200
NUM_JOINTS = 2

MODEL_W = 320
MODEL_H = 320


def build_packet(joints):
    payload = struct.pack(
        f'<{NUM_JOINTS}f',
        *joints
    )

    return bytes([0xAA, 0x55]) + payload


def coords_to_angles(coords, image_dim):
    W, H = image_dim

    x, y = coords
    xc, yc = W / 2, H / 2

    x_norm = (x - xc) / xc
    y_norm = (y - yc) / yc

    return (
        -x_norm * (np.pi / IMAGE_FACTOR),
        -y_norm * (np.pi / IMAGE_FACTOR)
    )


class RoboController(Node):

    def __init__(self):
        super().__init__('robo_controller')

        self.ser = serial.Serial(PORT, BAUD)
        time.sleep(2)

        self.object_sub = self.create_subscription(
            Float32MultiArray,
            '/robovision/object',
            self.object_callback,
            10
        )

    def object_callback(self, msg: Float32MultiArray):

        if len(msg.data) < 4:
            return

        x1, y1, x2, y2 = msg.data[:4]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        x_angle, y_angle = coords_to_angles(
            (cx, cy),
            (MODEL_W, MODEL_H)
        )

        packet = build_packet([
            float(x_angle),
            float(y_angle)
        ])

        self.ser.write(packet)


def main(args=None):
    rclpy.init(args=args)

    node = RoboController()

    try:
        rclpy.spin(node)

    finally:
        node.ser.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()