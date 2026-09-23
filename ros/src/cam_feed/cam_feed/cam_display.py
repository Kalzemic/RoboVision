import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import cv2
import time

class CamDisplay(Node):

    def __init__(self):
        super().__init__("Cam_Display")

        self.bridge = CvBridge()

        self.latest_object = None
        self.timeout = 0.15
        self.object_time = 0
        self.image_sub = self.create_subscription(
            Image,
            '/robovision/image',
            self.image_callback,
            10
        )

        self.object_sub = self.create_subscription(
            Float32MultiArray,
            '/robovision/object',
            self.object_callback,
            10
        )

        self.get_logger().info("Camera Display Initialized")


    def object_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.latest_object = msg.data
            self.object_time = time.time()


    def image_callback(self, image_msg: Image):

        frame = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding='bgr8'
        )

        if self.latest_object is not None and time.time() - self.object_time < self.timeout:

            x1, y1, x2, y2 = self.latest_object[:4]
            h, w = frame.shape[:2]

            scale_x = w / 320.0
            scale_y = h / 320.0

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            cv2.rectangle( frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2 )

        cv2.imshow(
            "RoboVision Camera Feed",
            frame
        )

        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = CamDisplay()

    try:
        rclpy.spin(node)

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()