import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CamFeed(Node):

    def __init__(self):
        super().__init__('cam_feed')

        self.publisher = self.create_publisher(Image,'/robovision/image', 10)

        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam")
            raise RuntimeError("Could not open webcam")

        self.timer = self.create_timer(0.1, self.timerCallback)

    def timerCallback(self):
        ok, frame = self.cap.read()

        if not ok:
            self.get_logger().error("Frame not processed")
            return

        cv2.imshow("RoboVision Camera Feed", frame)
        cv2.waitKey(1)
        
        msg = self.bridge.cv2_to_imgmsg( frame, encoding='bgr8')

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = CamFeed()
        rclpy.spin(node)

    except RuntimeError as e:
        print(e)

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()