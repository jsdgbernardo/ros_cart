import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

class PhoneCamNode(Node):
    def __init__(self):
        super().__init__('phone_cam_node')
        # Publisher for CompressedImage
        self.publisher = self.create_publisher(CompressedImage, '/camera/image/compressed', 10)

        self.stream_url = 'http://192.168.0.220:8080/video'  # Change to your phone IP
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open video stream: {self.stream_url}')
            return

        self.timer = self.create_timer(1/30.0, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from stream')
            return

        # Compress frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            self.get_logger().warn('Failed to compress frame')
            return

        msg = CompressedImage()
        msg.format = 'jpeg'
        msg.data = np.array(buffer).tobytes()
        msg.header.stamp = self.get_clock().now().to_msg()

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PhoneCamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
