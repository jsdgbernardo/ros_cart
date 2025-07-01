import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Point, '/pose/nose_point', 10)
        self.bridge = CvBridge()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.last_nose = None  # for smoothing

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            # Get the nose landmark (ID 0)
            h, w, _ = frame.shape
            nose = results.pose_landmarks.landmark[0]
            x, y, z = nose.x * w, nose.y * h, nose.z

            # Smoothing
            if self.last_nose is not None:
                alpha = 0.5
                x = alpha * x + (1 - alpha) * self.last_nose.x
                y = alpha * y + (1 - alpha) * self.last_nose.y
                z = alpha * z + (1 - alpha) * self.last_nose.z

            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = float(z)

            self.pub.publish(point)
            self.last_nose = point
            self.get_logger().info(f"Nose @ x:{x:.1f}, y:{y:.1f}, z:{z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
