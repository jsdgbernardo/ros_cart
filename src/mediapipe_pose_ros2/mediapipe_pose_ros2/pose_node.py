import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # Subscribe to phone camera compressed image
        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
        )

        # Publisher for nose landmark coordinates
        self.pub = self.create_publisher(Point, '/pose/nose_point', 10)

        # Publisher for annotated image
        self.annotated_pub = self.create_publisher(CompressedImage, '/pose/image/compressed', 10)

        # MediaPipe pose detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils

        self.last_nose = None  # Previous nose position for smoothing

    def image_callback(self, msg):
        # Decode the compressed JPEG to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode JPEG image.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # If landmarks detected
        if results.pose_landmarks:
            h, w, _ = frame.shape
            try:
                nose = results.pose_landmarks.landmark[0]
                x, y, z = nose.x * w, nose.y * h, nose.z

                # Smoothing nose coordinates
                if self.last_nose is not None:
                    alpha = 0.5
                    x = alpha * x + (1 - alpha) * self.last_nose.x
                    y = alpha * y + (1 - alpha) * self.last_nose.y
                    z = alpha * z + (1 - alpha) * self.last_nose.z

                point = Point(x=float(x), y=float(y), z=float(z))
                self.pub.publish(point)
                self.last_nose = point

                self.get_logger().info(f'Nose @ x:{x:.1f}, y:{y:.1f}, z:{z:.2f}')

            except IndexError:
                self.get_logger().warn("Nose landmark not found.")
        else:
            self.get_logger().info("No pose landmarks detected.")
            return

        # Draw landmarks and publish compressed annotated image
        self.drawing.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            annotated_msg = CompressedImage()
            annotated_msg.format = 'jpeg'
            annotated_msg.data = np.array(buffer).tobytes()
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            self.annotated_pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
