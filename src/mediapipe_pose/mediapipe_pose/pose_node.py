import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Float32
import cv2
import mediapipe as mp
import numpy as np
import math

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

        # Publishers
        self.nose_pub = self.create_publisher(Point, '/pose/nose_point', 10)
        self.yaw_pub = self.create_publisher(Float32, '/pose/head_yaw_angle', 10)
        self.body_pub = self.create_publisher(Float32, '/pose/body_angle', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/user/pose', 10)
        self.annotated_pub = self.create_publisher(CompressedImage, '/pose/image/compressed', 10)

        # MediaPipe pose detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils

        self.last_nose = None  # Previous nose position for smoothing

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def image_callback(self, msg):
        # Decode the compressed JPEG to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode JPEG image.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            self.get_logger().info("No pose landmarks detected.")
            return

        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark

        try:
            # --- Nose Tracking ---
            nose = lm[0]
            x, y, z = nose.x * w, nose.y * h, nose.z

            # Smoothing nose coordinates
            if self.last_nose is not None:
                alpha = 0.5
                x = alpha * x + (1 - alpha) * self.last_nose.x
                y = alpha * y + (1 - alpha) * self.last_nose.y
                z = alpha * z + (1 - alpha) * self.last_nose.z

            point = Point(x=float(x), y=float(y), z=float(z))
            self.nose_pub.publish(point)
            self.last_nose = point

            self.get_logger().info(f'Nose @ x:{x:.1f}, y:{y:.1f}, z:{z:.2f}')

            # --- Head Yaw Estimation (Eyes) ---
            left_eye = lm[2]
            right_eye = lm[5]
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            yaw = np.arctan2(dy, dx) * 180.0 / np.pi
            self.yaw_pub.publish(Float32(data=yaw))
            self.get_logger().info(f'Head Yaw: {yaw:.1f}°')

            # --- Body Orientation (Shoulders) ---
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            sdx = right_shoulder.x - left_shoulder.x
            sdy = right_shoulder.y - left_shoulder.y
            body_angle = np.arctan2(sdy, sdx) * 180.0 / np.pi
            self.body_pub.publish(Float32(data=body_angle))
            self.get_logger().info(f'Body Angle: {body_angle:.1f}°')

            # --- Simulated /user/pose using body angle ---
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'  # or 'odom' if that's your sim frame

            # # Simulated position; ----------------------------- update later with dynamic data   ------------------------------
            # pose_msg.pose.position.x = 2.0
            # pose_msg.pose.position.y = 3.0
            # pose_msg.pose.position.z = 0.0

            # Convert body angle to quaternion
            body_angle_rad = np.deg2rad(body_angle)
            quat = self.yaw_to_quaternion(body_angle_rad)
            pose_msg.pose.orientation = quat

            # self.pose_pub.publish(pose_msg)
            # self.get_logger().info(f'Published /user/pose at (2.0, 3.0), yaw: {body_angle:.1f}°')

        except IndexError:
            self.get_logger().warn("Required landmarks not found.")
            return

        # Draw landmarks and publish annotated image
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
