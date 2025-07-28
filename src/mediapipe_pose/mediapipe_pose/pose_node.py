import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point, PoseStamped, Quaternion, PointStamped
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
import tf2_geometry_msgs
import tf2_ros
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

        # Subscription to LiDAR scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.latest_scan = None

        # TF buffer and listender
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_h_fov_deg = 78.0
        self.camera_width = 640
        self.camera_height = 480

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

    def scan_callback(self, msg):
        self.latest_scan = msg
    
    def image_callback(self, msg):

        # msg.header.frame_id = 'camera_optical_link'

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
            eye_center_x = (left_eye.x + right_eye.x) / 2.0
            offset = nose.x - eye_center_x
            eye_distance = abs(right_eye.x - left_eye.x)
            if eye_distance > 0.01:
                normalized_offset = offset / eye_distance
            else:
                normalized_offset = 0.0
            normalized_offset = max(min(normalized_offset, 2.0), -2.0)
            yaw = -normalized_offset * 90.0
            yaw = ((yaw + 180) % 360) - 180 # Clamp to [-180, 180]
            self.yaw_pub.publish(Float32(data=yaw))
            self.get_logger().info(f'Head Yaw: {yaw:.1f}°')

            # --- Body Orientation (Shoulders) ---
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            sdx = left_shoulder.x - right_shoulder.x
            sdy = left_shoulder.y - right_shoulder.y
            body_angle = np.arctan2(sdy, sdx) * 180.0 / np.pi
            self.body_pub.publish(Float32(data=body_angle))
            self.get_logger().info(f'Body Angle: {body_angle:.1f}°')

            # ----- Estimate user position ------
            nose_angle_camera = ((x / self.camera_width) - 0.5) * math.radians(self.camera_h_fov_deg)
            nose_angle_robot = nose_angle_camera + math.pi

            # Normalize the angle to [-pi, pi]
            if nose_angle_robot > math.pi:
                nose_angle_robot -= 2 * math.pi
            elif nose_angle_robot < -math.pi:
                nose_angle_robot += 2 * math.pi

            scan = self.latest_scan
            if scan is None:
                self.get_logger().warn("No LaserScan data received yet.")
                return

            angle_min = scan.angle_min
            angle_max = scan.angle_max
            angle_increment = scan.angle_increment

            if nose_angle_robot < angle_min or nose_angle_robot > angle_max:
                self.get_logger().warn(f"Nose angle {math.degrees(nose_angle_robot):.1f}° out of LiDAR range")
                return

            index = int((nose_angle_robot - angle_min) / angle_increment)
            if index < 0 or index >= len(scan.ranges):
                self.get_logger().warn("LiDAR index out of range")
                return

            window = 5
            best_depth = None
            for offset in range(-window, window + 1):
                i = index + offset
                if 0 <= i < len(scan.ranges):
                    d = scan.ranges[i]
                    if not math.isinf(d) and not math.isnan(d) and d > 0.0:
                        best_depth = d
                        break

            if best_depth is None:
                self.get_logger().warn(f"No valid LiDAR depth near index {index} for angle {math.degrees(nose_angle_robot):.1f}°")
                return
            depth = best_depth

            # Calculate focal length in pixels
            fx = (self.camera_width / 2) / math.tan(math.radians(self.camera_h_fov_deg) / 2)
            fy = fx  # assume square pixels

            cx = self.camera_width / 2
            cy = self.camera_height / 2

            # Project pixel to 3D point in camera frame
            X_cam = depth
            Y_cam = (x - cx) * depth / fx
            Z_cam = (y - cy) * depth / fy

            person_camera = PointStamped()
            person_camera.header.frame_id = 'camera_link'  # camera frame
            person_camera.header.stamp = self.get_clock().now().to_msg()
            person_camera.point.x = X_cam
            person_camera.point.y = Y_cam
            person_camera.point.z = Z_cam

            # Transform to map frame
            try:
                person_map = self.tf_buffer.transform(person_camera, 'map', timeout=rclpy.duration.Duration(seconds=1.0))
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                return

            # Publish user pose with body orientation
            pose_msg = PoseStamped()
            pose_msg.header.stamp = person_map.header.stamp
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position = person_map.point
            body_angle_rad = np.deg2rad(body_angle)
            pose_msg.pose.orientation = self.yaw_to_quaternion(body_angle_rad)
            self.pose_pub.publish(pose_msg)

            self.get_logger().info(f'User Position: x={person_map.point.x:.2f}, y={person_map.point.y:.2f}')


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