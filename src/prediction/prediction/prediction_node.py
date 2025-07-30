import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped, Pose
from std_msgs.msg import String, Float32
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import math
import time
import numpy as np
import pandas as pd

class product():
    def __init__(self, name, probability, coordinates):
        self.name = name
        self.probability = probability
        self.coordinates = coordinates
        self.path = None

class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction_node')    
        
        self.items = [] # list of item objects

        # Corrected: removed distance argument from product instantiation
        self.items.append(product('bottled water', 1, Point(x=-1.1708265542984009, y=1.9727896451950073)))
        self.items.append(product('milk', 1, Point(x=3.6315736770629883, y=2.4937195777893066)))
        self.items.append(product('dessert', 1, Point(x=5.441387176513672, y=0.9294328689575195)))
        self.items.append(product('biscuit', 1, Point(x=3.8366289138793945, y=-1.176401138305664)))
        self.items.append(product('tissue roll', 1, Point(x=0.43349552154541016, y=0.72157883644104)))

        # Defining item relations (co-occurrence score)
        score = {
            'bottled water': {'milk': 0.1178, 'dessert': 0.0176, 'biscuit': 0.0474, 'tissue roll': 0.0033},
            'milk': {'bottled water': 0.0453, 'dessert': 0.0152, 'biscuit': 0.041, 'tissue roll': 0.0008},
            'dessert': {'bottled water': 0.0453, 'milk': 0.102, 'biscuit': 0.034, 'tissue roll': 0.000000001},
            'biscuit': {'bottled water': 0.0556, 'milk': 0.1253, 'dessert': 0.0155, 'tissue roll': 0.0026},
            'tissue roll': {'bottled water': 0.10, 'milk': 0.0667, 'dessert': 0.000000001, 'biscuit': 0.0667}
        }

        # Converting score data to a DataFrame
        self.df = pd.DataFrame.from_dict(score, orient='index')
        np.fill_diagonal(self.df.values, 0)

        self.held_items = [] # list of held item objects

        # Subscription to YOLOv8 subsystem
        self.create_subscription(
            String,
            'detected_item',
            self.held_items_callback,
            10
        )

        # Service client for ComputePathToPose
        self._action_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

        # Subscription to AMCL pose (robot)
        self.current_pose = None
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        # Subscription to user pose
        self.user_pose = None
        self.last_user_pose = None
        self.create_subscription(
            PoseStamped,
            '/user/pose',
            self.user_pose_callback,
            10
        )

        # Subscription to MediaPipe subsystem
        self.head_yaw = None
        self.create_subscription(
            Float32,
            '/pose/head_yaw_angle',
            self.head_yaw_callback,
            10
        )

        # Latest path storage
        self.latest_path = None
        self.latest_path_goal = None  # To track which goal this path corresponds to

        self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )
        
        # # Publish navigational goal
        # self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Publish path to the item with the highest probability
        self.path_pub = self.create_publisher(Path, 'best_path', 10)

        # Create a timer to periodically compute probabilities
        self.timer = self.create_timer(1.0, self.compute_probability)

         # Define manual paths (waypoints) for each item - temporarily starts at (0,0)
        self.paths = {
            'bottled water': [
                (0.0, 0.0),
                (-1.17, 1.97)  
            ],
            'milk': [
                (0.0, 0.0),
                # (-0.5, 1.8),
                (3.63, 2.49)
            ],
            'dessert': [
                (0.0, 0.0),
                # (4.91, 0.0),
                (5.44, 0.93)
            ],
            'biscuit': [
                (0.0, 0.0),
                (3.83, -1.17)
            ],
            'tissue roll': [
                (0.0, 0.0),
                (0.4, 0.7)
            ]
        }

    def create_manual_path_with_user_start(self, points, frame='map'):

        path = Path()
        path.header.frame_id = frame
        path.header.stamp = self.get_clock().now().to_msg()

        
        counter = 0

        # Determine user start position
        if self.user_pose is not None and counter < 5:
            user_x = self.user_pose.position.x
            user_y = self.user_pose.position.y
            counter = 0
            self.get_logger().info(f'User pose used: {user_x}, {user_y}')
        elif self.current_pose is not None:
            user_x = self.current_pose.position.x
            user_y = self.current_pose.position.y
            self.get_logger().info(f'Current pose used: {user_x}, {user_y}')
        else:
            user_x, user_y = points[0]  # fallback to first point if user_pose unknown

        if self.user_pose == self.last_user_pose: 
            counter += 1

        # Insert user position as first waypoint
        first_point = (user_x, user_y)
        waypoints = [first_point] + points[1:]  # replace original first waypoint with user position

        for x, y in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def compute_probability(self):

        # Shift first waypoint to user position or cart position
        for item in self.items:
            if item.name in self.paths:
                item.path = self.create_manual_path_with_user_start(self.paths[item.name])

        self.get_logger().info('Computing probabilities...')

        total_length = 0.0
        priors = {}
        likelihoods = {}

        # valid_items = []

        # # Filter items with valid paths
        # for item in self.items:
        #     if item.path is not None and len(item.path.poses) > 1:
        #         valid_items.append(item)

        path_lengths = [self.calculate_path_length(item.path) for item in self.items]
        max_length = max(path_lengths)
        min_length = min(path_lengths)

        # Compute priors and likelihoods
        for item, length in zip(self.items, path_lengths):
            priors[item.name] = item.probability
            held_prob = self.held_items_likelihood(item)
            path_prob = 1.0 - (length - min_length) / (max_length - min_length + 1e-6)
            gaze_prob = self.compute_gaze_alignment_score_with_path(item) 
            gaze_prob = 1 if gaze_prob == 0 else gaze_prob

            likelihoods[item.name] = held_prob * path_prob * gaze_prob
            self.get_logger().info(f'Item: {item.name}')
            self.get_logger().info(f'Held Likelihood: {held_prob:.4f}')
            self.get_logger().info(f'Path Likelihood: {path_prob:.4f}')
            self.get_logger().info(f'Gaze Likelihood: {gaze_prob:.4f}')

        # Normalize likelihoods
        evidence = sum(likelihoods[name] * priors[name] for name in likelihoods)

        # Compute posterior probabilities
        for item in self.items:
            numerator = likelihoods[item.name] * priors[item.name]
            posterior = numerator / evidence if evidence > 0 else 0.0
            item.probability = posterior
            if item in self.held_items:
                item.probability *= 0.5

        self.get_logger().info('Probabilities computed.')

        # Find the item with the highest probability
        if self.items:
            best_item = max(self.items, key=lambda x: x.probability)
            best_path = best_item.path

            # Publish the path
            nav_path = Path()
            nav_path.header.frame_id = 'map'
            nav_path.header.stamp = self.get_clock().now().to_msg()
            nav_path.poses = best_path.poses
            self.path_pub.publish(nav_path)

            if nav_path.poses:
                goal_msg = ComputePathToPose.Goal()
                goal_msg.goal = nav_path.poses[-1]  # last point in path
                goal_msg.goal.header.frame_id = 'map'
                goal_msg.goal.header.stamp = self.get_clock().now().to_msg()

                self._action_client.send_goal_async(goal_msg)

            self.get_logger().info(f'Published best path to item: {best_item.name} with probability: {best_item.probability:.4f}')
        else:
            self.get_logger().warn("No valid paths to any items. Nothing published.")

    def get_relation(self, item1, item2):
        try:
            return self.df.loc[item1, item2]
        except KeyError:
            return 0
    
    def held_items_callback(self, msg):
        # Add item to the list
        name = msg.data
        if name not in self.held_items:
            self.held_items.append(name)
        self.get_logger().info(f'Held items updated: {self.held_items}')

    def held_items_likelihood(self, item, alpha=1.0):
        # Compute the likelihood of the held items given the item
        # self.get_logger().info(f'Computing held items likelihood for {item.name}...')
        if len(self.held_items) == 0:
            return 1.0

        if item.name in self.held_items:
            return 0.0

        scores = []
        for held_item in self.held_items:
            numerator = self.get_relation(item.name, held_item)
            denominator = sum(
                self.get_relation(item.name, other_item.name) 
                for other_item in self.items 
                if other_item.name != held_item and other_item.name != item.name
            )
            if denominator == 0:
                continue
            smoothed = (numerator + alpha) / (denominator + alpha * len(self.items))
            scores.append(smoothed)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.get_logger().info(f'Current pose updated: {self.current_pose.position.x}, {self.current_pose.position.y}')

    def user_pose_callback(self, msg):
        self.user_pose = msg.pose
        # self.get_logger().info(f'User pose updated: {self.user_pose.position.x}, {self.user_pose.position.y}')

    def path_callback(self, msg):
        # Save latest path and goal position for matching
        # self.get_logger().info(f'Latest path received with {len(msg.poses)} poses.')
        self.latest_path = msg
        if len(msg.poses) > 0:
            last_pose = msg.poses[-1].pose.position
            self.latest_path_goal = (last_pose.x, last_pose.y)
        else:
            self.latest_path_goal = None
    
    def calculate_path_length(self, path):
        total = 0.0
        poses = path.poses
        for i in range(1, len(poses)):
            dx = poses[i].pose.position.x - poses[i-1].pose.position.x
            dy = poses[i].pose.position.y - poses[i-1].pose.position.y
            total += math.hypot(dx, dy)
        return total

    def head_yaw_callback(self, msg):
        self.head_yaw = msg.data
        # self.get_logger().info(f'Head yaw updated: {self.head_yaw}')

    def compute_gaze_alignment_score_with_path(self, item, num_waypoints=3, decay_rate=0.7):
        # Return 0 if necessary data is missing
        if self.head_yaw is None or self.user_pose is None or not hasattr(item, 'path') or item.path is None:
            return 0.0

        # User position
        ux, uy = self.user_pose.position.x, self.user_pose.position.y
        user_pos = np.array([ux, uy])

        # Get up to N waypoints
        path_points = item.path.poses
        if len(path_points) < num_waypoints:
            num_waypoints = len(path_points)
        if num_waypoints == 0:
            return 0.0

        # Weighted average direction vector
        weighted_sum = np.zeros(2)
        total_weight = 0.0

        for i in range(num_waypoints):
            px = path_points[i].pose.position.x
            py = path_points[i].pose.position.y
            vector = np.array([px, py]) - user_pos
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue

            unit_vector = vector / norm
            weight = decay_rate ** i  # Exponential decay weight
            weighted_sum += unit_vector * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        avg_direction = weighted_sum / total_weight
        avg_direction /= np.linalg.norm(avg_direction)  # normalize

        # Gaze vector from head yaw (assuming yaw in degrees)
        yaw_rad = math.radians(self.head_yaw)
        gaze_vector = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])

        # Compute alignment (dot product)
        dot = np.dot(gaze_vector, avg_direction)
        score = max(0.0, dot)  # Clamp to [0, 1]

        # self.get_logger().info(
        #     f"[Gaze Alignment] Gaze: {gaze_vector}, AvgPath: {avg_direction}, Score: {score:.2f}"
        # )

        return score


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
