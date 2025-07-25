import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped
from std_msgs.msg import String, Float32
import math
import time
import numpy as np
import pandas as pd

class product():
    def __init__(self, name, probability, coordinates):
        self.name = name
        self.probability = probability
        self.coordinates = coordinates

class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction_node')    
        
        self.items = [] # list of item objects

        self.items.append(product('bottled water', 1, Point(x=-0.9238461256027222, y=2.690908432006836)))
        self.items.append(product('milk', 1, Point(x=3.6315736770629883, y=2.4937195777893066)))
        self.items.append(product('dessert', 1, Point(x=5.441387176513672, y=0.9294328689575195)))
        self.items.append(product('biscuit', 1, Point(x=3.8366289138793945, y=-1.176401138305664)))
        self.items.append(product('tissue roll', 1, Point(x=0.43349552154541016, y=0.72157883644104)))

        # Defining item relations (co-occurence score)
        score = {
            'bottled water': {'milk': 0.1178, 'dessert': 0.0176, 'biscuit': 0.0474, 'tissue roll': 0.0033},
            'milk': {'bottled water': 0.0453, 'dessert': 0.0152, 'biscuit': 0.041, 'tissue roll': 0.0008},
            'dessert': {'bottled water': 0.0453, 'milk': 0.102, 'biscuit': 0.034, 'tissue roll': 0},
            'biscuit': {'bottled water': 0.0556, 'milk': 0.1253, 'dessert': 0.0155, 'tissue roll': 0.0026},
            'tissue roll': {'bottled water': 0.10, 'milk': 0.0667, 'dessert': 0, 'biscuit': 0.0667}
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

        # Subscription to AMCL pose
        self.current_pose = None
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
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
        
        # Publish navigational goal
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Create a timer to periodically compute probabilities
        self.timer = self.create_timer(1.0, self.compute_probability)

    def compute_probability(self):

        self.get_logger().info('Computing probabilities...')

        path_lengths = {}
        total_length = 0.0
        paths = {}
        priors = {}
        likelihoods = {}

        for item in self.items:
            path, length = self.compute_path_to(item.coordinates.x, item.coordinates.y)
            paths[item.name] = path
            path_lengths[item.name] = length
            total_length += length

        for item in self.items:
            priors[item.name] = item.probability
            held_prob = self.held_items_likelihood(item)
            path_prob = path_lengths[item.name] / total_length if total_length > 0 else 1.0
            gaze_prob = self.compute_gaze_alignment_score_with_path(item)

            likelihoods[item.name] = held_prob * path_prob * gaze_prob
        
        evidence = sum(likelihoods[name] * priors[name] for name in likelihoods)

        for item in self.items:
            numerator = likelihoods[item.name] * priors[item.name]
            posterior = numerator / evidence if evidence > 0 else 0.0
            item.probability = posterior

        best_item = max(self.items, key=lambda x: x.probability)

        goal_msg = ComputePathToPose.Goal()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position = best_item.coordinates
        goal_pose.pose.orientation.w = 1.0

        goal_msg.goal = goal_pose
        goal_msg.planner_id = ''
        goal_msg.use_start = False

        # Wait for the action server to be ready
        self.get_logger().info('Waiting for ComputePathToPose action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('ComputePathToPose action server is ready.')

        # Send goal and attach callbacks
        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)
        self.get_logger().info(f'Best item: {best_item.name} with probability: {best_item.probability:.4f}')

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

    # Compute the path length to a given (x, y) coordinate
    def compute_path_to(self, x, y):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position = Point(x=x, y=y, z=0.0)
        goal_pose.pose.orientation.w = 1.0

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = goal_pose
        
        if self.current_pose is not None:
            goal_msg.start.header.frame_id = 'map'
            goal_msg.start.pose = self.current_pose
        else:
            self.get_logger().warn("Current robot pose not available.")
            return None, float('inf')

        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.result() is not None:
            path = future.result().path
            length = self.calculate_path_length(path)
            return path, length
        else:
            self.get_logger().warn('Nav2 path service failed or timed out.')
            return None, float('inf')

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
        self.get_logger().info(f'Head yaw updated: {self.head_yaw}')

    def compute_gaze_alignment_score_with_path(self, item):
        if self.head_yaw is None or self.current_pose is None:
            return 0.0

        path, _ = self.compute_path_to(item.coordinates.x, item.coordinates.y)
        if path is None or len(path.poses) < 2:
            return 0.0

        poses = path.poses
        dx = poses[1].pose.position.x - poses[0].pose.position.x
        dy = poses[1].pose.position.y - poses[0].pose.position.y
        path_vector = np.array([dx, dy])
        norm = np.linalg.norm(path_vector)
        if norm == 0:
            return 0.0
        path_vector = path_vector / norm

        yaw_rad = math.radians(self.head_yaw)
        gaze_vector = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])

        dot = np.dot(gaze_vector, path_vector)
        return max(0.0, dot)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
