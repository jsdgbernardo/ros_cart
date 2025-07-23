import rclpy
from rclpy.node import Node
from nav2_msgs.srv import ComputePathToPose
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

        self.items.append(product('Bottled Water', 1, Point(x=1.0, y=2.0, z=0.0)))
        self.items.append(product('Milk', 1, Point(x=2.0, y=3.0, z=0.0)))
        self.items.append(product('Dessert', 1, Point(x=3.0, y=4.0, z=0.0)))
        self.items.append(product('Biscuits', 1, Point(x=4.0, y=5.0, z=0.0)))
        self.items.append(product('Tissue', 1, Point(x=5.0, y=6.0, z=0.0)))

        # Defining item relations (co-occurence score)
        score = {
            'Bottled Water': {'Milk': 0.1178, 'Dessert': 0, 'Biscuits': 0, 'Tissue': 0},
            'Milk': {'Bottled Water': 0.1178, 'Dessert': 0, 'Biscuits': 0, 'Tissue': 0},
            'Dessert': {'Bottled Water': 0, 'Milk': 0, 'Biscuits': 0, 'Tissue': 0},
            'Biscuits': {'Bottled Water': 0, 'Milk': 0, 'Dessert': 0, 'Tissue': 0},
            'Tissue': {'Bottled Water': 0, 'Milk': 0, 'Dessert': 0, 'Biscuits': 0}
        }


        # Converting score data to a DataFrame
        self.df = pd.DataFrame.from_dict(score, orient='index')
        np.fill_diagonal(self.df.values, 0)

        self.held_items = [] # list of held item objects
        
        # Subscription to YOLOv8 subsystem
        self.create_subscription(
            String,
            'held_items',
            self.held_items_callback,
            10
        )

        # Service client for ComputePathToPose
        self.client = self.create_client(ComputePathToPose, '/compute_path_to_pose')

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

        # Create a timer to periodically compute probabilities
        self.timer = self.create_timer(1.0, self.compute_probability)
        
        # Publish navigational goal

    def compute_probability(self):

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

    def held_items_likelihood(self, item, alpha=1.0):
        # Compute the likelihood of the held items given the item
        if len(self.held_items) == 0:
            return 1.0

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

    # Compute the path length to a given (x, y) coordinate
    def compute_path_to(self, x, y):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position = Point(x=x, y=y, z=0.0)
        goal_pose.pose.orientation.w = 1.0

        req = ComputePathToPose.Request()
        req.goal = goal_pose
        
        if self.current_pose is not None:
            req.start.header.frame_id = 'map'
            req.start.pose = self.current_pose
        else:
            self.get_logger().warn("Current robot pose not available.")
            return None, float('inf')

        future = self.client.call_async(req)
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
