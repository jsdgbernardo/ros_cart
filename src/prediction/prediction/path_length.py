import rclpy
from rclpy.node import Node
from nav2_msgs.serv import ComputePaththToPose
from geometry_msgs.msg import PoseStamped, Point
import math
import time

class PathLengthCalculator(Node):
    def __init__(self):
        super().__init__('path_length_calculator')

        # Create a service client for ComputePathToPose
        self.client = self.create_client(ComputePathToPose, 'compute_path_to_pose')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for ComputePathToPose service...')

        # # Example start and goal poses
        # self.start_pose = PoseStamped()
        # self.start_pose.header.frame_id = 'map'
        # self.start_pose.pose.position = Point(x=0.0, y=0.0, z=0.0)

        # self.goal_pose = PoseStamped()
        # self.goal_pose.header.frame_id = 'map'
        # self.goal_pose.pose.position = Point(x=5.0, y=5.0, z=0.0)

    def compute_path_length(self):
        request = ComputePathToPose.Request()
        request.start = self.start_pose
        request.goal = self.goal_pose

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            path_length = 0.0
            path = future.result().path
            for i in range(len(path.poses) - 1):
                p1 = path.poses[i].pose.position
                p2 = path.poses[i + 1].pose.position
                distance = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
                path_length += distance

            return path_length
        else:
            self.get_logger().error('Failed to compute path length.')
            return None