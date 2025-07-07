import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import xacro

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_ros2_control = LaunchConfiguration('use_ros2_control')

    pkg_path = get_package_share_directory('chassis')
    xacro_file = os.path.join(pkg_path, 'description', 'chassis.urdf.xacro')

    robot_description_config = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        xacro_file, ' ',
        'use_ros2_control:=', use_ros2_control, ' ',
        'sim_mode:=', use_sim_time
    ])

    params = {'robot_description': robot_description_config, 'use_sim_time': use_sim_time}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'),
        DeclareLaunchArgument(
            'use_ros2_control',
            default_value='true',
            description='Enable ros2_control'),
        node_robot_state_publisher
    ])
