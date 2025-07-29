from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_costmap_2d',
            executable='costmap_2d',
            name='global_costmap',
            output='screen',
            parameters=['src/chassis/config/minimal_nav2.yaml'],
            remappings=[('cmd_vel', 'cmd_vel')]
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=['src/chassis/config/minimal_nav2.yaml']
        ),
    ])
