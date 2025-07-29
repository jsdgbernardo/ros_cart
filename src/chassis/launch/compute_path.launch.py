from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_planner',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'autostart': True},
                {'node_names': ['planner_server', 'bt_navigator', 'global_costmap']}
            ]
        )
    ])
