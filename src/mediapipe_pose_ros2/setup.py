from setuptools import setup

package_name = 'mediapipe_pose_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jsdgbernardo',
    maintainer_email='bernardojonathansimon@gmail.com',
    description='ROS 2 node that integrates MediaPipe Pose estimation.',
    entry_points={
        'console_scripts': [
            'pose_node = mediapipe_pose_ros2.pose_node:main',
        ],
    },
)
