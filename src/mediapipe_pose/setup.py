from setuptools import setup

package_name = 'mediapipe_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jssgbernardo',
    maintainer_email='bernardojonathansimon@gmail.com',
    description='MediaPipe-based pose tracker for ROS 2',
    license='License',
    entry_points={
        'console_scripts': [
            'pose_node = mediapipe_pose.pose_node:main',
        ],
    },
)
