from setuptools import setup
import os

package_name = 'mediapipe_cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jsdgbernardo',
    maintainer_email='bernardojonathansimon@gmail.com',
    description='Publishes camera data from phone',
    license='License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_node = mediapipe_cam.pose_node:main',
        ],
    },
)
