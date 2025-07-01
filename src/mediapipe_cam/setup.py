from setuptools import setup

package_name = 'mediapipe_cam'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS 2 node to stream from phone camera using IP Webcam for MediaPipe input',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'phone_cam_node = mediapipe_cam.phone_cam_node:main',
        ],
    },
)
