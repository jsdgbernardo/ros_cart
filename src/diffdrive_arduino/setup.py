from setuptools import find_packages, setup

package_name = 'diffdrive_arduino'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','pyserial'],
    zip_safe=True,
    maintainer='jonathanb',
    maintainer_email='bernardojonathansimon@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
        'ros2_control.hardware_interfaces': [
            'hardware_interface = diffdrive_arduino.hardware_interface:DiffDriveArduino',
        ],
    },
)
