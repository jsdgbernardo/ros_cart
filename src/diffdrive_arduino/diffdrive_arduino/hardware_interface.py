import serial
from time import time
from math import pi
from ros2_control_interfaces.hardware_interface import SystemInterface
from ros2_control_interfaces.hardware_interface import return_type
from ros2_control_interfaces.hardware_info import HardwareInfo
from rclpy.time import Time

class DiffDriveArduino(SystemInterface):
    def __init__(self):
        self.port = None
        self.target_left = 0.0
        self.target_right = 0.0
        self.actual_left = 0.0
        self.actual_right = 0.0

    def configure(self, info: HardwareInfo):
        self.port_name = info.hardware_parameters["port"]
        self.baudrate = int(info.hardware_parameters.get("baud", 115200))
        self.ticks_per_rev = float(info.hardware_parameters.get("ticks_per_rev", 20))
        self.wheel_radius = float(info.hardware_parameters.get("wheel_radius", 0.035))

        self.port = serial.Serial(self.port_name, self.baudrate, timeout=0.1)
        return return_type.OK

    def export_state_interfaces(self):
        return [
            self.register_state_interface("left_wheel_joint/velocity", lambda: self.actual_left),
            self.register_state_interface("right_wheel_joint/velocity", lambda: self.actual_right),
        ]

    def export_command_interfaces(self):
        return [
            self.register_command_interface("left_wheel_joint/velocity", lambda: self.target_left),
            self.register_command_interface("right_wheel_joint/velocity", lambda: self.target_right),
        ]

    def read(self, time: Time, duration: Time):
        if self.port.in_waiting:
            try:
                line = self.port.readline().decode().strip()
                if line.startswith("V"):
                    _, l, r = line.split()
                    # ticks/s → rad/s
                    self.actual_left = (2 * pi * float(l)) / self.ticks_per_rev
                    self.actual_right = (2 * pi * float(r)) / self.ticks_per_rev
            except:
                pass
        return return_type.OK

    def write(self, time: Time, duration: Time):
        # rad/s → ticks/s
        left_cmd = (self.target_left * self.ticks_per_rev) / (2 * pi)
        right_cmd = (self.target_right * self.ticks_per_rev) / (2 * pi)
        self.port.write(f"{left_cmd:.2f} {right_cmd:.2f}\n".encode())
        return return_type.OK
