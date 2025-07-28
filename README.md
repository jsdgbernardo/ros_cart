# `ros_cart`

## Current Setup
- ROS2 Humble (due to more tutorials online, and a much more stable version)
- CycloneDDS (apparently is better than the default, FastDDS)
- Colcon
- `./src/mediapipe_cam`: Edit IP address to match phone's IP
## Downloaded files
- ROS2 is better supported with **Linux**
  - SLAM
  - Nav2, Nav2 BRINGUP
  - Twist Mux
  - Xacro
  - MediaPipe
  - YOLOv8
  - Tf2, Tf2 Tools, Tf2 Geometry
  - Joint State Publisher (and GUI)
  - Robot State Publisher
- New Gazebo
- CycloneDDS
- Colcon
- Xrdp (for remote desktop connection)
- Pandas (for dataframes)
## Issues
- **Raspberry Pi limitations**: May cause transform delays due to low computing power.
- **Control system errors**:
  - Motors used need higher torque for small movements.
    - Use **gear motors** (NOT stepper motors), as steppers require more power and are not ideal for continuous motion (though still an option).
  - Calibration of encoder feedback may be inaccurate.
    - Prefer motors with built-in **encoders** for better precision.
    - Explore better pulse tracking methods.
  - **PID control** is ineffective due to low torque and poor calibration.
- Prediction node not tested due to control system errors
## Considerations for ADMU
### Database for Items
- **YOLO**: Use a custom dataset based on experimental items.
  - Other technologies can be considered, such as RFID barcodes on the items.
- **Object Interactions (e.g., grocery items held together)**:
  - Can be custom-collected over time.
  - Alternatively, look for an existing online dataset.
### Camera Setup
- Prefer **direct camera connection** to Raspberry Pi over Wi-Fi streaming for:
  - Reduced latency
  - Improved compatibility with YOLO and MediaPipe
- Ensure **camera-to-map transform** is properly set up (should be manageable with `tf` transforms).
- Consider how the robot will know which person is the user.
### CPU and Processing Strategy
- Raspberry Pi can:
  - Handle **sensor data reception**
  - Transmit data to a **laptop** for heavy processing
  - Receive instructions back (if Wi-Fi latency is acceptable)
- May not be suitable for **precise real-time feedback** (e.g., in path planning).
### Chassis Design
- **Differential drive (diffdrive)** is easier to implement.
-   Three- or four-wheel configurations are possible but more complex. It can still be considered if a reference code is found.
- **LIDAR** should remain as unobstructed as possible.
- Take into account the weight of all the components as well as the items.
### Cart–User Interactions
- Cart should detect whether the user is **intently approaching** or not.
- Cart should know if the user is looking or has the intent to approach the items
### Path Planning
- Nav2 does not natively support multi-goal path planning
  - Task manager node can be built on top of it that sends navigational goals one at a time to Nav2 (order optimized by TSP heuristic)
## Subsystems That Can Be Added
- **User-based machine learning**:
  - Adapts to the user’s **gait** to predict short-term pathway
  - Learns from **purchase history** to predict future actions
- **Advanced path planning**:
  - Optimized for **crowded environments**
  - Handles **dynamic obstacles** effectively, which is more apt for a public area
  - No longer Nav2 package
