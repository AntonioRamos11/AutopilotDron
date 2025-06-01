# System Integration Guide

This document explains how all the components of the high-speed, accurate point-to-point drone autopilot system work together.

## System Architecture

![System Architecture Diagram](architecture.png)

### Components Overview

1. **PX4 Flight Controller**:
   - Handles low-level flight control and state estimation
   - Implements failsafe mechanisms
   - Interfaces with sensors and actuators
   - Configured with optimized parameters for high-speed flight

2. **RTK GPS System**:
   - Base station provides corrections
   - Rover (on drone) achieves cm-level positioning
   - Provides precise position and velocity inputs to PX4 EKF

3. **Trajectory Generation System**:
   - Implemented in the `min_snap_trajectory.py` module
   - Computes optimal smooth trajectories between waypoints
   - Minimizes snap (4th derivative) for smooth, efficient motion
   - Respects drone dynamic constraints

4. **Companion Computer with ROS 2**:
   - Runs the `trajectory_tracker.py` node
   - Bridges between high-level planning and low-level control
   - Monitors trajectory execution and position error
   - Provides visualization and telemetry

5. **Simulation Environment**:
   - Gazebo-based testing platform
   - Simulates RTK GPS for development without hardware
   - Validates trajectories before field testing

## Data Flow

1. **Mission Planning**:
   - Waypoints are defined (manually or by mission planner)
   - Sent to companion computer via `/trajectory_tracking/waypoints` topic

2. **Trajectory Generation**:
   - Companion computer generates minimum snap trajectory
   - Optimizes for smooth transitions between waypoints
   - Respects velocity and acceleration constraints

3. **Trajectory Execution**:
   - Trajectory is sampled at high frequency (50Hz)
   - Position/velocity setpoints sent to PX4 via MAVLink (MAVROS)
   - PX4 position controller tracks these setpoints

4. **Feedback Loop**:
   - RTK GPS provides cm-level position feedback
   - PX4 EKF2 fuses GPS, IMU, and other sensors
   - Position error is monitored and logged
   - Companion computer can adjust trajectory if needed

## Configuration Steps

### 1. Flight Controller Setup

1. Flash PX4 firmware to the flight controller
2. Set the parameters as defined in `firmware/px4_params.md`
3. Calibrate sensors following PX4 calibration procedures
4. Configure RTK GPS as per `rtk_setup/rtk_installation.md`

### 2. Companion Computer Setup

1. Install ROS 2 Humble on the companion computer
2. Set up the workspace:
   ```bash
   mkdir -p ~/drone_ws/src
   cd ~/drone_ws/src
   # Copy the companion_software and path_planning directories
   cd ..
   colcon build
   ```

3. Install dependencies:
   ```bash
   sudo apt install ros-humble-mavros ros-humble-mavros-extras
   pip install numpy scipy matplotlib
   ```

### 3. System Integration

1. Connect the companion computer to the flight controller via UART or USB
2. Configure MAVROS:
   ```bash
   # Edit ~/drone_ws/src/mavros_config/config/px4_config.yaml
   # Set appropriate serial device and baudrate
   ```

3. Launch the system:
   ```bash
   # Terminal 1: Start MAVROS
   ros2 launch mavros px4.launch.py
   
   # Terminal 2: Start trajectory tracker
   ros2 run autopilot_drone trajectory_tracker.py
   ```

## Operation Guide

### Starting a Mission

1. Place drone in takeoff position
2. Ensure RTK fix is acquired (check QGroundControl)
3. Send waypoints to the system:
   ```bash
   # Format: [start_vx, start_vy, start_vz, wp1_x, wp1_y, wp1_z, wp2_x, wp2_y, wp2_z, ..., end_vx, end_vy, end_vz]
   ros2 topic pub -1 /trajectory_tracking/waypoints std_msgs/msg/Float32MultiArray "data: [0,0,0, 0,0,2, 10,10,5, 20,0,10, 0,0,0]"
   ```

4. Start mission execution:
   ```bash
   ros2 topic pub -1 /trajectory_tracking/command std_msgs/msg/String "data: 'start'"
   ```

### Emergency Procedures

1. **Stop trajectory execution**:
   ```bash
   ros2 topic pub -1 /trajectory_tracking/command std_msgs/msg/String "data: 'stop'"
   ```

2. **Switch to manual control**:
   - Flip the mode switch on the remote control to POSITION or ALTITUDE mode
   - This overrides the offboard control

## Monitoring and Debugging

1. **Trajectory visualization**:
   ```bash
   ros2 run rviz2 rviz2 -d ~/drone_ws/src/autopilot_drone/config/trajectory_view.rviz
   ```

2. **Position error monitoring**:
   ```bash
   ros2 run rqt_plot rqt_plot /trajectory_tracking/position_error/data[0]:data[1]:data[2]
   ```

3. **Logging**:
   ```bash
   # Record all relevant topics
   ros2 bag record -o flight_data /trajectory_tracking/position_error /mavros/local_position/pose /trajectory_tracking/visualization
   ```

## Performance Tuning

### Increasing Speed

1. Gradually adjust the trajectory speed parameter:
   ```bash
   ros2 topic pub -1 /trajectory_tracking/command std_msgs/msg/String "data: 'speed 15'"
   ```

2. Update PX4 parameters for higher velocity:
   - Increase `MPC_XY_VEL_MAX` incrementally
   - Monitor position tracking error

### Improving Accuracy

1. Ensure RTK GPS has a clear view of the sky
2. Wait for RTK fixed solution (not just float)
3. Increase GPS update rate if supported by hardware
4. Fine-tune EKF2 parameters based on log analysis

## Common Issues and Solutions

### Poor Trajectory Tracking

**Symptoms**: Large position errors, oscillations
**Solutions**:
- Reduce maximum speed and acceleration
- Tune PX4 position controller gains
- Check for GPS signal interference
- Verify RTK corrections are being received

### RTK Fix Issues

**Symptoms**: No RTK fixed solution, high position uncertainty
**Solutions**:
- Ensure base station has clear sky view
- Check radio link between base and rover
- Verify RTCM3 message configuration
- Reset and re-survey base station

### Companion Computer Communication Problems

**Symptoms**: MAVROS connection drops, delayed commands
**Solutions**:
- Check physical connection (UART/USB)
- Increase serial baudrate if possible
- Reduce CPU load on companion computer
- Monitor ROS 2 message latency