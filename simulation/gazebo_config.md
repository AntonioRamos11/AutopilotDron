# Gazebo Simulation Setup for High-Speed Drone Testing

This document outlines the setup and configuration of a Gazebo simulation environment for testing high-speed, accurate point-to-point drone navigation.

## Prerequisites

- ROS 2 Humble or newer
- Gazebo Garden or newer
- PX4-SITL (Software-In-The-Loop)
- QGroundControl

## Installation

### 1. Install PX4 SITL with Gazebo

```bash
# Clone PX4-Autopilot
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot

# Install dependencies
bash ./Tools/setup/ubuntu.sh

# Build for simulation
DONT_RUN=1 make px4_sitl gazebo
```

### 2. Install ROS 2 and required packages

```bash
# Install ROS 2 (if not already installed)
# Follow instructions at https://docs.ros.org/en/humble/Installation.html

# Install additional packages
sudo apt install ros-humble-mavros ros-humble-mavros-extras
```

### 3. Set up workspace

```bash
# Create ROS 2 workspace
mkdir -p ~/px4_ros_com_ws/src
cd ~/px4_ros_com_ws/src

# Clone required packages
git clone https://github.com/PX4/px4_ros_com.git
git clone https://github.com/PX4/px4_msgs.git

# Build the workspace
cd ..
colcon build
```

## RTK-Enabled Drone Model

To simulate RTK GPS capabilities, create a custom drone model based on the standard iris model with the following modifications:

```xml
<!-- Example modifications to iris.sdf for RTK GPS -->
<model name="iris_rtk">
  <!-- Standard iris model components -->
  
  <!-- Add RTK GPS module -->
  <include>
    <uri>model://rtk_gps</uri>
    <pose>0 0 0 0 0 0</pose>
    <name>rtk_gps</name>
  </include>
  
  <!-- Connect the GPS to the autopilot -->
  <joint name="rtk_gps_joint" type="fixed">
    <parent>iris::base_link</parent>
    <child>rtk_gps::link</child>
  </joint>
</model>
```

### Create Custom World

Create a custom Gazebo world with features appropriate for testing high-speed navigation:

```xml
<!-- high_speed_test.world -->
<sdf version="1.6">
  <world name="high_speed_test">
    <!-- Physics settings optimized for high-speed motion -->
    <physics type="ode">
      <real_time_update_rate>1000</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
          <sor>1.3</sor>
        </solver>
      </ode>
    </physics>

    <!-- Basic scene -->
    <include>
      <uri>model://sun</uri>
    </include>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- RTK base station -->
    <include>
      <uri>model://rtk_base_station</uri>
      <pose>0 0 0.1 0 0 0</pose>
      <name>rtk_base</name>
    </include>
    
    <!-- Add waypoint markers -->
    <model name="waypoint1">
      <pose>10 10 2 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere><radius>0.5</radius></sphere>
          </geometry>
          <material>
            <ambient>1 0 0 0.5</ambient>
            <diffuse>1 0 0 0.5</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- More waypoint markers... -->
  </world>
</sdf>
```

## Launching the Simulation

### 1. Basic simulation launch

```bash
# Terminal 1: Start PX4 SITL with Gazebo
cd ~/PX4-Autopilot
make px4_sitl gazebo_iris__rtk

# Terminal 2: Launch ROS 2 bridge
cd ~/px4_ros_com_ws
source install/setup.bash
ros2 launch px4_ros_com px4_ros_com.launch.py
```

### 2. Custom simulation with RTK and high-speed parameters

Create a custom launch file:

```python
# high_speed_sim.launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start Gazebo with custom world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', 'high_speed_test.world'],
            output='screen'),
            
        # Start PX4 SITL
        ExecuteProcess(
            cmd=['PX4-Autopilot/build/px4_sitl_default/bin/px4', 
                 'PX4-Autopilot/ROMFS/px4fmu_common'],
            cwd='.',
            output='screen'),
            
        # Start MAVROS
        Node(
            package='mavros',
            executable='mavros_node',
            name='mavros',
            parameters=[
                {'fcu_url': 'udp://:14540@localhost:14557'}
            ],
            output='screen'
        ),
        
        # Start trajectory tracking node
        Node(
            package='autopilot_drone',
            executable='trajectory_tracker_node',
            name='trajectory_tracker',
            output='screen'
        )
    ])
```

## Testing Procedure

1. **Launch the simulation environment**

```bash
cd ~/px4_ros_com_ws
source install/setup.bash
ros2 launch autopilot_drone high_speed_sim.launch.py
```

2. **Load PX4 parameters for high-speed flight**

```bash
# Using mavros param load
ros2 run mavros mavparam load ~/autopilot_drone/firmware/high_speed_params.yaml
```

3. **Run trajectory generation and send to the drone**

```bash
# Generate and upload trajectory
ros2 run autopilot_drone generate_trajectory --waypoints "0,0,10;10,10,15;20,0,10" --speed 15
```

## Analyzing Performance

Use the following tools to analyze the performance of your high-speed navigation:

1. **PX4 Flight Review** - Upload logs for detailed analysis
2. **ROS 2 visualization**:

```bash
# Plot trajectory tracking error
ros2 run rqt_plot rqt_plot /trajectory_tracking/position_error
```

3. **Gazebo trajectory visualization**:

```bash
# Record drone path and visualize
ros2 bag record -o flight_data /mavros/local_position/pose
ros2 run autopilot_drone visualize_trajectory path.bag
```

## Common Issues and Solutions

1. **Simulation crashes at high speeds**
   - Increase physics iteration rate
   - Reduce simulation time step

2. **Poor trajectory tracking**
   - Tune PID gains in PX4 position controller
   - Check if dynamics constraints are realistic

3. **RTK simulation issues**
   - Ensure correct RTK plugin configuration
   - Verify RTCM message simulation is working

## Next Steps

After successful simulation testing:
1. Validate in HITL (Hardware-In-The-Loop) mode
2. Gradually transfer to real drone hardware, starting with low speeds
3. Incrementally increase flight speed and validate performance