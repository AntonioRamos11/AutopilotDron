from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Path to the airsim_ros_pkgs launch file
    airsim_ros_pkg_path = get_package_share_directory('airsim_ros_pkgs')
    
    # Path to your trajectory tracker executable
    # Note: Assumes your package is named 'autopilot_drone' and the node is 'trajectory_tracker'
    # You may need to adjust this based on your setup.
    
    return LaunchDescription([
        # 1. Start the AirSim ROS 2 Wrapper
        # This node connects to the simulator and provides the /airsim_node/* topics
        Node(
            package='airsim_ros_pkgs',
            executable='airsim_node',
            name='airsim_node',
            output='screen',
            parameters=[{
                'is_vulkan': False, # Set to True if using Vulkan
                'host_ip': '127.0.0.1'
            }]
        ),
        
        # 2. Start your Trajectory Tracker Node
        # This is the core of your autopilot logic
        Node(
            package='autopilot_drone', # Replace with your package name
            executable='trajectory_tracker', # The executable from your setup.py
            name='trajectory_tracker_node',
            output='screen'
        ),
        
        # 3. Start the new AirSim Interface Node
        # This node bridges the gap between AirSim and your autopilot
        Node(
            package='autopilot_drone', # Replace with your package name
            executable='airsim_interface', # The executable for the new script
            name='airsim_interface_node',
            output='screen'
        )
    ])