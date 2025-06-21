#!/usr/bin/env python3
"""
Trajectory Tracker Node for High-Speed Drone Navigation

This ROS 2 node interfaces between the minimum snap trajectory generator
and the PX4 flight controller to execute high-speed, accurate point-to-point
navigation with RTK GPS precision.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import time
import sys
import os
import threading

# Import PX4 message types
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleGpsPosition

# Import standard ROS2 message types
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose

# Add parent directory to sys.path to import the MinSnapTrajectory module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'path_planning'))
from min_snap_trajectory import MinSnapTrajectory, Trajectory
# Import the new avoidance modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'companion_computer', 'obstacle_avoidance'))
from obstacle_detector import ObstacleDetector
from avoidance_planner import ObstacleAvoidancePlanner


class TrajectoryTrackerNode(Node):
    """
    ROS 2 node for trajectory tracking with PX4 for high-speed, accurate navigation.
    """
    
    def __init__(self):
        super().__init__('trajectory_tracker')
        
        # Configure QoS profile for PX4 communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile)
        
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile)
            
        self.position_error_publisher = self.create_publisher(
            Float32MultiArray,
            '/trajectory_tracking/position_error',
            10)
            
        self.trajectory_visualization_publisher = self.create_publisher(
            PoseArray,
            '/trajectory_tracking/visualization',
            10)
            
        # Create subscribers
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
            
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)
            
        self.vehicle_gps_position_subscriber = self.create_subscription(
            VehicleGpsPosition,
            '/fmu/out/vehicle_gps_position',
            self.vehicle_gps_position_callback,
            qos_profile)
            
        self.trajectory_waypoints_subscriber = self.create_subscription(
            Float32MultiArray,
            '/trajectory_tracking/waypoints',
            self.trajectory_waypoints_callback,
            10)
            
        self.command_subscriber = self.create_subscription(
            String,
            '/trajectory_tracking/command',
            self.command_callback,
            10)
        
        # Create a timer for the main control loop
        self.timer_period = 0.02  # 50Hz for offboard control
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        
        # Create a timer for the avoidance planner loop (runs slower)
        self.avoidance_timer_period = 0.2 # 5Hz
        self.avoidance_timer = self.create_timer(self.avoidance_timer_period, self.avoidance_loop)
        
        # State variables
        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_gps_position = VehicleGpsPosition()
        self.trajectory = None
        self.execution_start_time = None
        self.is_executing_trajectory = False
        self.offboard_mode = False
        self.current_waypoint_idx = 0
        self.rtk_enabled = False
        
        # Create trajectory generator
        self.trajectory_generator = MinSnapTrajectory()
        
        # --- Initialize Obstacle Avoidance ---
        self.obstacle_detector = ObstacleDetector()
        # TODO: In a real system, you would set sensor interfaces here
        # self.obstacle_detector.set_sensor_interfaces(...) 
        self.obstacle_detector.start_detection()
        
        # The planner needs a publisher to send new waypoints back to this node
        waypoint_publisher = self.create_publisher(Float32MultiArray, '/trajectory_tracking/waypoints', 10)
        self.avoidance_planner = ObstacleAvoidancePlanner(
            self.obstacle_detector,
            self.trajectory_generator,
            waypoint_publisher
        )
        # --- End Obstacle Avoidance Init ---
        
        # Thread lock for thread safety
        self.lock = threading.Lock()
        
        self.get_logger().info('Trajectory Tracker Node initialized')
    
    def vehicle_status_callback(self, msg):
        """Process vehicle status updates."""
        self.vehicle_status = msg
        
        # Check if we've entered offboard mode
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if not self.offboard_mode:
                self.offboard_mode = True
                self.get_logger().info("Vehicle is now in OFFBOARD mode")
        else:
            if self.offboard_mode:
                self.offboard_mode = False
                self.is_executing_trajectory = False
                self.get_logger().info("Vehicle left OFFBOARD mode, trajectory execution stopped")
    
    def vehicle_local_position_callback(self, msg):
        """Process local position updates."""
        with self.lock:
            self.vehicle_local_position = msg
            
            # Update the avoidance planner with the drone's current state
            current_pos = np.array([msg.x, msg.y, msg.z])
            current_vel = np.array([msg.vx, msg.vy, msg.vz])
            self.avoidance_planner.update_drone_state(current_pos, current_vel)
            
            # Calculate position error if executing trajectory
            if self.is_executing_trajectory and self.trajectory is not None:
                current_time = time.time()
                trajectory_time = current_time - self.execution_start_time
                
                if trajectory_time <= self.trajectory.total_time:
                    # Get the setpoint from the trajectory
                    setpoint = self.trajectory.evaluate(trajectory_time)
                    
                    # Calculate error
                    position_error = np.array([
                        setpoint[0] - self.vehicle_local_position.x,
                        setpoint[1] - self.vehicle_local_position.y,
                        setpoint[2] - self.vehicle_local_position.z
                    ])
                    
                    # Publish position error
                    error_msg = Float32MultiArray()
                    error_msg.data = [float(e) for e in position_error] + [float(trajectory_time)]
                    self.position_error_publisher.publish(error_msg)
    
    def vehicle_gps_position_callback(self, msg):
        """Process GPS position updates."""
        self.vehicle_gps_position = msg
        
        # Check if RTK fix is available
        rtk_enabled = (msg.fix_type >= 5)  # 5 or 6 indicate RTK float or fixed
        
        if rtk_enabled != self.rtk_enabled:
            self.rtk_enabled = rtk_enabled
            if self.rtk_enabled:
                self.get_logger().info("RTK fix acquired")
            else:
                self.get_logger().warn("RTK fix lost")
    
    def trajectory_waypoints_callback(self, msg):
        """
        Process new trajectory waypoints.
        
        Expected format:
        - First 3 values: start velocity [vx, vy, vz]
        - Last 3 values: end velocity [vx, vy, vz]
        - Remaining values: waypoints in format [x1, y1, z1, x2, y2, z2, ...]
        """
        with self.lock:
            try:
                data = np.array(msg.data)
                
                # Extract start and end velocities
                start_vel = data[0:3]
                end_vel = data[-3:]
                
                # Extract waypoints (exclude first 3 and last 3 values)
                waypoints_flat = data[3:-3]
                
                if len(waypoints_flat) % 3 != 0:
                    self.get_logger().error(f"Invalid waypoints data: length {len(waypoints_flat)} is not divisible by 3")
                    return
                
                num_waypoints = len(waypoints_flat) // 3
                waypoints = waypoints_flat.reshape(num_waypoints, 3)
                
                self.get_logger().info(f"Received {num_waypoints} waypoints")
                
                # Default segment times based on distance and desired speed
                # Assume 10 m/s for now, this could be parametrized
                desired_speed = 15.0  # m/s
                segment_times = []
                
                for i in range(len(waypoints) - 1):
                    distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
                    time = max(distance / desired_speed, 0.1)  # At least 0.1 seconds per segment
                    segment_times.append(time)
                
                # Generate the trajectory
                self.get_logger().info(f"Generating trajectory with segment times: {segment_times}")
                
                trajectory = self.trajectory_generator.generate_trajectory(
                    waypoints, 
                    np.array(segment_times),
                    start_vel=start_vel,
                    end_vel=end_vel,
                    start_acc=np.zeros(3),
                    end_acc=np.zeros(3)
                )
                
                self.trajectory = trajectory
                self.current_waypoint_idx = 0
                self.is_executing_trajectory = False
                
                # Update the avoidance planner with the new trajectory
                self.avoidance_planner.update_trajectory(trajectory, waypoints[-1])
                
                # Publish trajectory visualization
                self.publish_trajectory_visualization()
                
                self.get_logger().info("New trajectory generated successfully")
            except Exception as e:
                self.get_logger().error(f"Error processing waypoints: {str(e)}")
    
    def command_callback(self, msg):
        """Process commands for trajectory execution."""
        command = msg.data.strip().lower()
        
        if command == "start":
            if self.trajectory is not None:
                self.execution_start_time = time.time()
                self.is_executing_trajectory = True
                self.get_logger().info("Starting trajectory execution")
            else:
                self.get_logger().warn("Cannot start: No trajectory available")
        
        elif command == "stop":
            self.is_executing_trajectory = False
            self.get_logger().info("Stopping trajectory execution")
        
        elif command == "reset":
            self.trajectory = None
            self.is_executing_trajectory = False
            self.get_logger().info("Trajectory reset")
            
        elif command.startswith("speed "):
            try:
                # Update the trajectory speed
                speed = float(command.split()[1])
                if self.trajectory is not None:
                    self.get_logger().info(f"Adjusting trajectory speed to {speed} m/s")
                    # This would require regenerating the trajectory with new segment times
                    # Not implemented here for simplicity
                else:
                    self.get_logger().warn("Cannot adjust speed: No trajectory available")
            except:
                self.get_logger().error("Invalid speed command format")
    
    def publish_trajectory_visualization(self):
        """Publish the trajectory for visualization in RViz."""
        if self.trajectory is None:
            return
            
        # Sample points along the trajectory
        num_samples = 100
        times = np.linspace(0, self.trajectory.total_time, num_samples)
        
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for t in times:
            position = self.trajectory.evaluate(t)
            velocity = self.trajectory.evaluate(t, derivative_order=1)
            
            # Create a pose
            pose = Pose()
            pose.position.x = float(position[0])
            pose.position.y = float(position[1])
            pose.position.z = float(position[2])
            
            # Use orientation to represent velocity direction
            # This is a simplification, a proper quaternion calculation would be better
            magnitude = np.linalg.norm(velocity)
            if magnitude > 0.001:
                # Normalize velocity
                velocity = velocity / magnitude
                
                # Very simple conversion from direction vector to quaternion
                # This is just for visualization purposes
                pose.orientation.x = float(velocity[0])
                pose.orientation.y = float(velocity[1])
                pose.orientation.z = float(velocity[2])
                pose.orientation.w = 0.0
            else:
                pose.orientation.w = 1.0
            
            pose_array.poses.append(pose)
        
        self.trajectory_visualization_publisher.publish(pose_array)
    
    def publish_offboard_control_mode(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        
        self.offboard_control_mode_publisher.publish(msg)
    
    def avoidance_loop(self):
        """Periodic loop to run the avoidance planner."""
        with self.lock:
            if self.is_executing_trajectory:
                self.avoidance_planner.check_and_replan()

    def publish_trajectory_setpoint(self, position, velocity):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.x = float(position[0])
        msg.y = float(position[1])
        msg.z = float(position[2])
        msg.vx = float(velocity[0])
        msg.vy = float(velocity[1])
        msg.vz = float(velocity[2])
        
        self.trajectory_setpoint_publisher.publish(msg)
    
    def control_loop(self):
        """Main control loop."""
        # Must always publish the offboard control mode
        self.publish_offboard_control_mode()
        
        # If we're executing a trajectory, publish the setpoint
        if self.is_executing_trajectory and self.trajectory is not None:
            current_time = time.time()
            trajectory_time = current_time - self.execution_start_time
            
            if trajectory_time <= self.trajectory.total_time:
                # Get the setpoint from the trajectory
                position = self.trajectory.evaluate(trajectory_time)
                velocity = self.trajectory.evaluate(trajectory_time, derivative_order=1)
                
                # Publish the setpoint
                self.publish_trajectory_setpoint(position, velocity)
            else:
                # Trajectory completed
                self.get_logger().info("Trajectory execution completed")
                self.is_executing_trajectory = False
                
                # Hold the last position
                position = self.trajectory.evaluate(self.trajectory.total_time)
                velocity = np.zeros(3)
                self.publish_trajectory_setpoint(position, velocity)
        else:
            # If not executing trajectory but in offboard mode, hold current position
            if self.offboard_mode:
                position = np.array([
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    self.vehicle_local_position.z
                ])
                velocity = np.zeros(3)
                self.publish_trajectory_setpoint(position, velocity)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = TrajectoryTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()