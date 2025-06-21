import rclpy
from rclpy.node import Node
import numpy as np
import time

# Import message types
from px4_msgs.msg import VehicleLocalPosition, TrajectorySetpoint, OffboardControlMode
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger

class AirSimInterfaceNode(Node):
    """
    A ROS 2 node to bridge the Autopilot system with the AirSim simulator.
    - Subscribes to AirSim's odometry and sensor data.
    - Publishes data in the format expected by the TrajectoryTrackerNode (PX4 messages).
    - Subscribes to control setpoints from the TrajectoryTrackerNode.
    - Calls AirSim services to arm, take off, and control the drone.
    """
    def __init__(self):
        super().__init__('airsim_interface')
        self.get_logger().info("AirSim Interface starting...")

        # --- State Variables ---
        self.is_armed = False
        self.is_flying = False
        self.offboard_mode_active = False

        # --- Publishers for the Autopilot System ---
        self.vehicle_local_position_pub = self.create_publisher(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', 10)
        
        # --- Subscribers to AirSim Data ---
        self.airsim_odom_sub = self.create_subscription(
            Odometry, '/airsim_node/drone_1/odom_local_ned', self.airsim_odom_callback, 10)
        
        # --- Subscribers to Autopilot Control Commands ---
        self.trajectory_setpoint_sub = self.create_subscription(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self.trajectory_setpoint_callback, 10)
        self.offboard_control_mode_sub = self.create_subscription(
            OffboardControlMode, '/fmu/in/offboard_control_mode', self.offboard_control_mode_callback, 10)

        # --- Publishers and Service Clients for AirSim Control ---
        self.airsim_vel_cmd_pub = self.create_publisher(
            Twist, '/airsim_node/drone_1/vel_cmd_body_frame', 10)
        self.arm_service_client = self.create_client(Trigger, '/airsim_node/drone_1/arm')
        self.takeoff_service_client = self.create_client(Trigger, '/airsim_node/drone_1/takeoff')

        self.get_logger().info("AirSim Interface node has started.")

    def airsim_odom_callback(self, msg: Odometry):
        """Translate AirSim odometry to PX4 VehicleLocalPosition and publish."""
        px4_pos_msg = VehicleLocalPosition()
        
        px4_pos_msg.timestamp = int(time.time() * 1e6)
        
        # Position
        px4_pos_msg.x = msg.pose.pose.position.x
        px4_pos_msg.y = msg.pose.pose.position.y
        px4_pos_msg.z = msg.pose.pose.position.z
        
        # Velocity
        px4_pos_msg.vx = msg.twist.twist.linear.x
        px4_pos_msg.vy = msg.twist.twist.linear.y
        px4_pos_msg.vz = msg.twist.twist.linear.z
        
        # Orientation (convert quaternion to euler for heading)
        q = msg.pose.pose.orientation
        _, _, yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        px4_pos_msg.heading = yaw
        
        self.vehicle_local_position_pub.publish(px4_pos_msg)

    def offboard_control_mode_callback(self, msg: OffboardControlMode):
        """Handle requests to enter offboard mode by arming and taking off."""
        if msg.position or msg.velocity:
            if not self.offboard_mode_active:
                self.get_logger().info("Offboard mode requested. Initiating takeoff sequence.")
                self.offboard_mode_active = True
                self.arm_and_takeoff()
        else:
            self.offboard_mode_active = False

    def trajectory_setpoint_callback(self, msg: TrajectorySetpoint):
        """Translate trajectory setpoints to AirSim velocity commands."""
        if not self.is_flying:
            return

        twist_msg = Twist()
        # The setpoint from our tracker is in world frame (NED), but AirSim's
        # vel_cmd_body_frame expects body frame. For simple hovering and movement
        # without complex rotation, we can approximate. A full implementation
        # would require a frame transformation.
        twist_msg.linear.x = msg.velocity[0]
        twist_msg.linear.y = msg.velocity[1]
        twist_msg.linear.z = msg.velocity[2]
        
        self.airsim_vel_cmd_pub.publish(twist_msg)

    def arm_and_takeoff(self):
        """Call AirSim services to arm the drone and take off."""
        if self.is_armed:
            return

        self.get_logger().info("Calling arm service...")
        self.arm_service_client.wait_for_service()
        arm_request = Trigger.Request()
        future = self.arm_service_client.call_async(arm_request)
        future.add_done_callback(self.arm_response_callback)

    def arm_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Arming successful!")
                self.is_armed = True
                time.sleep(1) # Wait a moment before takeoff
                
                self.get_logger().info("Calling takeoff service...")
                self.takeoff_service_client.wait_for_service()
                takeoff_request = Trigger.Request()
                future = self.takeoff_service_client.call_async(takeoff_request)
                future.add_done_callback(self.takeoff_response_callback)
            else:
                self.get_logger().error(f"Arming failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f'Service call failed %r' % (e,))

    def takeoff_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Takeoff successful! Ready to fly.")
                self.is_flying = True
            else:
                self.get_logger().error(f"Takeoff failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f'Service call failed %r' % (e,))

    def euler_from_quaternion(self, x, y, z, w):
        """Convert a quaternion into euler angles (roll, pitch, yaw)."""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z

def main(args=None):
    rclpy.init(args=args)
    node = AirSimInterfaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()