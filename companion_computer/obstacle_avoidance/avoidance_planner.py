import numpy as np
import logging
import time

# Import project-specific modules
from companion_computer.obstacle_avoidance.obstacle_detector import ObstacleDetector
from path_planning.min_snap_trajectory import MinSnapTrajectory, Trajectory
from std_msgs.msg import Float32MultiArray

class ObstacleAvoidancePlanner:
    """
    Monitors the drone's trajectory for potential collisions and re-plans a safe path.
    """
    def __init__(self, obstacle_detector: ObstacleDetector, trajectory_generator: MinSnapTrajectory, waypoint_publisher):
        """
        Initialize the planner.
        
        Args:
            obstacle_detector: An instance of the ObstacleDetector.
            trajectory_generator: An instance of the MinSnapTrajectory generator.
            waypoint_publisher: A ROS 2 publisher for sending new waypoints.
        """
        self.logger = logging.getLogger(__name__)
        self.obstacle_detector = obstacle_detector
        self.trajectory_generator = trajectory_generator
        self.waypoint_publisher = waypoint_publisher
        
        self.current_trajectory: Optional[Trajectory] = None
        self.current_drone_position = np.zeros(3)
        self.current_drone_velocity = np.zeros(3)
        self.goal_position = np.zeros(3)
        
        # Avoidance parameters
        self.replanning_horizon = 5.0  # seconds
        self.safety_corridor_width = 4.0  # meters
        self.avoidance_distance = 3.0  # meters to offset from obstacle

    def update_drone_state(self, position: np.ndarray, velocity: np.ndarray):
        """Update the drone's current position and velocity."""
        self.current_drone_position = position
        self.current_drone_velocity = velocity

    def update_trajectory(self, trajectory: Trajectory, goal_position: np.ndarray):
        """Update the current trajectory being tracked."""
        self.current_trajectory = trajectory
        self.goal_position = goal_position

    def check_and_replan(self):
        """
        The main loop for the planner. Checks for collisions and triggers re-planning if needed.
        """
        if self.current_trajectory is None:
            return  # No active trajectory to check

        # 1. Check for collisions on the current path
        is_clear, blocking_obstacle = self._is_path_imminently_blocked()
        
        if not is_clear:
            self.logger.warning(f"Collision predicted with obstacle {blocking_obstacle.id}! Re-planning...")
            
            # 2. Generate a new avoidance trajectory
            new_trajectory = self._plan_avoidance_trajectory(blocking_obstacle)
            
            if new_trajectory:
                # 3. Command the drone to follow the new path
                self.logger.info("Publishing new avoidance trajectory.")
                # The TrajectoryTrackerNode will receive this and start executing it
                self.waypoint_publisher.publish(new_trajectory)

    def _is_path_imminently_blocked(self) -> (bool, Optional[Obstacle]):
        """
        Checks if the path within the replanning horizon is blocked.
        """
        start_pos = self.current_drone_position
        
        # Evaluate a point on the trajectory a few seconds into the future
        future_time = min(self.replanning_horizon, self.current_trajectory.total_time)
        end_pos = self.current_trajectory.evaluate(future_time)
        
        is_clear, blocking_obstacles = self.obstacle_detector.is_path_clear(
            start_pos, end_pos, self.safety_corridor_width
        )
        
        return is_clear, blocking_obstacles[0] if not is_clear else None

    def _plan_avoidance_trajectory(self, obstacle: Obstacle) -> Optional[Float32MultiArray]:
        """
        Generates a new set of waypoints to navigate around a given obstacle.
        """
        # --- Decision Making ---
        # Simple strategy: find a safe point to the side of the obstacle.
        # A more advanced strategy would check for free space using sensor data.
        
        # Vector from drone to obstacle
        drone_to_obstacle = obstacle.position - self.current_drone_position
        
        # A vector perpendicular to the drone-to-obstacle vector (horizontal plane)
        # This gives us a direction to move "sideways"
        avoidance_direction = np.array([-drone_to_obstacle[1], drone_to_obstacle[0], 0])
        if np.linalg.norm(avoidance_direction) < 1e-6:
             # If obstacle is directly ahead, default to a rightward maneuver
            avoidance_direction = np.array([0, 1, 0])
            
        avoidance_direction /= np.linalg.norm(avoidance_direction)
        
        # Calculate a new intermediate waypoint next to the obstacle
        avoidance_waypoint = obstacle.position + avoidance_direction * (np.max(obstacle.size)/2 + self.avoidance_distance)
        
        # --- Re-planning ---
        # Create a new mission: Current Position -> Avoidance Waypoint -> Original Goal
        new_waypoints = np.array([
            self.current_drone_position,
            avoidance_waypoint,
            self.goal_position
        ])
        
        # Format for the TrajectoryTrackerNode
        # Format: [start_vel, waypoints_flat, end_vel]
        start_vel = self.current_drone_velocity
        end_vel = np.zeros(3) # Come to a stop at the final goal
        
        waypoints_flat = new_waypoints.flatten()
        
        msg_data = np.concatenate([start_vel, waypoints_flat, end_vel])
        
        # Create the ROS 2 message
        msg = Float32MultiArray()
        msg.data = [float(x) for x in msg_data]
        
        return msg