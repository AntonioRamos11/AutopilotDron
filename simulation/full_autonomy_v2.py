#!/usr/bin/env python3
"""
Advanced Full Autonomy Simulation

This script uses the actual TrajectoryOptimizer and a mock perception system
to simulate a mission with dynamic obstacle avoidance and re-planning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os
import time
import argparse
from typing import List, Tuple, Optional

# --- Import all the necessary modules from your project ---
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 1. The Drone Simulator
from trajectory_accuracy_analysis_gravity import DroneSimulator

# 2. The REAL Trajectory Optimizer
from companion_computer.trajectory_optimization.trajectory_optimizer import TrajectoryOptimizer, TrajectoryConstraints, Trajectory

# 3. The REAL Perception System
from companion_computer.obstacle_avoidance.obstacle_detector import ObstacleDetector, Obstacle, ObstacleType


# --- Main Simulation Class ---

class AdvancedAutonomySimulation:
    def __init__(self):
        # --- Mission Parameters ---
        self.start_pos = np.array([0, 0, 5], dtype=float)
        self.goal_pos = np.array([50, 0, 10], dtype=float)

        # --- Environment ---
        # We now use the real Obstacle dataclass for our environment definition.
        # It's critical to define the position/velocity vectors as floats to avoid
        # numpy casting errors during physics updates (int + float).
        self.obstacles = [
            Obstacle(id=1, position=np.array([25, 0, 7.5], dtype=float), velocity=np.array([0, 0, 0], dtype=float), size=np.array([4, 4, 15]), obstacle_type=ObstacleType.STATIC, confidence=1.0, timestamp=0),
            Obstacle(id=2, position=np.array([40, 15, 0], dtype=float), velocity=np.array([0, 0, 0], dtype=float), size=np.array([4, 4, 15]), obstacle_type=ObstacleType.STATIC, confidence=1.0, timestamp=0),
        ]

        # --- Initialize All System Components ---
        self.drone_sim = DroneSimulator(mass=1.5)
        self.drone_sim.pid_gains = {
            'kp': np.array([163.48, 163.48, 163.48]),
            'ki': np.array([11.01, 11.01, 11.01]),
            'kd': np.array([11.10, 11.10, 11.10])
        }
        self.drone_sim.reset(initial_position=self.start_pos)

        # Use the REAL TrajectoryOptimizer
        # The previous acceleration constraint (8.0 m/s^2) was too restrictive for the
        # dynamic maneuvers required for obstacle avoidance, causing planning failures.
        # Increasing it to 12.0 m/s^2 allows the optimizer to find valid solutions.
        constraints = TrajectoryConstraints(max_velocity=15.0, max_acceleration=12.0, safety_margin=2.0)
        self.optimizer = TrajectoryOptimizer(constraints=constraints)
        
        # Instantiate the REAL obstacle detector
        self.detector = ObstacleDetector()
        
        self.active_trajectory = None
        self.history = {'drone': [], 'path': [], 'events': [], 'planning_times': []}
        self.last_replan_time = -1

    def run(self, debug=False):
        """Run the full autonomy simulation."""
        dt = 0.02
        sim_time = 0.0
        step_counter = 0
        
        # 1. Initial Plan
        print("Generating initial trajectory...")
        initial_waypoints = np.array([self.start_pos, self.goal_pos])
        current_state = {'position': self.drone_sim.position, 'velocity': self.drone_sim.velocity, 'acceleration': self.drone_sim.acceleration}
        
        plan_start_time = time.time()
        opt_result = self.optimizer.optimize_trajectory(initial_waypoints, current_state)
        plan_time = time.time() - plan_start_time
        self.history['planning_times'].append(plan_time)
        
        if debug:
            print("\n--- Initial Planning Result ---")
            print(f"Success: {opt_result.success}")
            if opt_result.trajectory:
                print(f"Trajectory Segments: {opt_result.trajectory.n_segments}")
                print(f"Trajectory Time: {opt_result.trajectory.total_time:.2f}s")
            else:
                print("Trajectory object is None.")
            print("-----------------------------\n")

        # Add a robust check for trajectory validity
        valid_trajectory = False
        if opt_result.success and opt_result.trajectory is not None:
            # This is the corrected sanity check. It compares the number of time segments
            # with the number of coefficient segments, which is the source of the bug.
            n_segments_from_times = len(opt_result.trajectory.segment_times)
            n_segments_from_coeffs = opt_result.trajectory.coefficients.shape[1]
            if n_segments_from_times == n_segments_from_coeffs:
                valid_trajectory = True
            elif debug:
                print(f"!! Optimizer returned inconsistent trajectory: "
                      f"{n_segments_from_times} time segments vs "
                      f"{n_segments_from_coeffs} coefficient blocks. Discarding.")

        if not valid_trajectory:
            print("Initial planning failed or returned invalid trajectory! Aborting simulation.")
            return
            
        self.active_trajectory = opt_result.trajectory
        trajectory_start_time = sim_time

        # --- Main Simulation Loop ---
        while np.linalg.norm(self.drone_sim.position - self.goal_pos) > 1.5:
            if sim_time > 30:
                print("Simulation timed out.")
                break

            # 2. Perception (using the REAL detector)
            # In a real system, the detector would run in its own thread.
            # Here, we manually simulate the detection and tracking loop.
            
            # a) Simulate new detections from the environment's ground truth
            current_time = time.time()
            for obs in self.obstacles:
                obs.timestamp = current_time # Update timestamp for age tracking
            
            # b) Feed the ground truth to the detector's tracking logic
            self.detector._update_obstacle_tracking(self.obstacles, current_time)
            self.detector._cleanup_old_obstacles(current_time)
            
            # c) Get the list of tracked obstacles from the detector
            tracked_obstacles = self.detector.get_obstacles()
            
            # d) Feed the tracked obstacles to the optimizer
            self.optimizer.clear_obstacles()
            for obs in tracked_obstacles:
                # Pass the full size vector to the optimizer
                self.optimizer.add_obstacle(obs.position, obs.size)


            # 3. Decision Making & Re-planning
            # Check if we need to re-plan (don't re-plan too frequently)
            if (sim_time - self.last_replan_time) > 2.0:
                # The optimizer's internal logic can check for collisions
                # For this simulation, we'll trigger a replan if an obstacle is near the path
                
                # Pass the full obstacle data for a more precise check
                is_blocked, blocking_obs, blocked_point = self.is_path_blocked(self.active_trajectory, tracked_obstacles, sim_time - trajectory_start_time)
                
                if is_blocked:
                    print(f"Path blocked at t={sim_time:.2f}s! Re-planning with TrajectoryOptimizer...")
                    self.history['events'].append((sim_time, self.drone_sim.position, "Re-plan"))
                    
                    current_state = {'position': self.drone_sim.position, 'velocity': self.drone_sim.velocity, 'acceleration': self.drone_sim.acceleration}
                    
                    # --- Smarter Re-planning with Intermediate Waypoint (V3 - Stable Reference) ---
                    # The previous logic based on instantaneous velocity was unstable and caused
                    # re-plan thrashing. This new logic uses the stable vector towards the
                    # final goal as the primary axis for the avoidance maneuver.

                    # 1. Define the stable mission axis (drone-to-goal)
                    mission_direction = self.goal_pos - self.drone_sim.position
                    if np.linalg.norm(mission_direction) < 1e-6:
                        mission_direction = self.drone_sim.velocity # Fallback if on top of goal
                    mission_direction /= np.linalg.norm(mission_direction)

                    # 2. Determine if the obstacle is to the "left" or "right" of this axis
                    drone_to_obs = blocking_obs['position'] - self.drone_sim.position
                    cross_product = np.cross(mission_direction, drone_to_obs)

                    # 3. Define the "sideways" avoidance vector (perpendicular to mission axis)
                    # We create a default "left" dodge vector in the XY plane.
                    sideways_dir_xy = np.array([-mission_direction[1], mission_direction[0]])
                    
                    # To prevent re-plan thrashing, we add a dead-zone. The dodge direction
                    # only flips to "right" if the obstacle is *clearly* to the left.
                    # If it's dead-ahead (cross_product[2] is near zero), we stick with the
                    # default "left" dodge, providing stable, predictable behavior.
                    if cross_product[2] > 0.1: # Obstacle is unambiguously to the left
                        sideways_dir_xy *= -1.0 # Flip to a "right" dodge

                    # 4. Calculate the intermediate waypoint to the side of the OBSTACLE's center
                    avoid_distance = blocking_obs['radius'] + self.optimizer.constraints.safety_margin + 3.0 # Extra buffer
                    intermediate_pos_xy = blocking_obs['position'][:2] + sideways_dir_xy * avoid_distance
                    
                    # Use the obstacle's Z-height for a stable Z reference
                    intermediate_waypoint = np.array([intermediate_pos_xy[0], intermediate_pos_xy[1], blocking_obs['position'][2]])
                    
                    # New path: current_pos -> intermediate_waypoint -> goal_pos
                    new_waypoints = np.array([self.drone_sim.position, intermediate_waypoint, self.goal_pos])
                    
                    # The optimizer will now use its internal _avoid_obstacles logic
                    plan_start_time = time.time()
                    opt_result = self.optimizer.optimize_trajectory(new_waypoints, current_state)
                    plan_time = time.time() - plan_start_time
                    self.history['planning_times'].append(plan_time)

                    if debug:
                        print("\n--- Re-planning Result ---")
                        print(f"Success: {opt_result.success}")
                        if opt_result.trajectory:
                            print(f"Trajectory Segments: {opt_result.trajectory.n_segments}")
                            print(f"Trajectory Time: {opt_result.trajectory.total_time:.2f}s")
                        else:
                            print("Trajectory object is None.")
                        print("--------------------------\n")

                    # Add the same robust check for re-planning
                    valid_trajectory = False
                    if opt_result.success and opt_result.trajectory is not None:
                        # Corrected sanity check for re-planning
                        n_segments_from_times = len(opt_result.trajectory.segment_times)
                        n_segments_from_coeffs = opt_result.trajectory.coefficients.shape[1]
                        if n_segments_from_times == n_segments_from_coeffs:
                            valid_trajectory = True
                        elif debug:
                            print(f"!! Optimizer returned inconsistent trajectory on re-plan. Discarding.")

                    if valid_trajectory:
                        self.active_trajectory = opt_result.trajectory
                        trajectory_start_time = sim_time # Reset the clock for the new trajectory
                        self.last_replan_time = sim_time
                    else:
                        print("Re-planning failed or returned invalid trajectory. Continuing on old path.")

            # 4. Control
            trajectory_time = sim_time - trajectory_start_time
            if trajectory_time > self.active_trajectory.total_time:
                trajectory_time = self.active_trajectory.total_time # Hold last point

            target_pos = self.active_trajectory.evaluate(trajectory_time)
            target_vel = self.active_trajectory.evaluate(trajectory_time, derivative_order=1)
            target_acc = self.active_trajectory.evaluate(trajectory_time, derivative_order=2)

            if debug and step_counter % 100 == 0: # Print every 25 steps (0.5s)
                print(f"--- t={sim_time:.2f}s ---")
                print(f"  Drone Pos:  {self.drone_sim.position[0]:>6.2f}, {self.drone_sim.position[1]:>6.2f}, {self.drone_sim.position[2]:>6.2f}")
                print(f"  Target Pos: {target_pos[0]:>6.2f}, {target_pos[1]:>6.2f}, {target_pos[2]:>6.2f}")
                print(f"  Traj Time: {trajectory_time:.2f}s / {self.active_trajectory.total_time:.2f}s")

            # 5. Simulation Step
            self.drone_sim.update(target_pos, target_vel, target_acc, dt)
            
            # Update the position of obstacles in the environment based on their velocity.
            # This was previously handled by a method in the MockObstacle class.
            for obs in self.obstacles:
                obs.position += obs.velocity * dt

            # Store history
            self.history['drone'].append(self.drone_sim.position.copy())
            self.history['path'].append(target_pos.copy())
            
            sim_time += dt
            step_counter += 1

        print("Goal reached!")
        self.analyze_performance()
        self.visualize()

    def analyze_performance(self):
        """Calculate and print key performance metrics for the mission."""
        print("\n--- Mission Performance Analysis ---")

        drone_path = np.array(self.history['drone'])
        planned_path = np.array(self.history['path'])
        dt = 0.02 # Assuming fixed timestep

        # 1. Mission Efficiency
        total_time = len(drone_path) * dt
        distances = np.linalg.norm(np.diff(drone_path, axis=0), axis=1)
        total_distance = np.sum(distances)
        straight_line_dist = np.linalg.norm(self.goal_pos - self.start_pos)
        path_efficiency = straight_line_dist / total_distance if total_distance > 0 else 0

        print("\n[Efficiency Metrics]")
        print(f"  Total Flight Time:    {total_time:.2f} s")
        print(f"  Total Flight Distance:  {total_distance:.2f} m")
        print(f"  Path Efficiency Ratio:  {path_efficiency:.2f} (1.0 is optimal)")

        # 2. Control & Tracking Performance
        errors = np.linalg.norm(drone_path - planned_path, axis=1)
        rmse_tracking_error = np.sqrt(np.mean(errors**2))
        max_tracking_error = np.max(errors) if len(errors) > 0 else 0

        print("\n[Control & Tracking Metrics]")
        print(f"  RMSE Tracking Error:    {rmse_tracking_error:.3f} m")
        print(f"  Max Tracking Error:     {max_tracking_error:.3f} m")

        # 3. Planning Performance
        num_replans = len([event for event in self.history['events'] if event[2] == "Re-plan"])
        avg_plan_time = np.mean(self.history['planning_times']) if self.history['planning_times'] else 0

        print("\n[Planning Metrics]")
        print(f"  Number of Re-plans:     {num_replans}")
        print(f"  Average Planning Time:  {avg_plan_time * 1000:.2f} ms")
        print("------------------------------------")


    def is_path_blocked(self, trajectory: Trajectory, obstacles: List[Obstacle], current_traj_time: float) -> Tuple[bool, Optional[dict], Optional[np.ndarray]]:
        """
        Check if the trajectory path is blocked by any obstacles using AABB collision detection.
        This is more accurate than using a simple radius for non-spherical obstacles.
        """
        if trajectory is None:
            return False, None, None

        horizon_time = 4.0
        t_steps = np.linspace(current_traj_time, min(current_traj_time + horizon_time, trajectory.total_time), 15)
        
        for t in t_steps:
            path_point = trajectory.evaluate(t)
            for obs in obstacles:
                # AABB (Axis-Aligned Bounding Box) collision check
                half_size = obs.size / 2.0
                safety_margin = self.optimizer.constraints.safety_margin
                
                # Calculate the min and max corners of the obstacle's bounding box
                min_corner = obs.position - half_size - safety_margin
                max_corner = obs.position + half_size + safety_margin
                
                # Check if the path point is inside the bounding box
                if np.all(path_point >= min_corner) and np.all(path_point <= max_corner):
                    # Convert the Obstacle object to the dictionary format expected by the re-planner
                    blocking_obs_dict = {'position': obs.position, 'size': obs.size, 'radius': np.max(obs.size) / 2.0}
                    return True, blocking_obs_dict, path_point
                    
        return False, None, None

    def visualize(self):
        # (Visualization code is the same as before)
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-5, 60)
        ax.set_ylim(-15, 15)
        ax.set_title("Advanced Autonomy Simulation with TrajectoryOptimizer")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        
        drone_path = np.array(self.history['drone'])
        planned_path = np.array(self.history['path']);
        ax.plot(drone_path[:, 0], drone_path[:, 1], 'r-', label="Drone Path")
        ax.plot(planned_path[:, 0], planned_path[:, 1], 'b--', alpha=0.5, label="Planned Path")

        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=10, label="Start")
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'ro', markersize=10, label="Goal")

        for obs in self.obstacles:
            c = Circle(obs.position[:2], radius=obs.size[0], color='gray', alpha=0.8)
            ax.add_patch(c)
            ax.text(obs.position[0], obs.position[1], f"Obs {obs.id}", ha='center', va='center')

        for t, pos, event in self.history['events']:
            ax.plot(pos[0], pos[1], 'y*', markersize=15, label=f"{event} @ {t:.1f}s")

        ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    # Add a simple argument parser for the debug flag
    parser = argparse.ArgumentParser(description="Advanced Autonomy Simulation")
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug printing.')
    args = parser.parse_args()

    sim = AdvancedAutonomySimulation()
    sim.run(debug=args.debug)
