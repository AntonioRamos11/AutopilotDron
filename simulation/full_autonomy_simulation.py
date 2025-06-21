#!/usr/bin/env python3
"""
Full Autonomy Simulation for Drone Navigation

This script integrates path planning, perception, decision-making, and control
to simulate a fully autonomous mission. The drone will attempt to fly to a goal
while detecting and avoiding dynamic and static obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import sys
import os

# --- Import all the necessary modules from your project ---

# 1. The Drone Simulator and Controller from your previous work
#    (We will copy the class here for simplicity, but in a real project, you'd import it)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from simulation.trajectory_accuracy_analysis_gravity import DroneSimulator

# 2. The Path Planner
from path_planning.min_snap_trajectory import MinSnapTrajectory, Trajectory

# 3. The Perception System (Obstacle Detector)
#    We will create a simplified mock version for this simulation
class MockObstacle:
    def __init__(self, id, position, velocity, size):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.size = np.array(size, dtype=float)

    def update(self, dt):
        self.position += self.velocity * dt

class MockObstacleDetector:
    """A simplified detector for the simulation environment."""
    def __init__(self, environment_obstacles):
        self.obstacles = environment_obstacles

    def get_obstacles(self):
        # In a real system, this would involve complex sensor fusion.
        # Here, we just return the ground truth from the environment.
        return self.obstacles

# 4. The Decision-Making System (Avoidance Planner)
#    A simplified version of your avoidance planner logic.
class MockAvoidancePlanner:
    def __init__(self, safety_distance=3.0):
        self.safety_distance = safety_distance

    def is_path_blocked(self, drone_pos, trajectory, obstacles, horizon_time):
        """Check if the trajectory collides with any obstacle within a time horizon."""
        t_steps = np.linspace(0, min(horizon_time, trajectory.total_time), 20)
        for t in t_steps:
            path_point = trajectory.evaluate(t)
            for obs in obstacles:
                dist = np.linalg.norm(path_point - obs.position)
                if dist < (np.max(obs.size) / 2.0 + self.safety_distance):
                    return True, obs # Path is blocked by this obstacle
        return False, None

    def plan_avoidance_path(self, drone_pos, original_goal, blocking_obstacle):
        """Generate a new waypoint to go around the obstacle."""
        # Vector from drone to obstacle
        drone_to_obstacle = blocking_obstacle.position - drone_pos
        
        # A vector perpendicular to the drone-to-obstacle vector (horizontal plane)
        avoidance_direction = np.array([-drone_to_obstacle[1], drone_to_obstacle[0], 0])
        if np.linalg.norm(avoidance_direction) < 1e-6:
            avoidance_direction = np.array([0, 1, 0]) # Default if aligned
            
        avoidance_direction /= np.linalg.norm(avoidance_direction)
        
        # Calculate a new intermediate waypoint next to the obstacle
        avoidance_waypoint = blocking_obstacle.position + avoidance_direction * (np.max(blocking_obstacle.size) + self.safety_distance)
        
        # New mission: Current Position -> Avoidance Waypoint -> Original Goal
        return np.array([drone_pos, avoidance_waypoint, original_goal])


# --- Main Simulation Class ---

class FullAutonomySimulation:
    def __init__(self):
        # --- Mission Parameters ---
        self.start_pos = np.array([0, 0, 5])
        self.goal_pos = np.array([50, 0, 10])
        self.desired_speed = 10.0  # m/s

        # --- Environment ---
        self.obstacles = [
            MockObstacle(id=1, position=[25, 5, 7], velocity=[0, -2, 0], size=[3, 3, 10]), # Dynamic
            MockObstacle(id=2, position=[40, -8, 0], velocity=[0, 0, 0], size=[4, 4, 15]), # Static
        ]

        # --- Initialize All System Components ---
        self.drone_sim = DroneSimulator(mass=1.5) # Using the tuned gains from your file
        self.drone_sim.pid_gains = {
            'kp': np.array([163.48, 163.48, 163.48]),
            'ki': np.array([11.01, 11.01, 11.01]),
            'kd': np.array([11.10, 11.10, 11.10])
        }
        self.drone_sim.reset(initial_position=self.start_pos)

        self.planner = MinSnapTrajectory()
        self.detector = MockObstacleDetector(self.obstacles)
        self.avoider = MockAvoidancePlanner()
        
        self.active_trajectory = None
        self.history = {'drone': [], 'path': [], 'events': []}

    def generate_new_trajectory(self, waypoints):
        """Generates a new minimum snap trajectory."""
        n_segments = len(waypoints) - 1
        distances = [np.linalg.norm(waypoints[i+1] - waypoints[i]) for i in range(n_segments)]
        segment_times = [max(dist / self.desired_speed, 1.0) for dist in distances]
        
        print(f"Planning new trajectory through {len(waypoints)} waypoints over {sum(segment_times):.2f}s.")
        
        return self.planner.generate_trajectory(
            waypoints, 
            segment_times,
            start_vel=self.drone_sim.velocity,
            end_vel=np.zeros(3)
        )

    def run(self):
        """Run the full autonomy simulation."""
        dt = 0.02  # 50 Hz
        sim_time = 0.0
        
        # 1. Initial Plan
        self.active_trajectory = self.generate_new_trajectory(np.array([self.start_pos, self.goal_pos]))

        # --- Main Simulation Loop ---
        while np.linalg.norm(self.drone_sim.position - self.goal_pos) > 1.5:
            if sim_time > 30: # Safety break
                print("Simulation timed out.")
                break

            # 2. Perception
            detected_obstacles = self.detector.get_obstacles()

            # 3. Decision Making (Check for Collision)
            is_blocked, blocking_obs = self.avoider.is_path_blocked(
                self.drone_sim.position, self.active_trajectory, detected_obstacles, horizon_time=3.0
            )

            # 4. Re-planning
            if is_blocked:
                print(f"Path blocked by obstacle {blocking_obs.id} at t={sim_time:.2f}s! Re-planning...")
                self.history['events'].append((sim_time, self.drone_sim.position, "Re-plan"))
                new_waypoints = self.avoider.plan_avoidance_path(self.drone_sim.position, self.goal_pos, blocking_obs)
                self.active_trajectory = self.generate_new_trajectory(new_waypoints)
                # Reset trajectory time to start of new path
                self.active_trajectory.total_time += sim_time 

            # 5. Control
            # Evaluate the current trajectory to get the setpoint
            target_pos = self.active_trajectory.evaluate(sim_time)
            target_vel = self.active_trajectory.evaluate(sim_time, derivative_order=1)
            target_acc = self.active_trajectory.evaluate(sim_time, derivative_order=2)

            # 6. Simulation Step
            # Update drone physics
            self.drone_sim.update(target_pos, target_vel, target_acc, dt)
            # Update obstacle physics
            for obs in self.obstacles:
                obs.update(dt)

            # Store history for visualization
            self.history['drone'].append(self.drone_sim.position.copy())
            self.history['path'].append(target_pos.copy())
            
            sim_time += dt

        print("Goal reached!")
        self.visualize()

    def visualize(self):
        """Visualize the simulation results."""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-5, 60)
        ax.set_ylim(-15, 15)
        ax.set_title("Full Autonomy Simulation")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        
        # Plot history
        drone_path = np.array(self.history['drone'])
        planned_path = np.array(self.history['path'])
        ax.plot(drone_path[:, 0], drone_path[:, 1], 'r-', label="Drone Path")
        ax.plot(planned_path[:, 0], planned_path[:, 1], 'b--', alpha=0.5, label="Planned Path")

        # Plot start and goal
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=10, label="Start")
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'ro', markersize=10, label="Goal")

        # Plot obstacles at their final positions
        for obs in self.obstacles:
            c = Circle(obs.position[:2], radius=np.max(obs.size)/2, color='gray', alpha=0.8)
            ax.add_patch(c)
            ax.text(obs.position[0], obs.position[1], f"Obs {obs.id}", ha='center', va='center')

        # Plot re-planning events
        for t, pos, event in self.history['events']:
            ax.plot(pos[0], pos[1], 'y*', markersize=15, label=f"{event} @ {t:.1f}s")

        ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    sim = FullAutonomySimulation()
    sim.run()
