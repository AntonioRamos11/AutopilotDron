#!/usr/bin/env python3
"""
Trajectory Accuracy Analysis for Drone Navigation

This script simulates and analyzes the accuracy of drone trajectory tracking
by comparing planned trajectories against simulated actual paths with various
levels of disturbance. It helps evaluate how well the drone will track the
minimum snap trajectory in real-world conditions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.distance import euclidean
import argparse

# Add path to import the MinSnapTrajectory module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'path_planning'))
from min_snap_trajectory import MinSnapTrajectory, Trajectory

class DroneSimulator:
    """Simulates drone flight with PID control and disturbances."""
    
    def __init__(self, mass=1.5, pid_gains=None, disturbance_levels=None):
        """
        Initialize the drone simulator.
        
        Args:
            mass: Drone mass in kg
            pid_gains: Dict with PID gains {kp, ki, kd}
            disturbance_levels: Dict with disturbance levels {position, velocity}
        """
        self.mass = float(mass)
        
        # Default PID gains if not provided
        if pid_gains is None:
            self.pid_gains = {
                'kp': np.array([2.0, 2.0, 2.0]),  # Position proportional gain
                'ki': np.array([0.1, 0.1, 0.1]),  # Position integral gain
                'kd': np.array([0.5, 0.5, 0.5])   # Position derivative gain
            }
        else:
            self.pid_gains = pid_gains
            
        # Default disturbance levels if not provided
        if disturbance_levels is None:
            self.disturbance_levels = {
                'position': 0.05,  # Position disturbance in meters (std dev)
                'velocity': 0.2    # Velocity disturbance in m/s (std dev)
            }
        else:
            self.disturbance_levels = disturbance_levels
            
        # Initialize state - ensure all are float arrays
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.integral_error = np.zeros(3, dtype=np.float64)
        self.prev_error = np.zeros(3, dtype=np.float64)
        
    def reset(self, initial_position=None):
        """Reset the simulator state."""
        if initial_position is not None:
            self.position = np.array(initial_position, dtype=np.float64)
        else:
            self.position = np.zeros(3, dtype=np.float64)
            
        self.velocity = np.zeros(3, dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.integral_error = np.zeros(3, dtype=np.float64)
        self.prev_error = np.zeros(3, dtype=np.float64)
        
    def update(self, target_position, target_velocity, dt):
        """
        Update the drone state based on control inputs and disturbances.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_velocity: Target velocity [vx, vy, vz] in m/s
            dt: Time step in seconds
            
        Returns:
            Updated position, velocity, and acceleration
        """
        # Calculate position error
        position_error = target_position - self.position
        
        # Update integral and derivative terms
        self.integral_error += position_error * dt
        
        # Anti-windup for integral term - limit the integral error
        max_integral = 1.0  # Maximum integral term to prevent windup
        for i in range(3):
            if abs(self.integral_error[i]) > max_integral:
                self.integral_error[i] = max_integral * np.sign(self.integral_error[i])
        
        derivative_error = (position_error - self.prev_error) / dt if dt > 0 else np.zeros(3)
        self.prev_error = position_error.copy()
        
        # PID control for acceleration command
        accel_cmd = (self.pid_gains['kp'] * position_error + 
                     self.pid_gains['ki'] * self.integral_error +
                     self.pid_gains['kd'] * derivative_error)
        
        # Add velocity feedforward term (better prediction of required acceleration)
        accel_cmd += 1.0 * (target_velocity - self.velocity)
        
        # Limit maximum acceleration (realistic limit for quadcopters)
        max_accel = 10.0  # m/s^2
        accel_norm = np.linalg.norm(accel_cmd)
        if accel_norm > max_accel:
            accel_cmd = accel_cmd * (max_accel / accel_norm)
        
        # Add disturbances
        position_disturbance = np.random.normal(
            0, self.disturbance_levels['position'], size=3)
        velocity_disturbance = np.random.normal(
            0, self.disturbance_levels['velocity'], size=3)
        
        # Update acceleration, velocity, and position
        self.acceleration = accel_cmd + velocity_disturbance / self.mass
        self.velocity += self.acceleration * dt
        
        # Limit maximum velocity (realistic limit for high-speed drones)
        max_vel = 20.0  # m/s
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > max_vel:
            self.velocity = self.velocity * (max_vel / vel_norm)
            
        self.position += self.velocity * dt + position_disturbance
        
        return self.position.copy(), self.velocity.copy(), self.acceleration.copy()


class TrajectoryAccuracySimulation:
    """Simulates and analyzes trajectory tracking accuracy."""
    
    def __init__(self, waypoints, segment_times=None, rtk_enabled=True, wind_strength=0.0, sim_options=None):
        """
        Initialize the trajectory accuracy simulation.
        
        Args:
            waypoints: List of waypoints [[x1, y1, z1], [x2, y2, z2], ...]
            segment_times: List of times for each segment [t1, t2, ...] or None for auto
            rtk_enabled: Whether RTK GPS is enabled (affects accuracy)
            wind_strength: Wind disturbance strength in m/s (0.0 = no wind)
            sim_options: Dictionary of simulation options (e.g., dt, duration_factor)
        """
        self.waypoints = np.array(waypoints)
        self.rtk_enabled = rtk_enabled
        self.wind_strength = wind_strength
        
        # Create trajectory generator
        self.traj_generator = MinSnapTrajectory()
        
        # Create drone simulator with appropriate disturbance levels
        # RTK GPS provides ~1cm accuracy, while standard GPS is ~50-100cm
        disturbance_levels = {
            'position': 0.01 if rtk_enabled else 0.5,  # RTK: ~1cm, Normal GPS: ~50cm
            'velocity': 0.01 if rtk_enabled else 0.3   # Lower velocity disturbance with RTK
        }
        
        # PID gains significantly tuned for RTK precision
        pid_gains = {
            'kp': np.array([15.0, 15.0, 15.0]) if rtk_enabled else np.array([2.0, 2.0, 2.0]),
            'ki': np.array([0.8, 0.8, 0.8]) if rtk_enabled else np.array([0.1, 0.1, 0.1]), 
            'kd': np.array([5.0, 5.0, 5.0]) if rtk_enabled else np.array([0.5, 0.5, 0.5])
        }
        
        self.drone_sim = DroneSimulator(disturbance_levels=disturbance_levels, pid_gains=pid_gains)
        
        # Auto-generate segment times based on distance if not provided
        if segment_times is None:
            desired_speed = 15.0 if rtk_enabled else 10.0  # m/s
            segment_times = []
            
            for i in range(len(self.waypoints) - 1):
                distance = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
                time = max(distance / desired_speed, 0.1)
                segment_times.append(time)
                
        self.segment_times = np.array(segment_times)
        
        # Generate the trajectory
        start_vel = np.zeros(3)
        end_vel = np.zeros(3)
        self.trajectory = self.traj_generator.generate_trajectory(
            self.waypoints,
            self.segment_times,
            start_vel=start_vel,
            end_vel=end_vel
        )
        
        # Simulation results
        self.results = {
            'time': [],
            'planned_position': [],
            'actual_position': [],
            'position_error': [],
            'velocity_error': []
        }
        
        # Initialize wind parameters if enabled
        self.wind_direction = np.random.uniform(0, 2*np.pi)  # Random wind direction
        self.wind_variation_freq = 0.1  # Hz - how quickly wind changes
        self.wind_turbulence = 0.2      # Turbulence intensity

        # Set simulation options
        self.dt = 0.02  # Default time step: 50 Hz
        self.duration_factor = 1.0  # Default duration factor (1.0 = real time)
        
        if sim_options is not None:
            for key in sim_options:
                setattr(self, key, sim_options[key])
                
    def run_simulation(self, dt=0.02):
        """
        Run the trajectory tracking simulation.
        
        Args:
            dt: Time step in seconds (default: 0.02s = 50Hz)
        """
        # Reset the drone simulator
        self.drone_sim.reset(initial_position=self.waypoints[0])
        
        # Clear previous results
        for key in self.results:
            self.results[key] = []
            
        # Run simulation
        t = 0
        while t <= self.trajectory.total_time:
            # Get planned position and velocity from trajectory
            planned_position = self.trajectory.evaluate(t)
            planned_velocity = self.trajectory.evaluate(t, derivative_order=1)
            
            # Apply wind disturbance if enabled
            if self.wind_strength > 0:
                # Calculate time-varying wind direction
                wind_dir_variation = 0.2 * np.sin(2 * np.pi * self.wind_variation_freq * t)
                current_wind_dir = self.wind_direction + wind_dir_variation
                
                # Calculate wind vector (2D horizontal wind)
                wind_speed = self.wind_strength * (1 + self.wind_turbulence * np.sin(5 * t))
                wind_velocity = np.array([
                    wind_speed * np.cos(current_wind_dir),
                    wind_speed * np.sin(current_wind_dir),
                    0.1 * wind_speed * np.sin(2.5 * t)  # Small vertical component
                ])
                
                # Add wind effect to drone velocity
                self.drone_sim.velocity += wind_velocity * dt * 0.1  # Reduced effect for stability
            
            # Update drone simulator
            actual_position, actual_velocity, _ = self.drone_sim.update(
                planned_position, planned_velocity, dt)
            
            # Calculate errors
            position_error = np.linalg.norm(planned_position - actual_position)
            velocity_error = np.linalg.norm(planned_velocity - actual_velocity)
            
            # Store results
            self.results['time'].append(t)
            self.results['planned_position'].append(planned_position.copy())
            self.results['actual_position'].append(actual_position.copy())
            self.results['position_error'].append(position_error)
            self.results['velocity_error'].append(velocity_error)
            
            # Increment time
            t += dt
            
        # Convert lists to numpy arrays
        for key in self.results:
            if key in ['planned_position', 'actual_position']:
                self.results[key] = np.array(self.results[key])
            else:
                self.results[key] = np.array(self.results[key])
                
    def analyze_results(self):
        """
        Analyze the simulation results.
        
        Returns:
            Dictionary with analysis metrics
        """
        position_errors = np.array(self.results['position_error'])
        
        analysis = {
            'mean_position_error': np.mean(position_errors),
            'max_position_error': np.max(position_errors),
            'std_position_error': np.std(position_errors),
            'rmse_position': np.sqrt(np.mean(position_errors**2)),
            'percentage_under_10cm': np.mean(position_errors < 0.1) * 100,
            'percentage_under_5cm': np.mean(position_errors < 0.05) * 100,
            'rtk_enabled': self.rtk_enabled
        }
        
        return analysis
        
    def plot_results(self, show_animation=False, save_path=None):
        """
        Plot the simulation results with enhanced visualization.
        
        Args:
            show_animation: Whether to show an animation of the trajectory
            save_path: Path to save the plots to (None for no saving)
        """
        # Create figure for 3D plot with enhanced appearance
        plt.style.use('ggplot')  # Use a more visually appealing style
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot with improved appearance
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # Plot the planned trajectory with smoother appearance
        ax1.plot(self.results['planned_position'][:, 0],
                self.results['planned_position'][:, 1],
                self.results['planned_position'][:, 2],
                'b-', linewidth=2.5, label='Planned')
        
        # Plot the actual trajectory with color gradient based on error
        points = np.array([self.results['actual_position'][:, 0],
                          self.results['actual_position'][:, 1],
                          self.results['actual_position'][:, 2]]).T
        
        # Normalize errors for coloring
        errors = self.results['position_error']
        max_error = np.max(errors) if np.max(errors) > 0 else 1.0
        norm = plt.Normalize(0, max_error)
        
        # Create a color map that shows error magnitude (red = high error, green = low error)
        for i in range(len(points)-1):
            ax1.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2],
                    color=plt.cm.RdYlGn_r(norm(errors[i])), linewidth=2.5)
        
        # Add a colorbar to show error scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, pad=0.1)
        cbar.set_label('Position Error [m]')
        
        # Mark waypoints more clearly
        ax1.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2],
                   c='k', marker='o', s=100, label='Waypoints', edgecolors='w')
        
        # Improve the appearance with a grid and better labeling
        ax1.grid(True)
        ax1.set_xlabel('X [m]', fontweight='bold')
        ax1.set_ylabel('Y [m]', fontweight='bold')
        ax1.set_zlabel('Z [m]', fontweight='bold')
        ax1.set_title('3D Trajectory Visualization', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        # Position error over time with improved styling
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Plot error with gradient fill below to highlight areas of concern
        times = np.array(self.results['time'])
        errors = np.array(self.results['position_error'])
        
        ax2.plot(times, errors, 'b-', linewidth=2, label='Position Error')
        ax2.fill_between(times, errors, alpha=0.3, color='blue')
        
        # Add threshold lines for reference
        rtk_status = "RTK Enabled" if self.rtk_enabled else "Standard GPS"
        err_threshold = 0.05 if self.rtk_enabled else 0.5  # Adjust threshold based on GPS type
        
        ax2.axhline(y=0.01, color='g', linestyle='-', linewidth=2, alpha=0.7, label='1cm Error')
        ax2.axhline(y=0.05, color='y', linestyle='-', linewidth=2, alpha=0.7, label='5cm Error')
        ax2.axhline(y=0.10, color='r', linestyle='-', linewidth=2, alpha=0.7, label='10cm Error')
        
        # Highlight areas where error exceeds threshold
        ax2.fill_between(times, errors, err_threshold, 
                        where=(errors > err_threshold),
                        color='red', alpha=0.3, interpolate=True)
        
        ax2.set_xlabel('Time [s]', fontweight='bold')
        ax2.set_ylabel('Position Error [m]', fontweight='bold')
        ax2.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right', frameon=True)
        
        # Position error histogram with improved styling
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Use better binning and styling for histogram
        n, bins, patches = ax3.hist(self.results['position_error'], bins=30, 
                                   color='skyblue', edgecolor='black', alpha=0.7)
        
        # Color bins based on error magnitude
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col) if max(col) > 0 else 1.0
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', plt.cm.RdYlGn_r(c))
        
        # Add a vertical line at the mean error
        mean_error = np.mean(self.results['position_error'])
        ax3.axvline(x=mean_error, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean Error: {mean_error:.3f}m')
        
        # Add vertical lines at key thresholds
        ax3.axvline(x=0.01, color='g', linestyle='-', linewidth=2, alpha=0.7, label='1cm')
        ax3.axvline(x=0.05, color='y', linestyle='-', linewidth=2, alpha=0.7, label='5cm')
        ax3.axvline(x=0.10, color='r', linestyle='-', linewidth=2, alpha=0.7, label='10cm')
        
        ax3.set_xlabel('Position Error [m]', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Position Error Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right', frameon=True)
        
        # XY plot (top-down view) with improved styling
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Plot the planned path
        ax4.plot(self.results['planned_position'][:, 0],
                self.results['planned_position'][:, 1],
                'b-', linewidth=2.5, label='Planned')
        
        # Plot the actual path with color gradient showing the error at each point
        points = np.array([self.results['actual_position'][:, 0], 
                          self.results['actual_position'][:, 1]]).T
        
        # Create a color map based on error
        for i in range(len(points)-1):
            ax4.plot(points[i:i+2, 0], points[i:i+2, 1],
                    color=plt.cm.RdYlGn_r(norm(errors[i])), linewidth=2.5)
        
        # Add waypoint markers
        ax4.scatter(self.waypoints[:, 0], self.waypoints[:, 1],
                   c='k', marker='o', s=100, label='Waypoints', edgecolors='w')
        
        # Add start and end markers
        ax4.scatter(self.waypoints[0, 0], self.waypoints[0, 1],
                   c='g', marker='*', s=200, label='Start', edgecolors='w')
        ax4.scatter(self.waypoints[-1, 0], self.waypoints[-1, 1],
                   c='r', marker='*', s=200, label='End', edgecolors='w')
        
        # Add a grid, equal aspect ratio, and labels
        ax4.set_xlabel('X [m]', fontweight='bold')
        ax4.set_ylabel('Y [m]', fontweight='bold')
        ax4.set_title('Top-Down View', fontsize=14, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.axis('equal')
        ax4.legend(loc='upper right', frameon=True)
        
        # Add analysis text with better formatting
        analysis = self.analyze_results()
        rtk_status = "RTK Enabled" if analysis['rtk_enabled'] else "Standard GPS"
        fig.suptitle(f'Trajectory Accuracy Analysis ({rtk_status})', fontsize=16, fontweight='bold')
        
        analysis_text = (
            f"Mean Error: {analysis['mean_position_error']:.3f} m\n"
            f"Max Error: {analysis['max_position_error']:.3f} m\n"
            f"RMSE: {analysis['rmse_position']:.3f} m\n"
            f"% under 10cm: {analysis['percentage_under_10cm']:.1f}%\n"
            f"% under 5cm: {analysis['percentage_under_5cm']:.1f}%\n"
            f"% under 1cm: {np.mean(self.results['position_error'] < 0.01) * 100:.1f}%"
        )
        
        fig.text(0.02, 0.02, analysis_text, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='gray'))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show_animation:
            plt.draw()  # Show this figure first
            plt.pause(0.1)  # Small pause to ensure the figure is shown
            self._animate_trajectory(save_animation_path=save_path.replace('.png', '_animation.gif') if save_path else None)
        else:
            plt.show()
            
    def _animate_trajectory(self, save_animation_path=None):
        """
        Create and show an enhanced animation of the trajectory.
        
        Args:
            save_animation_path: Path to save the animation to (None for no saving)
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits with some padding
        min_vals = np.min(np.vstack([self.results['planned_position'], 
                                    self.results['actual_position']]), axis=0) - 0.2
        max_vals = np.max(np.vstack([self.results['planned_position'], 
                                    self.results['actual_position']]), axis=0) + 0.2
        
        ax.set_xlim([min_vals[0], max_vals[0]])
        ax.set_ylim([min_vals[1], max_vals[1]])
        ax.set_zlim([min_vals[2], max_vals[2]])
        
        # Plot the complete planned trajectory as reference
        ax.plot(self.results['planned_position'][:, 0],
               self.results['planned_position'][:, 1],
               self.results['planned_position'][:, 2],
               'b-', linewidth=1.5, alpha=0.5, label='Planned Path')
        
        # Plot waypoints
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2],
                  c='k', marker='o', s=100, label='Waypoints', edgecolors='w')
        
        # Initialize drone position and trail
        drone_pos = ax.scatter([], [], [], c='r', marker='o', s=150, edgecolors='k',
                              label='Drone Position')
        
        # Initialize trajectory trails
        planned_trail, = ax.plot([], [], [], 'b-', linewidth=2, label='Planned')
        actual_trail, = ax.plot([], [], [], 'r-', linewidth=2, label='Actual')
        
        # Add time display
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add error display
        error_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=12,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Set labels and title
        ax.set_xlabel('X [m]', fontweight='bold')
        ax.set_ylabel('Y [m]', fontweight='bold')
        ax.set_zlabel('Z [m]', fontweight='bold')
        ax.set_title('Drone Trajectory Animation', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        # Make the plot look nicer
        ax.grid(True)
        fig.tight_layout()
        
        # Create a panel for additional stats
        stats_ax = fig.add_axes([0.15, 0.02, 0.7, 0.08])  # [left, bottom, width, height]
        stats_ax.axis('off')
        
        # Add current velocity display
        vel_text = stats_ax.text(0.5, 0.5, '', fontsize=14, ha='center',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        rtk_status = "RTK Enabled" if self.rtk_enabled else "Standard GPS"
        fig.suptitle(f'Drone Trajectory Animation ({rtk_status})', fontsize=16, fontweight='bold')
        
        # Trail length (how many previous points to show)
        trail_length = 100
        
        def update(frame):
            """Update function for animation."""
            if frame >= len(self.results['time']):
                frame = len(self.results['time']) - 1
                
            # Set start of trail (don't go negative)
            trail_start = max(0, frame - trail_length)
                
            # Update planned trajectory trail
            planned_trail.set_data(self.results['planned_position'][trail_start:frame+1, 0], 
                                  self.results['planned_position'][trail_start:frame+1, 1])
            planned_trail.set_3d_properties(self.results['planned_position'][trail_start:frame+1, 2])
            
            # Update actual trajectory trail
            actual_trail.set_data(self.results['actual_position'][trail_start:frame+1, 0], 
                                 self.results['actual_position'][trail_start:frame+1, 1])
            actual_trail.set_3d_properties(self.results['actual_position'][trail_start:frame+1, 2])
            
            # Update drone position
            drone_pos._offsets3d = ([self.results['actual_position'][frame, 0]], 
                                   [self.results['actual_position'][frame, 1]], 
                                   [self.results['actual_position'][frame, 2]])
            
            # Update time text
            time_text.set_text(f'Time: {self.results["time"][frame]:.2f}s')
            
            # Update error text
            error_text.set_text(f'Position Error: {self.results["position_error"][frame]:.3f}m')
            
            # Calculate current velocity magnitude
            if frame > 0:
                dt = self.results["time"][frame] - self.results["time"][frame-1]
                if dt > 0:
                    dx = self.results["actual_position"][frame] - self.results["actual_position"][frame-1]
                    velocity = np.linalg.norm(dx) / dt
                    vel_text.set_text(f'Speed: {velocity:.2f} m/s | ' +
                                     f'Progress: {int(100 * frame / len(self.results["time"]))}%')
            
            # Return updated artists
            return planned_trail, actual_trail, drone_pos, time_text, error_text, vel_text
            
        # Create animation with smoother playback
        frames = len(self.results['time'])
        step = max(1, frames // 100)  # Reduce frames for smoother saving
        num_frames = len(range(0, frames, step))
        
        print(f"Generating animation with {num_frames} frames...")
        ani = FuncAnimation(fig, update, frames=range(0, frames, step),
                           interval=50, blit=True)
        
        if save_animation_path:
            try:
                # Save as GIF if save_animation_path ends with .gif
                if save_animation_path.endswith('.gif'):
                    print(f"Saving animation as GIF to {save_animation_path}...")
                    ani.save(save_animation_path, writer='pillow', fps=15, 
                            dpi=100, progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'))
                    print(f"Animation saved to: {save_animation_path}")
                # Otherwise save as MP4
                else:
                    mp4_path = save_animation_path.replace('.gif', '.mp4')
                    print(f"Saving animation as MP4 to {mp4_path}...")
                    ani.save(mp4_path, writer='ffmpeg', fps=15, dpi=100, 
                           extra_args=['-vcodec', 'libx264'], progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'))
                    print(f"Animation saved to: {mp4_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Continuing with regular display...")
        
        # Use HTML5 video for smoother playback if supported
        try:
            plt.rcParams['animation.html'] = 'html5'
        except:
            pass
            
        plt.show()
        
    def create_video(self, filename='trajectory_video.mp4', fps=30):
        """
        Create a high-quality video visualization of the trajectory.
        
        Args:
            filename: Output video filename
            fps: Frames per second for the video
        """
        import matplotlib.animation as animation
        
        # Create figure with 3D perspective
        fig = plt.figure(figsize=(12, 9), dpi=150)
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits with some padding
        min_vals = np.min(np.vstack([self.results['planned_position'], 
                                   self.results['actual_position']]), axis=0) - 0.2
        max_vals = np.max(np.vstack([self.results['planned_position'], 
                                   self.results['actual_position']]), axis=0) + 0.2
        
        ax.set_xlim([min_vals[0], max_vals[0]])
        ax.set_ylim([min_vals[1], max_vals[1]])
        ax.set_zlim([min_vals[2], max_vals[2]])
        
        # Improve plot appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('X [m]', fontweight='bold')
        ax.set_ylabel('Y [m]', fontweight='bold')
        ax.set_zlabel('Z [m]', fontweight='bold')
        ax.set_title('Drone Trajectory Simulation', fontsize=14, fontweight='bold')
        
        # Create background elements that don't change
        # Plot complete planned trajectory
        ax.plot(self.results['planned_position'][:, 0],
              self.results['planned_position'][:, 1],
              self.results['planned_position'][:, 2],
              'b-', linewidth=1.5, alpha=0.6, label='Planned')
        
        # Plot waypoints
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2],
                 c='k', marker='o', s=100, label='Waypoints', edgecolors='w')
        
        # Initialize dynamic elements
        drone_marker = ax.scatter([], [], [], c='r', marker='o', s=150, edgecolors='k',
                                label='Drone Position')
        actual_trail, = ax.plot([], [], [], 'r-', linewidth=2.5, label='Actual')
        
        # Add error indicator (line from drone to planned position)
        error_line, = ax.plot([], [], [], 'g--', linewidth=1.5, alpha=0.7)
        
        # Information displays
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        error_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=12,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        speed_text = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=12,
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add color-coded rings to represent error magnitude
        error_ring = None
        
        # Set up statistics panel
        stats_ax = fig.add_axes([0.7, 0.05, 0.25, 0.2])  # [left, bottom, width, height]
        stats_ax.axis('off')
        
        analysis = self.analyze_results()
        stats_str = (f"Mean Error: {analysis['mean_position_error']:.3f} m\n"
                    f"Max Error: {analysis['max_position_error']:.3f} m\n"
                    f"RMSE: {analysis['rmse_position']:.3f} m\n"
                    f"% under 10cm: {analysis['percentage_under_10cm']:.1f}%\n"
                    f"% under 5cm: {analysis['percentage_under_5cm']:.1f}%\n"
                    f"RTK: {'Enabled' if self.rtk_enabled else 'Disabled'}")
        
        stats_text = stats_ax.text(0, 0.5, stats_str, fontsize=12, va='center',
                                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5', 
                                          edgecolor='gray'))
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        # Trail length (how many previous points to show)
        trail_length = 100
        
        # Define update function for animation frames
        def update(frame):
            """Update function for animation."""
            if frame >= len(self.results['time']):
                frame = len(self.results['time']) - 1
                
            # Get current positions
            actual_pos = self.results['actual_position'][frame]
            planned_pos = self.results['planned_position'][frame]
            
            # Set start of trail (don't go negative)
            trail_start = max(0, frame - trail_length)
                
            # Update actual trail
            actual_trail.set_data(self.results['actual_position'][trail_start:frame+1, 0], 
                                self.results['actual_position'][trail_start:frame+1, 1])
            actual_trail.set_3d_properties(self.results['actual_position'][trail_start:frame+1, 2])
            
            # Update drone position
            drone_marker._offsets3d = ([actual_pos[0]], [actual_pos[1]], [actual_pos[2]])
            
            # Update error line (connection from drone to planned position)
            error_line.set_data([actual_pos[0], planned_pos[0]], [actual_pos[1], planned_pos[1]])
            error_line.set_3d_properties([actual_pos[2], planned_pos[2]])
            
            # Update information displays
            time_text.set_text(f"Time: {self.results['time'][frame]:.2f}s")
            error_text.set_text(f"Error: {self.results['position_error'][frame]:.3f}m")
            
            # Calculate and display current speed
            if frame > 0:
                dt = self.results["time"][frame] - self.results["time"][frame-1]
                if dt > 0:
                    dx = self.results["actual_position"][frame] - self.results["actual_position"][frame-1]
                    speed = np.linalg.norm(dx) / dt
                    speed_text.set_text(f"Speed: {speed:.2f} m/s")
            
            return drone_marker, actual_trail, error_line, time_text, error_text, speed_text
        
        # Create animation
        print(f"Generating video with {len(self.results['time'])} frames at {fps} fps...")
        anim = animation.FuncAnimation(fig, update, frames=len(self.results['time']),
                                      interval=1000/fps, blit=True)
        
        # Add a progress bar using tqdm if available
        try:
            from tqdm import tqdm
            progress_callback = lambda i, n: tqdm(total=n, initial=i, desc="Rendering video")
        except ImportError:
            progress_callback = lambda i, n: print(f"Rendering frame {i}/{n}")
        
        # Save the animation with high quality
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=150,
                 extra_args=['-vcodec', 'libx264', '-crf', '18'],  # High quality encoding
                 progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"Video saved to: {filename}")
    
    def export_results(self, filename='trajectory_analysis.csv'):
        """
        Export simulation results to CSV.
        
        Args:
            filename: Output filename
        """
        # Create a pandas DataFrame from the results
        df = pd.DataFrame({
            'time': self.results['time'],
            'planned_x': self.results['planned_position'][:, 0],
            'planned_y': self.results['planned_position'][:, 1],
            'planned_z': self.results['planned_position'][:, 2],
            'actual_x': self.results['actual_position'][:, 0],
            'actual_y': self.results['actual_position'][:, 1],
            'actual_z': self.results['actual_position'][:, 2],
            'position_error': self.results['position_error'],
            'velocity_error': self.results['velocity_error']
        })
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        
    def create_comparison_video(self, other_sim, filename='comparison_video.mp4', fps=30):
        """
        Create a side-by-side comparison video of two simulations (RTK vs standard GPS).
        
        Args:
            other_sim: Another TrajectoryAccuracySimulation instance to compare with
            filename: Output video filename
            fps: Frames per second for the video
        """
        import matplotlib.animation as animation
        
        # Make sure the number of time steps match
        min_steps = min(len(self.results['time']), len(other_sim.results['time']))
        
        # Use more frames for smoother video - don't reduce frames too aggressively
        # Calculate frame step to give at least 300 frames or use every frame if less than 300
        total_frames = min(300, min_steps)
        frame_step = max(1, min_steps // total_frames)
        
        fig = plt.figure(figsize=(16, 8))
        
        # Setup RTK plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlim([self.waypoints[:, 0].min() - 0.2, self.waypoints[:, 0].max() + 0.2])
        ax1.set_ylim([self.waypoints[:, 1].min() - 0.2, self.waypoints[:, 1].max() + 0.2])
        ax1.set_zlim([self.waypoints[:, 2].min() - 0.2, self.waypoints[:, 2].max() + 0.2])
        
        # Plot complete planned trajectory for reference
        ax1.plot(self.results['planned_position'][:, 0],
                self.results['planned_position'][:, 1],
                self.results['planned_position'][:, 2],
                'b-', linewidth=1.5, alpha=0.5)
                
        # Plot waypoints
        ax1.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2],
                   c='k', marker='o', s=100, label='Waypoints')
                   
        # Initialize RTK drone
        rtk_drone = ax1.scatter([], [], [], c='g', marker='o', s=150, edgecolors='k', label='Drone (RTK)')
        rtk_trail, = ax1.plot([], [], [], 'g-', linewidth=2, label='Actual')
        
        ax1.set_title('RTK GPS Trajectory', fontsize=14)
        ax1.set_xlabel('X [m]', fontweight='bold')
        ax1.set_ylabel('Y [m]', fontweight='bold')
        ax1.set_zlabel('Z [m]', fontweight='bold')
        ax1.legend(loc='upper right')
        
        # Setup Standard GPS plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlim([self.waypoints[:, 0].min() - 0.2, self.waypoints[:, 0].max() + 0.2])
        ax2.set_ylim([self.waypoints[:, 1].min() - 0.2, self.waypoints[:, 1].max() + 0.2])
        ax2.set_zlim([self.waypoints[:, 2].min() - 0.2, self.waypoints[:, 2].max() + 0.2])
        
        # Plot complete planned trajectory for reference
        ax2.plot(other_sim.results['planned_position'][:, 0],
                other_sim.results['planned_position'][:, 1],
                other_sim.results['planned_position'][:, 2],
                'b-', linewidth=1.5, alpha=0.5)
                
        # Plot waypoints
        ax2.scatter(other_sim.waypoints[:, 0], other_sim.waypoints[:, 1], other_sim.waypoints[:, 2],
                   c='k', marker='o', s=100, label='Waypoints')
                   
        # Initialize standard GPS drone
        std_drone = ax2.scatter([], [], [], c='r', marker='o', s=150, edgecolors='k', label='Drone (STD)')
        std_trail, = ax2.plot([], [], [], 'r-', linewidth=2, label='Actual')
        
        ax2.set_title('Standard GPS Trajectory', fontsize=14)
        ax2.set_xlabel('X [m]', fontweight='bold')
        ax2.set_ylabel('Y [m]', fontweight='bold')
        ax2.set_zlabel('Z [m]', fontweight='bold')
        ax2.legend(loc='upper right')
        
        # Add time and stats display
        time_text = fig.text(0.5, 0.95, '', fontsize=14, ha='center',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        rtk_error_text = fig.text(0.25, 0.03, '', fontsize=12, ha='center',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        std_error_text = fig.text(0.75, 0.03, '', fontsize=12, ha='center',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        fig.suptitle('RTK vs Standard GPS Comparison', fontsize=16, fontweight='bold')
        fig.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        # Set up trail length (number of points to show in trail)
        trail_length = 100
        
        def update(frame_idx):
            """Update function for the animation."""
            # Convert frame index to actual data index
            frame = frame_idx * frame_step
            if frame >= min_steps:
                frame = min_steps - 1
                
            # Calculate trail start (don't go negative)
            trail_start = max(0, frame - trail_length)
            
            # Update RTK drone
            rtk_drone._offsets3d = ([self.results['actual_position'][frame, 0]],
                                   [self.results['actual_position'][frame, 1]],
                                   [self.results['actual_position'][frame, 2]])
                                   
            # Update RTK trail
            rtk_trail.set_data(self.results['actual_position'][trail_start:frame+1, 0],
                              self.results['actual_position'][trail_start:frame+1, 1])
            rtk_trail.set_3d_properties(self.results['actual_position'][trail_start:frame+1, 2])
            
            # Update standard GPS drone
            std_drone._offsets3d = ([other_sim.results['actual_position'][frame, 0]],
                                   [other_sim.results['actual_position'][frame, 1]],
                                   [other_sim.results['actual_position'][frame, 2]])
                                   
            # Update standard GPS trail
            std_trail.set_data(other_sim.results['actual_position'][trail_start:frame+1, 0],
                              other_sim.results['actual_position'][trail_start:frame+1, 1])
            std_trail.set_3d_properties(other_sim.results['actual_position'][trail_start:frame+1, 2])
            
            # Update time text
            if frame < len(self.results['time']):
                time_text.set_text(f'Time: {self.results["time"][frame]:.2f}s')
                
            # Update error texts
            rtk_error = self.results['position_error'][frame]
            std_error = other_sim.results['position_error'][frame]
            
            rtk_error_text.set_text(f'RTK Error: {rtk_error:.3f}m')
            std_error_text.set_text(f'Standard GPS Error: {std_error:.3f}m')
            
            # Add camera rotation for more dynamic view
            ax1.view_init(elev=30, azim=frame_idx % 360)
            ax2.view_init(elev=30, azim=frame_idx % 360)
            
            return rtk_drone, rtk_trail, std_drone, std_trail, time_text, rtk_error_text, std_error_text
        
        # Create animation with increased frame count
        frame_count = min_steps // frame_step
        print(f"Generating comparison video with {frame_count} frames...")
        
        ani = animation.FuncAnimation(fig, update, frames=frame_count,
                                     interval=1000/fps, blit=True)
        
        # Try to use tqdm for progress reporting if available
        try:
            from tqdm import tqdm
            progress_callback = lambda i, n: tqdm(total=n, initial=i, desc="Rendering video frames")
        except ImportError:
            progress_callback = lambda i, n: print(f"Saving frame {i}/{n}")
        
        # Save the animation with high quality settings
        ani.save(filename, writer='ffmpeg', fps=fps, dpi=200, 
                extra_args=['-vcodec', 'libx264', '-crf', '18'],  # High quality codec settings
                progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"Comparison video saved to: {filename}")
    
    def interactive_pid_tuning(self, waypoints):
        """
        Provide an interactive interface for tuning PID parameters.
        
        Args:
            waypoints: Array of waypoints to use for simulation
        """
        from matplotlib.widgets import Slider, Button
        
        # Create figure for interactive tuning
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(bottom=0.35)
        
        # Plot empty trajectory initially
        line_planned, = ax1.plot([], [], 'b-', linewidth=2, label='Planned')
        line_actual, = ax1.plot([], [], 'r-', linewidth=2, label='Actual')
        scatter_waypoints = ax1.scatter(waypoints[:, 0], waypoints[:, 1], c='k', marker='o', s=100, label='Waypoints')
        
        # Plot empty errors initially
        line_error, = ax2.plot([], [], 'g-', linewidth=2, label='Position Error')
        
        # Configure axes
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Trajectory (Top View)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Error [m]')
        ax2.set_title('Position Error')
        ax2.grid(True)
        
        # Add sliders for PID parameters
        ax_kp = plt.axes([0.25, 0.22, 0.65, 0.03])
        ax_ki = plt.axes([0.25, 0.17, 0.65, 0.03])
        ax_kd = plt.axes([0.25, 0.12, 0.65, 0.03])
        ax_wind = plt.axes([0.25, 0.07, 0.65, 0.03])
        
        # Define slider ranges
        kp_slider = Slider(ax_kp, 'Kp', 0.1, 30.0, 
                          valinit=self.drone_sim.pid_gains['kp'][0], valstep=0.5)
        ki_slider = Slider(ax_ki, 'Ki', 0.0, 2.0, 
                          valinit=self.drone_sim.pid_gains['ki'][0], valstep=0.05)
        kd_slider = Slider(ax_kd, 'Kd', 0.0, 10.0, 
                          valinit=self.drone_sim.pid_gains['kd'][0], valstep=0.1)
        wind_slider = Slider(ax_wind, 'Wind', 0.0, 5.0, 
                           valinit=self.wind_strength, valstep=0.1)
        
        # Add button to run simulation
        ax_run = plt.axes([0.4, 0.01, 0.2, 0.04])
        run_button = Button(ax_run, 'Run Simulation')
        
        # Add text for statistics
        stats_text = plt.figtext(0.02, 0.02, '', fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.8))
        
        def run_simulation(event):
            """Run simulation with current PID parameters."""
            # Update PID gains from sliders
            kp_value = kp_slider.val
            ki_value = ki_slider.val
            kd_value = kd_slider.val
            
            # Update gains in simulator
            self.drone_sim.pid_gains = {
                'kp': np.array([kp_value, kp_value, kp_value]),
                'ki': np.array([ki_value, ki_value, ki_value]),
                'kd': np.array([kd_value, kd_value, kd_value])
            }
            
            # Update wind strength
            self.wind_strength = wind_slider.val
            
            # Run the simulation
            self.run_simulation()
            
            # Update plots
            update_plots()
        
        def update_plots():
            """Update plot with simulation results."""
            # Update trajectory plot
            line_planned.set_data(self.results['planned_position'][:, 0], 
                                 self.results['planned_position'][:, 1])
            line_actual.set_data(self.results['actual_position'][:, 0],
                                self.results['actual_position'][:, 1])
                                
            # Adjust axis limits
            all_x = np.concatenate([self.results['planned_position'][:, 0], 
                                  self.results['actual_position'][:, 0]])
            all_y = np.concatenate([self.results['planned_position'][:, 1], 
                                  self.results['actual_position'][:, 1]])
            margin = 0.1 * (max(all_x) - min(all_x))
            ax1.set_xlim([min(all_x) - margin, max(all_x) + margin])
            ax1.set_ylim([min(all_y) - margin, max(all_y) + margin])
            
            # Update error plot
            line_error.set_data(self.results['time'], self.results['position_error'])
            ax2.set_xlim([0, max(self.results['time'])])
            ax2.set_ylim([0, max(self.results['position_error']) * 1.1])
            
            # Update statistics
            analysis = self.analyze_results()
            stats_str = (f"Mean Error: {analysis['mean_position_error']:.3f} m\n"
                        f"Max Error: {analysis['max_position_error']:.3f} m\n"
                        f"RMSE: {analysis['rmse_position']:.3f} m\n"
                        f"% under 10cm: {analysis['percentage_under_10cm']:.1f}%")
            stats_text.set_text(stats_str)
            
            fig.canvas.draw_idle()
        
        # Connect the button to the function
        run_button.on_clicked(run_simulation)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trajectory Accuracy Simulation')
    parser.add_argument('--rtk', action='store_true', default=True,
                       help='Enable RTK GPS simulation (default: enabled)')
    parser.add_argument('--no-rtk', action='store_false', dest='rtk',
                       help='Disable RTK GPS simulation')
    parser.add_argument('--animate', action='store_true',
                       help='Show trajectory animation')
    parser.add_argument('--export', type=str, default='',
                       help='Export results to CSV file')
    parser.add_argument('--compare', action='store_true',
                       help='Compare RTK and standard GPS')
    parser.add_argument('--save', type=str, default='',
                       help='Save visualization to image file (png, jpg, pdf)')
    parser.add_argument('--save-animation', type=str, default='',
                       help='Save animation to gif or mp4 file')
    parser.add_argument('--no-display', action='store_true',
                       help='Run in headless mode without displaying plots')
    parser.add_argument('--video', type=str, default='',
                       help='Save high-quality video to mp4 file')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video export (default: 30)')
    parser.add_argument('--wind', type=float, default=0.0,
                       help='Add simulated wind disturbance of specified strength (m/s)')
    parser.add_argument('--tune-pid', action='store_true',
                       help='Enable interactive PID tuning during simulation')
    parser.add_argument('--scale', type=float, default=0.1,
                       help='Scale factor for waypoints (default: 0.1)')
    parser.add_argument('--duration', type=float, default=20.0, 
                       help='Duration of simulation in seconds (default: 20.0)')
    parser.add_argument('--trajectory', type=str, default='standard',
                       help='Trajectory type: standard, complex, circle, spiral (default: standard)')
    
    args = parser.parse_args()
    
    # Check if running in a headless environment (no display available)
    headless_mode = args.no_display
    try:
        # Try to create a test figure to see if display works
        if not headless_mode:
            test_fig = plt.figure()
            test_fig.close()
    except:
        print("No display detected. Running in headless mode and saving results to files.")
        headless_mode = True
    
    # If no save path is specified and we're in headless mode, create default paths
    if headless_mode and not args.save:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        if args.compare:
            args.save = f"trajectory_comparison_{timestamp}.png"
        else:
            gps_type = "rtk" if args.rtk else "std"
            args.save = f"trajectory_{gps_type}_{timestamp}.png"
        print(f"Auto-saving visualization to: {args.save}")
    
    # Define waypoints based on trajectory type
    if args.trajectory == 'standard':
        # Basic 5-waypoint trajectory
        waypoints = np.array([
            [0, 0, 0],       # Start
            [5, 5, 5],       # Waypoint 1
            [10, 0, 10],     # Waypoint 2
            [15, -5, 5],     # Waypoint 3
            [20, 0, 0]       # End
        ])
    elif args.trajectory == 'complex':
        # More complex trajectory with more waypoints
        waypoints = np.array([
            [0, 0, 0],        # Start
            [5, 5, 5],        # Waypoint 1
            [10, 0, 8],       # Waypoint 2
            [15, 5, 12],      # Waypoint 3
            [20, -5, 10],     # Waypoint 4
            [25, 0, 15],      # Waypoint 5
            [30, 10, 10],     # Waypoint 6
            [35, 5, 5],       # Waypoint 7
            [40, 0, 8],       # Waypoint 8
            [45, -5, 4],      # Waypoint 9
            [50, 0, 0]        # End
        ])
    elif args.trajectory == 'circle':
        # Circular trajectory
        t = np.linspace(0, 2*np.pi, 20)
        radius = 20
        waypoints = np.array([[radius * np.cos(theta), radius * np.sin(theta), 10 * np.sin(2*theta)] for theta in t])
        # Add start/end at origin
        waypoints = np.vstack(([0, 0, 0], waypoints, [0, 0, 0]))
    elif args.trajectory == 'spiral':
        # Spiral trajectory
        t = np.linspace(0, 4*np.pi, 30)
        radius_growth = 0.5
        waypoints = np.array([[t[i] * radius_growth * np.cos(t[i]), 
                             t[i] * radius_growth * np.sin(t[i]), 
                             t[i]] for i in range(len(t))])
        # Add start/end at origin
        waypoints = np.vstack(([0, 0, 0], waypoints, [0, 0, 0]))
    else:
        print(f"Unknown trajectory type '{args.trajectory}', using standard.")
        waypoints = np.array([
            [0, 0, 0],       # Start
            [5, 5, 5],       # Waypoint 1
            [10, 0, 10],     # Waypoint 2
            [15, -5, 5],     # Waypoint 3
            [20, 0, 0]       # End
        ])
    
    # Scale waypoints
    waypoints = waypoints * args.scale
    
    # Create simulation options with desired duration
    sim_options = {
        'dt': 0.02  # 50 Hz simulation
    }
    
    # Adjust segment times based on desired duration
    if args.duration != 20.0:
        sim_options['duration_factor'] = args.duration / 20.0
    
    if args.compare:
        # Run both RTK and standard GPS simulations for comparison
        rtk_sim = TrajectoryAccuracySimulation(waypoints, rtk_enabled=True, wind_strength=args.wind, 
                                              sim_options=sim_options)
        std_sim = TrajectoryAccuracySimulation(waypoints, rtk_enabled=False, wind_strength=args.wind,
                                              sim_options=sim_options)
        
        print("Running RTK simulation...")
        rtk_sim.run_simulation(dt=sim_options['dt'])
        rtk_analysis = rtk_sim.analyze_results()
        
        print("Running standard GPS simulation...")
        std_sim.run_simulation(dt=sim_options['dt'])
        std_analysis = std_sim.analyze_results()
        
        print("\nComparison Results:")
        print(f"{'Metric':<20} {'RTK':<15} {'Standard GPS':<15}")
        print("-" * 50)
        print(f"{'Mean Error:':<20} {rtk_analysis['mean_position_error']:.3f} m{'':<10} {std_analysis['mean_position_error']:.3f} m")
        print(f"{'Max Error:':<20} {rtk_analysis['max_position_error']:.3f} m{'':<10} {std_analysis['max_position_error']:.3f} m")
        print(f"{'RMSE:':<20} {rtk_analysis['rmse_position']:.3f} m{'':<10} {std_analysis['rmse_position']:.3f} m")
        print(f"{'% under 10cm:':<20} {rtk_analysis['percentage_under_10cm']:.1f}%{'':<10} {std_analysis['percentage_under_10cm']:.1f}%")
        print(f"{'% under 5cm:':<20} {rtk_analysis['percentage_under_5cm']:.1f}%{'':<10} {std_analysis['percentage_under_5cm']:.1f}%")
        print(f"{'% under 1cm:':<20} {np.mean(rtk_sim.results['position_error'] < 0.01) * 100:.1f}%{'':<10} {np.mean(std_sim.results['position_error'] < 0.01) * 100:.1f}%")
        print(f"{'Simulation time:':<20} {rtk_sim.results['time'][-1]:.1f}s{'':<10} {std_sim.results['time'][-1]:.1f}s")
        
        # Plot both results side-by-side
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('RTK vs Standard GPS Comparison', fontsize=16)
        
        # RTK 3D trajectory
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(rtk_sim.results['planned_position'][:, 0],
                rtk_sim.results['planned_position'][:, 1],
                rtk_sim.results['planned_position'][:, 2],
                'b-', linewidth=2, label='Planned')
        ax1.plot(rtk_sim.results['actual_position'][:, 0],
                rtk_sim.results['actual_position'][:, 1],
                rtk_sim.results['actual_position'][:, 2],
                'g-', linewidth=1.5, label='RTK')
        ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                   c='r', marker='o', s=100, label='Waypoints')
        ax1.set_title('RTK Trajectory')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.legend()
        
        # Standard GPS 3D trajectory
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot(std_sim.results['planned_position'][:, 0],
                std_sim.results['planned_position'][:, 1],
                std_sim.results['planned_position'][:, 2],
                'b-', linewidth=2, label='Planned')
        ax2.plot(std_sim.results['actual_position'][:, 0],
                std_sim.results['actual_position'][:, 1],
                std_sim.results['actual_position'][:, 2],
                'r-', linewidth=1.5, label='Standard GPS')
        ax2.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                   c='r', marker='o', s=100, label='Waypoints')
        ax2.set_title('Standard GPS Trajectory')
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_zlabel('Z [m]')
        ax2.legend()
        
        # Position error comparison
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(rtk_sim.results['time'], rtk_sim.results['position_error'], 'g-', label='RTK')
        ax3.plot(std_sim.results['time'], std_sim.results['position_error'], 'r-', label='Standard GPS')
        ax3.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='5cm Error')
        ax3.axhline(y=0.10, color='b', linestyle='--', alpha=0.5, label='10cm Error')
        ax3.set_title('Position Error Over Time')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Position Error [m]')
        ax3.legend()
        ax3.grid(True)
        
        # Error distribution comparison
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(rtk_sim.results['position_error'], bins=30, alpha=0.5, label='RTK', color='g')
        ax4.hist(std_sim.results['position_error'], bins=30, alpha=0.5, label='Standard GPS', color='r')
        ax4.set_title('Error Distribution Comparison')
        ax4.set_xlabel('Position Error [m]')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save comparison visualization
        if args.save:
            save_path = args.save
            if not any(save_path.endswith(ext) for ext in ['.png', '.jpg', '.pdf']):
                save_path = save_path + '.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to: {save_path}")
        
        # Only show if not in headless mode
        if not headless_mode:
            plt.show()
        else:
            plt.close()
        
        # Export results
        if args.export:
            rtk_sim.export_results('rtk_' + args.export)
            std_sim.export_results('std_' + args.export)
        else:
            # Auto-export in headless mode
            if headless_mode:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                rtk_sim.export_results(f'rtk_results_{timestamp}.csv')
                std_sim.export_results(f'std_gps_results_{timestamp}.csv')
                
        # Generate video if requested
        if args.video:
            video_path = args.video
            if not video_path.endswith('.mp4'):
                video_path += '.mp4'
            print(f"Generating side-by-side comparison video to {video_path}...")
            rtk_sim.create_comparison_video(std_sim, video_path, fps=args.fps)
            print(f"Comparison video saved to: {video_path}")
    else:
        # Create simulation with specified RTK mode
        sim = TrajectoryAccuracySimulation(waypoints, rtk_enabled=args.rtk, 
                                          wind_strength=args.wind, sim_options=sim_options)
        
        # Run simulation
        print(f"Running simulation with RTK {'enabled' if args.rtk else 'disabled'}...")
        sim.run_simulation(dt=sim_options['dt'])
        
        # Analyze results
        analysis = sim.analyze_results()
        print("\nSimulation Results:")
        print(f"{'RTK Status:':<20} {'Enabled' if args.rtk else 'Disabled'}")
        print(f"{'Mean Error:':<20} {analysis['mean_position_error']:.3f} m")
        print(f"{'Max Error:':<20} {analysis['max_position_error']:.3f} m")
        print(f"{'RMSE:':<20} {analysis['rmse_position']:.3f} m")
        print(f"{'% under 10cm:':<20} {analysis['percentage_under_10cm']:.1f}%")
        print(f"{'% under 5cm:':<20} {analysis['percentage_under_5cm']:.1f}%")
        print(f"{'% under 1cm:':<20} {np.mean(sim.results['position_error'] < 0.01) * 100:.1f}%")
        print(f"{'Simulation time:':<20} {sim.results['time'][-1]:.1f}s")
        
        # Setup the save paths
        save_path = args.save
        if save_path and not any(save_path.endswith(ext) for ext in ['.png', '.jpg', '.pdf']):
            save_path = save_path + '.png'
        
        # Always save in headless mode
        if headless_mode and not save_path:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            gps_type = "rtk" if args.rtk else "std"
            save_path = f"trajectory_{gps_type}_{timestamp}.png"
            print(f"Auto-saving visualization to: {save_path}")
        
        # Plot results and save
        if save_path or not headless_mode:
            sim.plot_results(show_animation=(args.animate and not headless_mode), save_path=save_path)
        
        # Export results if requested or in headless mode
        if args.export:
            sim.export_results(args.export)
        elif headless_mode:
            # Auto-export in headless mode
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            gps_type = "rtk" if args.rtk else "std"
            sim.export_results(f'{gps_type}_results_{timestamp}.csv')
            
        # Generate video if requested
        if args.video:
            video_path = args.video
            if not video_path.endswith('.mp4'):
                video_path += '.mp4'
            print(f"Generating high-quality video to {video_path}...")
            sim.create_video(video_path, fps=args.fps)
            print(f"Video saved to: {video_path}")
            
        # Interactive PID tuning
        if args.tune_pid and not headless_mode:
            sim.interactive_pid_tuning(waypoints)


# Add this line to call main() when the script is run directly
if __name__ == "__main__":
    main()