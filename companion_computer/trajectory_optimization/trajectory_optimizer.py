#!/usr/bin/env python3
"""
Trajectory Optimizer - Integrates minimum snap trajectory generation with real-time optimization
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

# Import our existing min_snap_trajectory module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../path_planning'))
from min_snap_trajectory import MinSnapTrajectory, Trajectory

@dataclass
class TrajectoryConstraints:
    """Constraints for trajectory optimization."""
    max_velocity: float = 15.0
    max_acceleration: float = 5.0
    max_jerk: float = 10.0
    max_snap: float = 20.0
    corridor_width: float = 5.0
    safety_margin: float = 2.0

@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""
    trajectory: Trajectory
    success: bool
    optimization_time: float
    cost: float
    iterations: int
    message: str

class TrajectoryOptimizer:
    """Real-time trajectory optimization for autonomous navigation."""
    
    def __init__(self, constraints: Optional[TrajectoryConstraints] = None):
        """Initialize the trajectory optimizer."""
        self.logger = logging.getLogger(__name__)
        self.constraints = constraints or TrajectoryConstraints()
        self.min_snap_generator = MinSnapTrajectory()
        self.current_trajectory: Optional[Trajectory] = None
        self.obstacles: List[dict] = []
        
    def set_constraints(self, constraints: TrajectoryConstraints):
        """Update trajectory constraints."""
        self.constraints = constraints
        self.logger.info("Trajectory constraints updated")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray):
        """Add an obstacle to be avoided, using its full dimensions."""
        # This method signature is now corrected to accept a 'size' vector,
        # matching what the main simulation provides.
        obstacle = {
            'position': np.array(position),
            'size': np.array(size),
            # We still calculate a radius for any legacy code or visualization,
            # but the primary data is the size vector.
            'radius': np.max(size) / 2.0
        }
        self.obstacles.append(obstacle)
        # self.logger.info(f"Added obstacle at {position} with size {size}")
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles = []
        self.logger.info("All obstacles cleared")
    
    def optimize_trajectory(self, waypoints: np.ndarray, 
                          current_state: dict,
                          segment_times: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Optimize trajectory through waypoints considering constraints and obstacles.
        
        Args:
            waypoints: Array of waypoints (N x 3)
            current_state: Current drone state (position, velocity, acceleration)
            segment_times: Optional time allocation for segments
            
        Returns:
            OptimizationResult with optimized trajectory
        """
        start_time = time.time()
        
        try:
            # Extract current state
            current_pos = np.array(current_state.get('position', [0, 0, 0]))
            current_vel = np.array(current_state.get('velocity', [0, 0, 0]))
            current_acc = np.array(current_state.get('acceleration', [0, 0, 0]))
            
            # Ensure waypoints include current position as first point
            if not np.allclose(waypoints[0], current_pos, atol=0.1):
                waypoints = np.vstack([current_pos, waypoints])
            
            # Auto-generate segment times if not provided
            if segment_times is None:
                segment_times = self._calculate_segment_times(waypoints)
            
            # CRITICAL FIX: The internal _avoid_obstacles logic is naive and conflicts
            # with the smarter re-planning logic in the main simulation. The high-level
            # planner is responsible for generating safe waypoints. This call is
            # the source of the crash and is being removed.
            # if self.obstacles:
            #     waypoints = self._avoid_obstacles(waypoints)
            
            # --- Trajectory Generation and Validation Loop ---
            # The previous logic only tried once. If the relaxed trajectory was still
            # invalid, it would fail without retrying. We now implement a loop with
            # a maximum number of attempts.
            max_attempts = 3
            trajectory = None
            validation_result = {'valid': False}
            attempts = 0
            
            while not validation_result['valid'] and attempts < max_attempts:
                attempts += 1
                
                # Generate initial trajectory
                trajectory = self.min_snap_generator.generate_trajectory(
                    waypoints=waypoints,
                    segment_times=segment_times,
                    start_vel=current_vel,
                    end_vel=np.zeros(3),
                    start_acc=current_acc,
                    end_acc=np.zeros(3)
                )
                
                # Validate trajectory against constraints
                validation_result = self._validate_trajectory(trajectory)
                
                if not validation_result['valid']:
                    # Re-optimize with relaxed timing
                    self.logger.warning(f"Trajectory invalid: {validation_result['reason']}")
                    segment_times = self._relax_segment_times(segment_times, validation_result)
            
            optimization_time = time.time() - start_time
            
            # Calculate trajectory cost (integral of snap)
            cost = self._calculate_trajectory_cost(trajectory)
            
            self.current_trajectory = trajectory
            
            return OptimizationResult(
                trajectory=trajectory,
                success=True,
                optimization_time=optimization_time,
                cost=cost,
                iterations=attempts,
                message="Trajectory optimized successfully"
            )
            
        except Exception as e:
            optimization_time = time.time() - start_time
            self.logger.error(f"Trajectory optimization failed: {e}")
            
            return OptimizationResult(
                trajectory=None,
                success=False,
                optimization_time=optimization_time,
                cost=float('inf'),
                iterations=0,
                message=f"Optimization failed: {str(e)}"
            )
    
    def _calculate_segment_times(self, waypoints: np.ndarray) -> np.ndarray:
        """Calculate optimal segment times based on distance and constraints."""
        segment_distances = []
        
        for i in range(len(waypoints) - 1):
            distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
            segment_distances.append(distance)
        
        segment_distances = np.array(segment_distances)
        
        # Calculate times based on maximum velocity with safety margin
        safe_velocity = self.constraints.max_velocity * 0.8
        base_times = segment_distances / safe_velocity
        
        # Ensure minimum time for acceleration/deceleration
        min_time = 2.0
        segment_times = np.maximum(base_times, min_time)
        
        return segment_times
    
    def _avoid_obstacles(self, waypoints: np.ndarray) -> np.ndarray:
        """Modify waypoints to avoid obstacles."""
        if not self.obstacles:
            return waypoints
        
        modified_waypoints = waypoints.copy()
        
        for i in range(1, len(waypoints) - 1):  # Don't modify start/end points
            wp = waypoints[i]
            
            for obstacle in self.obstacles:
                obs_pos = obstacle['position']
                obs_radius = obstacle['radius']
                safety_radius = obs_radius + self.constraints.safety_margin
                
                # Check if waypoint is too close to obstacle
                distance = np.linalg.norm(wp - obs_pos)
                
                if distance < safety_radius:
                    # Move waypoint away from obstacle
                    direction = (wp - obs_pos) / distance
                    new_position = obs_pos + direction * safety_radius
                    modified_waypoints[i] = new_position
                    
                    self.logger.info(f"Moved waypoint {i} to avoid obstacle")
        
        return modified_waypoints
    
    def _validate_trajectory(self, trajectory: Trajectory) -> dict:
        """Validate trajectory against dynamic constraints."""
        # Sample trajectory at high resolution
        dt = 0.1
        times = np.arange(0, trajectory.total_time, dt)
        
        max_vel = 0
        max_acc = 0
        max_jerk = 0
        
        for t in times:
            velocity = trajectory.evaluate(t, derivative_order=1)
            acceleration = trajectory.evaluate(t, derivative_order=2)
            jerk = trajectory.evaluate(t, derivative_order=3)
            
            vel_mag = np.linalg.norm(velocity)
            acc_mag = np.linalg.norm(acceleration)
            jerk_mag = np.linalg.norm(jerk)
            
            max_vel = max(max_vel, vel_mag)
            max_acc = max(max_acc, acc_mag)
            max_jerk = max(max_jerk, jerk_mag)
            
            # Check constraints
            if vel_mag > self.constraints.max_velocity:
                return {
                    'valid': False,
                    'reason': f'Velocity constraint violated: {vel_mag:.2f} > {self.constraints.max_velocity}',
                    'max_velocity': max_vel
                }
            
            if acc_mag > self.constraints.max_acceleration:
                return {
                    'valid': False,
                    'reason': f'Acceleration constraint violated: {acc_mag:.2f} > {self.constraints.max_acceleration}',
                    'max_acceleration': max_acc
                }
            
            if jerk_mag > self.constraints.max_jerk:
                return {
                    'valid': False,
                    'reason': f'Jerk constraint violated: {jerk_mag:.2f} > {self.constraints.max_jerk}',
                    'max_jerk': max_jerk
                }
        
        return {
            'valid': True,
            'reason': 'All constraints satisfied',
            'max_velocity': max_vel,
            'max_acceleration': max_acc,
            'max_jerk': max_jerk
        }
    
    def _relax_segment_times(self, segment_times: np.ndarray, 
                           validation_result: dict) -> np.ndarray:
        """Relax segment times to satisfy constraints."""
        # Increase segment times by factor based on constraint violation
        if 'max_velocity' in validation_result:
            violation_factor = validation_result['max_velocity'] / self.constraints.max_velocity
        elif 'max_acceleration' in validation_result:
            violation_factor = validation_result['max_acceleration'] / self.constraints.max_acceleration
        elif 'max_jerk' in validation_result:
            violation_factor = validation_result['max_jerk'] / self.constraints.max_jerk
        else:
            violation_factor = 1.5
        
        # Apply safety factor
        relaxation_factor = violation_factor * 1.2
        relaxed_times = segment_times * relaxation_factor
        
        self.logger.info(f"Relaxed segment times by factor {relaxation_factor:.2f}")
        return relaxed_times
    
    def _calculate_trajectory_cost(self, trajectory: Trajectory) -> float:
        """Calculate the cost of the trajectory (integral of snap squared)."""
        dt = 0.05
        times = np.arange(0, trajectory.total_time, dt)
        
        total_cost = 0.0
        
        for t in times:
            snap = trajectory.evaluate(t, derivative_order=4)
            snap_magnitude = np.linalg.norm(snap)
            total_cost += snap_magnitude**2 * dt
        
        return total_cost
    
    def replan_trajectory(self, current_state: dict, 
                         remaining_waypoints: np.ndarray,
                         look_ahead_time: float = 5.0) -> OptimizationResult:
        """Replan trajectory from current state to remaining waypoints."""
        if self.current_trajectory is None:
            return self.optimize_trajectory(remaining_waypoints, current_state)
        
        # Find current position in trajectory
        current_time = time.time()  # In practice, this would be trajectory time
        
        # Extract future waypoints within look-ahead window
        future_waypoints = self._extract_future_waypoints(
            remaining_waypoints, look_ahead_time
        )
        
        return self.optimize_trajectory(future_waypoints, current_state)
    
    def _extract_future_waypoints(self, waypoints: np.ndarray, 
                                 look_ahead_time: float) -> np.ndarray:
        """Extract waypoints within the look-ahead time window."""
        # This is a simplified implementation
        # In practice, you'd use actual trajectory timing
        max_waypoints = int(look_ahead_time / 2.0)  # Assume ~2s per waypoint
        
        return waypoints[:min(len(waypoints), max_waypoints)]
    
    def get_trajectory_info(self) -> dict:
        """Get information about the current trajectory."""
        if self.current_trajectory is None:
            return {"status": "no_trajectory"}
        
        return {
            "status": "trajectory_ready",
            "total_time": self.current_trajectory.total_time,
            "segments": self.current_trajectory.n_segments,
            "dimensions": self.current_trajectory.dimension
        }