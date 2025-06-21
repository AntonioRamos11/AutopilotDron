#!/usr/bin/env python3
"""
Minimum Snap Trajectory Generation for High-Speed Drone Navigation

This module implements minimum snap trajectory optimization for smooth, 
high-speed point-to-point navigation. The algorithm minimizes the snap 
(4th derivative of position) to create smooth trajectories that respect 
the dynamic constraints of the drone.

Reference: "Minimum Snap Trajectory Generation and Control for Quadrotors"
           by Daniel Mellinger and Vijay Kumar
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix, linalg as sla, lil_matrix,bmat
from mpl_toolkits.mplot3d import Axes3D




class MinSnapTrajectory:
    """Minimum Snap Trajectory Generator for quadrotor UAVs."""
    
    def __init__(self):
        """Initialize the trajectory generator."""
        self.dimension = 3  # 3D trajectories
        self.order = 8  # Polynomial order
        self.continuity_order = 4  # Continuity up to snap
        
    def generate_trajectory(self, waypoints, segment_times, corridor_constraints=None, 
                            start_vel=None, end_vel=None, start_acc=None, end_acc=None):
        """
        Generate a minimum snap trajectory through the given waypoints.
        
        Args:
            waypoints: Array of shape (N, 3) containing N waypoints in 3D space
            segment_times: Array of shape (N-1,) containing time allocated to each segment
            corridor_constraints: Optional constraints on the path corridor
            start_vel: Optional initial velocity constraint (3D vector)
            end_vel: Optional final velocity constraint (3D vector)
            start_acc: Optional initial acceleration constraint (3D vector)
            end_acc: Optional final acceleration constraint (3D vector)
            
        Returns:
            Trajectory object containing polynomial coefficients and timing information
        """
        n_waypoints = len(waypoints)
        n_segments = n_waypoints - 1
        
        # Default constraints if not provided
        if start_vel is None:
            start_vel = np.zeros(3)
        if end_vel is None:
            end_vel = np.zeros(3)
        if start_acc is None:
            start_acc = np.zeros(3)
        if end_acc is None:
            end_acc = np.zeros(3)
            
        # Setup optimization problem
        # Number of coefficients per segment per dimension
        n_coef_per_seg = self.order
        # Total number of coefficients for all segments and dimensions
        n_coef_total = n_segments * n_coef_per_seg * self.dimension
        
        # Cost matrix - minimize snap (4th derivative)
        cost_matrix = self._create_cost_matrix(n_segments, segment_times)
        
        # Equality constraints matrix
        A_eq, b_eq = self._create_constraint_matrices(
            waypoints, n_segments, segment_times, n_coef_per_seg,
            start_vel, end_vel, start_acc, end_acc
        )
        
        # Solve the QP problem
        # Minimize 0.5 * x^T * P * x subject to A_eq * x = b_eq
        P = csc_matrix(cost_matrix)
        
        # Solve the system A_eq * x = b_eq while minimizing x^T * P * x
        coefficients = self._solve_qp(P, A_eq, b_eq)
        
        # Reshape the coefficients for easier handling
        coefficients = coefficients.reshape(self.dimension, n_segments, n_coef_per_seg)
        
        return Trajectory(coefficients, segment_times)
    
    def _create_cost_matrix(self, n_segments, segment_times):
        """Create the cost matrix for minimum snap optimization."""
        cost_matrix = np.zeros((n_segments * self.order * self.dimension, 
                               n_segments * self.order * self.dimension))
        
        # For each dimension, compute the cost matrix
        for dim in range(self.dimension):
            dim_cost_matrix = np.zeros((n_segments * self.order, n_segments * self.order))
            
            for seg in range(n_segments):
                T = segment_times[seg]
                
                # Compute the Q matrix for the snap minimization
                Q = np.zeros((self.order, self.order))
                # Populate Q for minimizing the integral of squared snap
                for i in range(4, self.order):
                    for j in range(4, self.order):
                        # Compute the cost based on the 4th derivative (snap)
                        p = i - 4
                        q = j - 4
                        Q[i, j] = (i * (i-1) * (i-2) * (i-3) * 
                                  j * (j-1) * (j-2) * (j-3) * 
                                  T**(p+q+1)) / (p+q+1)
                
                # Add the Q matrix to the corresponding segment in the cost matrix
                start_idx = seg * self.order
                end_idx = (seg + 1) * self.order
                dim_cost_matrix[start_idx:end_idx, start_idx:end_idx] = Q
            
            # Add the dimension's cost matrix to the overall cost matrix
            start_idx = dim * n_segments * self.order
            end_idx = (dim + 1) * n_segments * self.order
            cost_matrix[start_idx:end_idx, start_idx:end_idx] = dim_cost_matrix
        
        return cost_matrix
    
    def _create_constraint_matrices(self, waypoints, n_segments, segment_times, n_coef_per_seg,
                                   start_vel, end_vel, start_acc, end_acc):
        """Create block-diagonal constraint matrices for the optimization problem."""
        n_waypoints = n_segments + 1
        
        # --- Calculate number of constraints PER DIMENSION ---
        # Position constraints: one at the start, one at the end of each segment
        n_pos_constraints = n_segments + 1 
        # Velocity constraints: one at the start, one at the end of the whole trajectory
        n_vel_constraints = 2
        # Acceleration constraints: one at the start, one at the end of the whole trajectory
        n_acc_constraints = 2
        # Continuity constraints at intermediate waypoints (n_segments - 1 junctions)
        n_cont_constraints = (n_segments - 1) * self.continuity_order
        
        n_constraints_per_dim = n_pos_constraints + n_vel_constraints + n_acc_constraints + n_cont_constraints
        
        # --- Total number of variables and constraints for ALL DIMENSIONS ---
        n_coeffs_per_dim = n_segments * n_coef_per_seg
        n_coeffs_total = self.dimension * n_coeffs_per_dim
        n_constraints_total = self.dimension * n_constraints_per_dim

        # Initialize matrices using LIL format for efficient construction
        A_eq = lil_matrix((n_constraints_total, n_coeffs_total))
        b_eq = np.zeros(n_constraints_total)
        
        # --- Populate matrices for each dimension ---
        for dim in range(self.dimension):
            # Calculate offsets for this dimension's block
            col_offset = dim * n_coeffs_per_dim
            row_offset = dim * n_constraints_per_dim
            
            # Reset relative row index for this dimension's constraints
            row_idx = 0
            
            # --- Position constraints ---
            # Constraint for the start of the first segment (t=0)
            A_eq[row_offset + row_idx, col_offset : col_offset + n_coef_per_seg] = self._poly_eval_coeffs(0)
            b_eq[row_offset + row_idx] = waypoints[0, dim]
            row_idx += 1
            
            # Constraint for the end of each segment
            for seg in range(n_segments):
                A_eq[row_offset + row_idx, col_offset + seg * n_coef_per_seg : col_offset + (seg + 1) * n_coef_per_seg] = \
                    self._poly_eval_coeffs(segment_times[seg])
                b_eq[row_offset + row_idx] = waypoints[seg + 1, dim]
                row_idx += 1

            # --- Velocity constraints at endpoints ---
            # Initial velocity
            A_eq[row_offset + row_idx, col_offset : col_offset + n_coef_per_seg] = self._poly_eval_deriv_coeffs(0, 1)
            b_eq[row_offset + row_idx] = start_vel[dim]
            row_idx += 1
            
            # Final velocity
            A_eq[row_offset + row_idx, col_offset + (n_segments - 1) * n_coef_per_seg : col_offset + n_segments * n_coef_per_seg] = \
                self._poly_eval_deriv_coeffs(segment_times[-1], 1)
            b_eq[row_offset + row_idx] = end_vel[dim]
            row_idx += 1
            
            # --- Acceleration constraints at endpoints ---
            # Initial acceleration
            A_eq[row_offset + row_idx, col_offset : col_offset + n_coef_per_seg] = self._poly_eval_deriv_coeffs(0, 2)
            b_eq[row_offset + row_idx] = start_acc[dim]
            row_idx += 1
            
            # Final acceleration
            A_eq[row_offset + row_idx, col_offset + (n_segments - 1) * n_coef_per_seg : col_offset + n_segments * n_coef_per_seg] = \
                self._poly_eval_deriv_coeffs(segment_times[-1], 2)
            b_eq[row_offset + row_idx] = end_acc[dim]
            row_idx += 1
            
            # --- Continuity constraints at intermediate waypoints ---
            for seg in range(n_segments - 1):
                # Continuity for derivatives 0 (pos) to continuity_order-1
                for deriv in range(self.continuity_order):
                    # Derivative at end of segment 'seg'
                    A_eq[row_offset + row_idx, col_offset + seg * n_coef_per_seg : col_offset + (seg + 1) * n_coef_per_seg] = \
                        self._poly_eval_deriv_coeffs(segment_times[seg], deriv)
                    
                    # Must equal derivative at start of segment 'seg+1'
                    A_eq[row_offset + row_idx, col_offset + (seg + 1) * n_coef_per_seg : col_offset + (seg + 2) * n_coef_per_seg] = \
                        -self._poly_eval_deriv_coeffs(0, deriv)
                    
                    # The difference should be zero
                    b_eq[row_offset + row_idx] = 0
                    row_idx += 1
        
        return A_eq, b_eq
    
    def _poly_eval_coeffs(self, t):
        """Get the coefficients for evaluating a polynomial at time t."""
        return np.array([t**i for i in range(self.order)])
    
    def _poly_eval_deriv_coeffs(self, t, deriv_order):
        """Get the coefficients for evaluating the derivative of a polynomial at time t."""
        coeffs = np.zeros(self.order)
        for i in range(deriv_order, self.order):
            # Calculate the coefficient for the (i-deriv_order)'th term of the derivative
            coef = 1
            for j in range(deriv_order):
                coef *= (i - j)
            coeffs[i] = coef * t**(i-deriv_order)
        return coeffs
    
    def _solve_qp(self, P, A_eq, b_eq):
        """
        Solve the quadratic program:
        minimize 0.5 * x^T * P * x subject to A_eq * x = b_eq
        """
        # Use sparse linear algebra for efficiency
        A_eq_sparse = csc_matrix(A_eq)
        
        # Create the KKT matrix using the block matrix constructor for efficiency
        n_p = P.shape[0]
        n_a = A_eq_sparse.shape[0]
        
        # Create a zero block of the correct size for the bottom-right
        zero_block = csc_matrix((n_a, n_a))
        
        # Construct the KKT matrix from blocks
        kkt_matrix = bmat([
            [P, A_eq_sparse.T],
            [A_eq_sparse, zero_block]
        ], format='csc')
        
        # Create the right-hand side vector
        kkt_rhs = np.concatenate([np.zeros(n_p), b_eq])
        
        # Solve the KKT system
        try:
            sol = sla.spsolve(kkt_matrix, kkt_rhs)
            x = sol[:n_p]
        except Exception as e:
            self.logger.error(f"Failed to solve KKT system: {e}")
            # Fallback to a least-squares solution if direct solve fails
            x = sla.lsqr(A_eq_sparse, b_eq)[0]

        return x


class Trajectory:
    """Represents a trajectory with polynomial coefficients and timing information."""
    
    def __init__(self, coefficients, segment_times):
        """
        Initialize a trajectory.
        
        Args:
            coefficients: Array of shape (dimension, n_segments, n_coef_per_segment)
                containing the polynomial coefficients for each dimension and segment
            segment_times: Array of shape (n_segments,) containing time allocated 
                to each segment
        """
        self.coefficients = coefficients
        self.segment_times = segment_times
        self.dimension = coefficients.shape[0]
        self.n_segments = coefficients.shape[1]
        self.order = coefficients.shape[2]
        
        # Compute cumulative segment times for easier trajectory evaluation
        self.cum_segment_times = np.zeros(self.n_segments + 1)
        self.cum_segment_times[1:] = np.cumsum(segment_times)
        self.total_time = self.cum_segment_times[-1]
        
    def evaluate(self, t, derivative_order=0):
        """
        Evaluate the trajectory at time t.
        
        Args:
            t: Time at which to evaluate the trajectory
            derivative_order: Order of derivative to evaluate (0=position, 1=velocity, etc.)
            
        Returns:
            Position (or derivative) at time t as a vector of dimension self.dimension
        """
        # Clip time to valid range
        t = np.clip(t, 0, self.total_time)
        
        # Find which segment this time belongs to
        segment_idx = np.searchsorted(self.cum_segment_times[1:], t, side='right')
        
        # Get the time relative to the start of the segment
        if segment_idx > 0:
            relative_t = t - self.cum_segment_times[segment_idx]
        else:
            relative_t = t
        
        # Evaluate the polynomial for this segment at the relative time
        result = np.zeros(self.dimension)
        for dim in range(self.dimension):
            coeffs = self.coefficients[dim, segment_idx]
            
            # Evaluate the appropriate derivative
            deriv_coeffs = coeffs.copy()
            for i in range(derivative_order):
                # Differentiate the polynomial coefficients
                for j in range(1, self.order):
                    deriv_coeffs[j-1] = j * deriv_coeffs[j]
                deriv_coeffs[-1] = 0
            
            # Evaluate the polynomial
            value = 0
            for i in range(self.order):
                value += deriv_coeffs[i] * (relative_t ** i)
            
            result[dim] = value
            
        return result
    
    def visualize(self, resolution=100, show_waypoints=True, waypoints=None):
        """
        Visualize the trajectory in 3D.
        
        Args:
            resolution: Number of points to sample along the trajectory
            show_waypoints: Whether to show the waypoints used to generate the trajectory
            waypoints: Array of waypoints to show (if not provided, will be inferred from trajectory)
        """
        # Sample points along the trajectory
        times = np.linspace(0, self.total_time, resolution)
        positions = np.array([self.evaluate(t) for t in times])
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
        
        # Show waypoints if requested
        if show_waypoints and waypoints is not None:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                      c='r', marker='o', s=100, label='Waypoints')
        
        # Show start and end points
        start_pos = self.evaluate(0)
        end_pos = self.evaluate(self.total_time)
        ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], 
                  c='g', marker='o', s=100, label='Start')
        ax.scatter([end_pos[0]], [end_pos[1]], [end_pos[2]], 
                  c='m', marker='o', s=100, label='End')
        
        # Add labels and legend
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        ax.set_title('Minimum Snap Trajectory')
        
        plt.show()
        
    def export_to_px4_mission(self, filename, altitude=10, speed=5):
        """
        Export the trajectory to a PX4 mission file format.
        
        Args:
            filename: Name of the output file
            altitude: Default altitude in meters (AGL)
            speed: Speed between waypoints in m/s
            
        Note:
            This is a simplified export. For real missions, you would need
            to convert local coordinates to GPS coordinates.
        """
        # Sample the trajectory to get waypoints
        # We'll use a simple approach: sample points at regular intervals
        num_samples = min(50, self.n_segments * 10)  # Limit to 50 waypoints for simplicity
        times = np.linspace(0, self.total_time, num_samples)
        positions = np.array([self.evaluate(t) for t in times])
        
        # Write the mission file
        with open(filename, 'w') as f:
            # Write header
            f.write("QGC WPL 110\n")
            
            # Home position (assumed to be the start of the trajectory)
            f.write(f"0\t1\t0\t16\t0\t0\t0\t0\t{positions[0, 0]}\t{positions[0, 1]}\t{altitude}\t1\n")
            
            # Waypoints
            for i in range(num_samples):
                # WP index, current, frame, command, p1, p2, p3, p4, lat, lon, alt, autocontinue
                f.write(f"{i+1}\t0\t3\t16\t0\t0\t0\t0\t{positions[i, 0]}\t{positions[i, 1]}\t{positions[i, 2]}\t1\n")
        
        print(f"Exported trajectory to {filename}")


if __name__ == "__main__":
    # Example usage
    waypoints = np.array([
        [0, 0, 0],
        [5, 5, 5],
        [10, 0, 5],
        [15, -5, 0],
        [20, 0, 0]
    ])
    
    # Time allocation to each segment (in seconds)
    segment_times = np.array([5.0, 5.0, 5.0, 5.0])
    
    # Initial and final velocities and accelerations
    start_vel = np.array([0, 0, 0])
    end_vel = np.array([0, 0, 0])
    start_acc = np.array([0, 0, 0])
    end_acc = np.array([0, 0, 0])
    
    # Generate the trajectory
    traj_gen = MinSnapTrajectory()
    trajectory = traj_gen.generate_trajectory(
        waypoints, segment_times,
        start_vel=start_vel, end_vel=end_vel,
        start_acc=start_acc, end_acc=end_acc
    )
    
    # Visualize the trajectory
    trajectory.visualize(waypoints=waypoints)
    
    # Export to PX4 mission file
    trajectory.export_to_px4_mission("example_mission.txt")
    
    # You can evaluate the trajectory at any time
    t = 7.5  # Time in seconds
    position = trajectory.evaluate(t)
    velocity = trajectory.evaluate(t, derivative_order=1)
    acceleration = trajectory.evaluate(t, derivative_order=2)
    
    print(f"At t={t}s:")
    print(f"Position: {position}")
    print(f"Velocity: {velocity}")
    print(f"Acceleration: {acceleration}")