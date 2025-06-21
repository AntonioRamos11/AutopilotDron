#!/usr/bin/env python3
"""
Obstacle Detection and Avoidance System
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import threading
import time

class ObstacleType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    GROUND = "ground"
    BUILDING = "building"
    VEGETATION = "vegetation"
    AIRCRAFT = "aircraft"

@dataclass
class Obstacle:
    """Represents a detected obstacle."""
    id: int
    position: np.ndarray  # 3D position [x, y, z]
    velocity: np.ndarray  # 3D velocity [vx, vy, vz]
    size: np.ndarray      # Size [width, height, depth]
    obstacle_type: ObstacleType
    confidence: float
    timestamp: float
    is_active: bool = True

@dataclass
class AvoidanceCommand:
    """Command for obstacle avoidance maneuver."""
    maneuver_type: str  # "climb", "descend", "left", "right", "stop", "rtl"
    target_position: np.ndarray
    urgency: float  # 0.0 to 1.0
    duration: float
    reason: str

class ObstacleDetector:
    """Multi-sensor obstacle detection system."""
    
    def __init__(self):
        """Initialize the obstacle detector."""
        self.logger = logging.getLogger(__name__)
        self.obstacles: Dict[int, Obstacle] = {}
        self.next_obstacle_id = 1
        self.detection_range = 50.0  # meters
        self.is_running = False
        self.detection_thread = None
        
        # Sensor interfaces (will be initialized by sensor managers)
        self.depth_camera = None
        self.lidar = None
        self.radar = None
        
        # Detection parameters
        self.min_obstacle_size = 0.5  # meters
        self.max_obstacle_age = 5.0   # seconds
        self.confidence_threshold = 0.7
        self.detection_loop_sleep_time = 0.1  # seconds
        
    def start_detection(self):
        """Start the obstacle detection system."""
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        self.logger.info("Obstacle detection started")
    
    def stop_detection(self):
        """Stop the obstacle detection system."""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join()
        self.logger.info("Obstacle detection stopped")
    
    def set_sensor_interfaces(self, depth_camera=None, lidar=None, radar=None):
        """Set sensor interfaces for obstacle detection."""
        self.depth_camera = depth_camera
        self.lidar = lidar
        self.radar = radar
        self.logger.info("Sensor interfaces configured")
    
    def _detection_loop(self):
        """Main detection loop running in separate thread."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Detect obstacles from all available sensors
                new_obstacles = []
                
                if self.depth_camera:
                    depth_obstacles = self._detect_from_depth_camera()
                    new_obstacles.extend(depth_obstacles)
                
                if self.lidar:
                    lidar_obstacles = self._detect_from_lidar()
                    new_obstacles.extend(lidar_obstacles)
                
                if self.radar:
                    radar_obstacles = self._detect_from_radar()
                    new_obstacles.extend(radar_obstacles)
                
                # Fuse detections from multiple sensors
                fused_obstacles = self._fuse_detections(new_obstacles)
                
                # Update obstacle tracking
                self._update_obstacle_tracking(fused_obstacles, current_time)
                
                # Remove old obstacles
                self._cleanup_old_obstacles(current_time)
                
                time.sleep(self.detection_loop_sleep_time)  # 10 Hz detection rate
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(1.0)
    
    def _detect_from_depth_camera(self) -> List[Obstacle]:
        """Detect obstacles from depth camera data."""
        obstacles = []
        
        try:
            # Get depth image and camera info
            depth_data = self.depth_camera.get_depth_image()
            camera_info = self.depth_camera.get_camera_info()
            
            if depth_data is None:
                return obstacles
            
            # Process depth image to find obstacles
            obstacle_points = self._process_depth_image(depth_data, camera_info)
            
            # Cluster points into obstacles
            clustered_obstacles = self._cluster_points(obstacle_points)
            
            for cluster in clustered_obstacles:
                obstacle = Obstacle(
                    id=self._get_next_obstacle_id(),
                    position=cluster['position'],
                    velocity=np.zeros(3),  # Static assumption initially
                    size=cluster['size'],
                    obstacle_type=ObstacleType.STATIC,
                    confidence=cluster['confidence'],
                    timestamp=time.time()
                )
                obstacles.append(obstacle)
                
        except Exception as e:
            self.logger.error(f"Depth camera detection error: {e}")
        
        return obstacles
    
    def _detect_from_lidar(self) -> List[Obstacle]:
        """Detect obstacles from LiDAR data."""
        obstacles = []
        
        try:
            # Get LiDAR point cloud
            point_cloud = self.lidar.get_point_cloud()
            
            if point_cloud is None:
                return obstacles
            
            # Process point cloud to find obstacles
            obstacle_clusters = self._process_lidar_data(point_cloud)
            
            for cluster in obstacle_clusters:
                obstacle = Obstacle(
                    id=self._get_next_obstacle_id(),
                    position=cluster['position'],
                    velocity=np.zeros(3),
                    size=cluster['size'],
                    obstacle_type=self._classify_lidar_obstacle(cluster),
                    confidence=cluster['confidence'],
                    timestamp=time.time()
                )
                obstacles.append(obstacle)
                
        except Exception as e:
            self.logger.error(f"LiDAR detection error: {e}")
        
        return obstacles
    
    def _detect_from_radar(self) -> List[Obstacle]:
        """Detect obstacles from radar data."""
        obstacles = []
        
        try:
            # Get radar targets
            radar_targets = self.radar.get_targets()
            
            if not radar_targets:
                return obstacles
            
            for target in radar_targets:
                obstacle = Obstacle(
                    id=self._get_next_obstacle_id(),
                    position=np.array(target['position']),
                    velocity=np.array(target['velocity']),
                    size=np.array([2.0, 2.0, 2.0]),  # Default size for radar targets
                    obstacle_type=ObstacleType.DYNAMIC if np.linalg.norm(target['velocity']) > 0.5 else ObstacleType.STATIC,
                    confidence=target['confidence'],
                    timestamp=time.time()
                )
                obstacles.append(obstacle)
                
        except Exception as e:
            self.logger.error(f"Radar detection error: {e}")
        
        return obstacles
    
    def _process_depth_image(self, depth_data: np.ndarray, camera_info: dict) -> List[np.ndarray]:
        """Process depth image to extract obstacle points."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated computer vision algorithms
        
        height, width = depth_data.shape
        fx, fy = camera_info['fx'], camera_info['fy']
        cx, cy = camera_info['cx'], camera_info['cy']
        
        obstacle_points = []
        
        # Find pixels with obstacles (closer than expected ground plane)
        for v in range(0, height, 10):  # Sample every 10 pixels for performance
            for u in range(0, width, 10):
                depth = depth_data[v, u]
                
                if depth > 0 and depth < self.detection_range:
                    # Convert pixel to 3D point
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    
                    point = np.array([x, y, z])
                    obstacle_points.append(point)
        
        return obstacle_points
    
    def _cluster_points(self, points: List[np.ndarray]) -> List[dict]:
        """Cluster 3D points into obstacles."""
        if not points:
            return []
        
        # Simple clustering based on distance
        # In practice, you'd use DBSCAN or similar algorithms
        
        clusters = []
        points_array = np.array(points)
        used = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points_array):
            if used[i]:
                continue
            
            # Find nearby points
            distances = np.linalg.norm(points_array - point, axis=1)
            cluster_mask = distances < 2.0  # 2m clustering distance
            cluster_points = points_array[cluster_mask]
            used[cluster_mask] = True
            
            if len(cluster_points) > 5:  # Minimum points for valid obstacle
                # Calculate cluster properties
                center = np.mean(cluster_points, axis=0)
                size = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                confidence = min(1.0, len(cluster_points) / 20.0)
                
                clusters.append({
                    'position': center,
                    'size': size,
                    'confidence': confidence,
                    'points': cluster_points
                })
        
        return clusters
    
    def _process_lidar_data(self, point_cloud: np.ndarray) -> List[dict]:
        """Process LiDAR point cloud to find obstacles."""
        # Filter points by range
        distances = np.linalg.norm(point_cloud, axis=1)
        valid_mask = (distances > 1.0) & (distances < self.detection_range)
        filtered_points = point_cloud[valid_mask]
        
        # Ground plane removal (simple approach)
        # Assume ground is below certain height
        ground_height = -2.0  # meters below drone
        non_ground_mask = filtered_points[:, 2] > ground_height
        obstacle_points = filtered_points[non_ground_mask]
        
        # Cluster the remaining points
        return self._cluster_points(obstacle_points.tolist())
    
    def _classify_lidar_obstacle(self, cluster: dict) -> ObstacleType:
        """Classify obstacle type based on LiDAR cluster properties."""
        size = cluster['size']
        position = cluster['position']
        
        # Simple classification based on size and position
        if size[2] > 10.0:  # Tall objects
            return ObstacleType.BUILDING
        elif size[2] < 2.0 and np.linalg.norm(size[:2]) < 3.0:  # Small, low objects
            return ObstacleType.VEGETATION
        else:
            return ObstacleType.STATIC
    
    def _fuse_detections(self, new_obstacles: List[Obstacle]) -> List[Obstacle]:
        """Fuse detections from multiple sensors."""
        if not new_obstacles:
            return []
        
        # Simple fusion: remove duplicates based on proximity
        fused_obstacles = []
        
        for obstacle in new_obstacles:
            is_duplicate = False
            
            for existing in fused_obstacles:
                distance = np.linalg.norm(obstacle.position - existing.position)
                if distance < 3.0:  # Consider as same obstacle if within 3m
                    # Keep the one with higher confidence
                    if obstacle.confidence > existing.confidence:
                        fused_obstacles.remove(existing)
                        fused_obstacles.append(obstacle)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                fused_obstacles.append(obstacle)
        
        return fused_obstacles
    
    def _update_obstacle_tracking(self, new_obstacles: List[Obstacle], current_time: float):
        """Update obstacle tracking with new detections."""
        # Associate new detections with existing obstacles
        for new_obs in new_obstacles:
            best_match = None
            best_distance = float('inf')
            
            for obs_id, existing_obs in self.obstacles.items():
                distance = np.linalg.norm(new_obs.position - existing_obs.position)
                if distance < best_distance and distance < 5.0:  # 5m association threshold
                    best_distance = distance
                    best_match = obs_id
            
            if best_match:
                # Update existing obstacle
                existing_obs = self.obstacles[best_match]
                
                # Calculate velocity estimate
                dt = current_time - existing_obs.timestamp
                if dt > 0:
                    velocity = (new_obs.position - existing_obs.position) / dt
                    # Apply simple filtering
                    existing_obs.velocity = 0.7 * existing_obs.velocity + 0.3 * velocity
                
                # Update other properties
                existing_obs.position = new_obs.position
                existing_obs.confidence = max(existing_obs.confidence, new_obs.confidence)
                existing_obs.timestamp = current_time
                existing_obs.is_active = True
            else:
                # Add new obstacle
                self.obstacles[new_obs.id] = new_obs
    
    def _cleanup_old_obstacles(self, current_time: float):
        """Remove obstacles that haven't been detected recently."""
        to_remove = []
        
        for obs_id, obstacle in self.obstacles.items():
            age = current_time - obstacle.timestamp
            if age > self.max_obstacle_age:
                to_remove.append(obs_id)
        
        for obs_id in to_remove:
            del self.obstacles[obs_id]
            self.logger.debug(f"Removed old obstacle {obs_id}")
    
    def _get_next_obstacle_id(self) -> int:
        """Get next unique obstacle ID."""
        obstacle_id = self.next_obstacle_id
        self.next_obstacle_id += 1
        return obstacle_id
    
    def get_obstacles(self, max_distance: Optional[float] = None) -> List[Obstacle]:
        """Get list of currently detected obstacles."""
        obstacles = list(self.obstacles.values())
        
        if max_distance is not None:
            obstacles = [obs for obs in obstacles 
                        if np.linalg.norm(obs.position) <= max_distance]
        
        return [obs for obs in obstacles if obs.confidence >= self.confidence_threshold]
    
    def get_obstacle_count(self) -> int:
        """Get number of currently tracked obstacles."""
        return len([obs for obs in self.obstacles.values() 
                   if obs.confidence >= self.confidence_threshold])
    
    def is_path_clear(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                     corridor_width: float = 5.0) -> Tuple[bool, List[Obstacle]]:
        """Check if path between two points is clear of obstacles."""
        path_vector = end_pos - start_pos
        path_length = np.linalg.norm(path_vector)
        
        if path_length == 0:
            return True, []
        
        path_direction = path_vector / path_length
        blocking_obstacles = []
        
        for obstacle in self.get_obstacles():
            # Vector from start to obstacle
            to_obstacle = obstacle.position - start_pos
            
            # Project onto path direction
            along_path = np.dot(to_obstacle, path_direction)
            
            # Check if obstacle is along the path
            if 0 <= along_path <= path_length:
                # Calculate perpendicular distance
                perpendicular = to_obstacle - along_path * path_direction
                perp_distance = np.linalg.norm(perpendicular)
                
                # Check if obstacle blocks the corridor
                obstacle_radius = np.max(obstacle.size) / 2
                if perp_distance <= (corridor_width / 2 + obstacle_radius):
                    blocking_obstacles.append(obstacle)
        
        return len(blocking_obstacles) == 0, blocking_obstacles