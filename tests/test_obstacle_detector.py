import unittest
import numpy as np
import time
import sys
import os
import logging

# Add the path to the module we are testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'companion_computer', 'obstacle_avoidance')))
from obstacle_detector import ObstacleDetector, ObstacleType

# Suppress logging during tests to keep output clean
logging.basicConfig(level=logging.INFO)

# --- Mock Sensor Classes ---

class MockDepthCamera:
    """A mock depth camera to simulate depth sensor data."""
    def __init__(self):
        self._depth_image = None
        self._camera_info = {
            'fx': 525.0, 'fy': 525.0,
            'cx': 319.5, 'cy': 239.5
        }

    def set_depth_image(self, image):
        self._depth_image = image

    def get_depth_image(self):
        return self._depth_image

    def get_camera_info(self):
        return self._camera_info

class MockLidar:
    """A mock LiDAR to simulate point cloud data."""
    def __init__(self):
        self._point_cloud = None

    def set_point_cloud(self, points):
        self._point_cloud = points

    def get_point_cloud(self):
        points = self._point_cloud
        self._point_cloud = None # Consume the data after reading
        return points

class MockRadar:
    """A mock radar to simulate radar target data."""
    def __init__(self):
        self._targets = []

    def set_targets(self, targets):
        self._targets = targets

    def get_targets(self):
        targets = self._targets
        self._targets = [] # Consume the data after reading
        return targets

# --- Test Suite ---

class TestObstacleDetector(unittest.TestCase):
    """Test suite for the ObstacleDetector class."""

    def setUp(self):
        """Set up for each test."""
        self.detector = ObstacleDetector()
        # Reduce sleep time in the loop for faster tests
        self.detector.detection_loop_sleep_time = 0.01
        self.mock_lidar = MockLidar()
        self.mock_radar = MockRadar()
        self.mock_depth_camera = MockDepthCamera()

    def tearDown(self):
        """Tear down after each test."""
        if self.detector.is_running:
            self.detector.stop_detection()

    def test_initialization(self):
        """Test that the ObstacleDetector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertFalse(self.detector.is_running)
        self.assertEqual(self.detector.get_obstacle_count(), 0)

    def test_start_and_stop_detection(self):
        """Test if the detection thread starts and stops correctly."""
        self.detector.start_detection()
        self.assertTrue(self.detector.is_running)
        self.assertIsNotNone(self.detector.detection_thread)
        self.assertTrue(self.detector.detection_thread.is_alive())
        
        self.detector.stop_detection()
        time.sleep(0.1) # Give thread time to stop
        self.assertFalse(self.detector.is_running)
        self.assertFalse(self.detector.detection_thread.is_alive())

    def test_detection_from_lidar(self):
        """Test obstacle detection from a single LiDAR sensor."""
        # Create a 1x1x1 cube of points at (10, 0, 0)
        points = np.random.rand(100, 3) - 0.5  # Centered at origin
        points += np.array([10, 0, 0])         # Move to (10, 0, 0)
        self.mock_lidar.set_point_cloud(points)
        
        self.detector.set_sensor_interfaces(lidar=self.mock_lidar)
        self.detector.start_detection()
        time.sleep(0.1) # Allow detection loop to run

        obstacles = self.detector.get_obstacles()
        self.assertEqual(len(obstacles), 1)
        
        obstacle = obstacles[0]
        self.assertAlmostEqual(obstacle.position[0], 10.0, delta=0.5)
        self.assertAlmostEqual(obstacle.size[0], 1.0, delta=0.5)

    def test_detection_from_depth_camera(self):
        """Test obstacle detection from a mock depth camera image."""
        # Create a synthetic depth image (480x640)
        # Background is far away (50m), with a 100x100 pixel box in the center that is close (5m)
        depth_image = np.full((480, 640), 50.0, dtype=np.float32)
        depth_image[190:290, 270:370] = 5.0  # 100x100 obstacle at 5 meters

        self.mock_depth_camera.set_depth_image(depth_image)
        self.detector.set_sensor_interfaces(depth_camera=self.mock_depth_camera)
        self.detector.start_detection()
        time.sleep(0.1) # Allow detection loop to run

        obstacles = self.detector.get_obstacles()
        self.assertEqual(len(obstacles), 1, "Should detect one obstacle from the depth image")

        # Check the obstacle's properties
        obstacle = obstacles[0]
        # The obstacle should be detected at ~5 meters depth (z-axis in camera frame)
        self.assertAlmostEqual(obstacle.position[2], 5.0, delta=0.2)
        # Since the obstacle is in the center of the image, its x and y should be close to 0
        self.assertAlmostEqual(obstacle.position[0], 0.0, delta=0.2)
        self.assertAlmostEqual(obstacle.position[1], 0.0, delta=0.2)
        # Check the approximate size. Using camera intrinsics, a 100px wide object at 5m is ~0.95m wide.
        self.assertAlmostEqual(obstacle.size[0], 0.95, delta=0.2) # x-size (width)
        self.assertAlmostEqual(obstacle.size[1], 0.95, delta=0.2) # y-size (height)


    def test_detection_from_radar(self):
        """Test obstacle detection from a single radar sensor."""
        targets = [{
            'position': np.array([20, 5, 2]),
            'velocity': np.array([1, 0, 0]),
            'confidence': 0.9
        }]
        self.mock_radar.set_targets(targets)

        self.detector.set_sensor_interfaces(radar=self.mock_radar)
        self.detector.start_detection()
        time.sleep(0.1)

        obstacles = self.detector.get_obstacles()
        self.assertEqual(len(obstacles), 1)
        
        obstacle = obstacles[0]
        self.assertEqual(obstacle.obstacle_type, ObstacleType.DYNAMIC)
        np.testing.assert_array_almost_equal(obstacle.position, [20, 5, 2])
        np.testing.assert_array_almost_equal(obstacle.velocity, [1, 0, 0])

    def test_sensor_fusion(self):
        """Test fusion of detections from LiDAR and Radar."""
        # LiDAR detects an obstacle with lower confidence
        lidar_points = np.random.rand(10, 3) + np.array([15, 2, 1]) # 10 points -> confidence = 0.5
        self.mock_lidar.set_point_cloud(lidar_points)

        # Radar detects a very close obstacle with higher confidence
        radar_targets = [{
            'position': np.array([15.1, 2.1, 1.1]),
            'velocity': np.zeros(3),
            'confidence': 0.95 # Higher confidence
        }]
        self.mock_radar.set_targets(radar_targets)

        self.detector.set_sensor_interfaces(lidar=self.mock_lidar, radar=self.mock_radar)
        self.detector.start_detection()
        time.sleep(0.1)

        # Should be fused into a single obstacle
        obstacles = self.detector.get_obstacles()
        self.assertEqual(len(obstacles), 1)
        
        # The fused obstacle should have properties from the higher confidence source (Radar)
        obstacle = obstacles[0]
        self.assertAlmostEqual(obstacle.position[0], 15.1, delta=0.2)
        self.assertAlmostEqual(obstacle.confidence, 0.95, delta=0.01)

    def test_obstacle_tracking(self):
        """Test if the detector tracks a moving obstacle over time."""
        self.detector.set_sensor_interfaces(radar=self.mock_radar)
        self.detector.start_detection()

        # Frame 1
        self.mock_radar.set_targets([{'position': np.array([30, 0, 0]), 'velocity': np.zeros(3), 'confidence': 0.9}])
        time.sleep(0.1)
        obstacles = self.detector.get_obstacles()
        self.assertEqual(len(obstacles), 1)
        self.assertAlmostEqual(obstacles[0].position[0], 30.0, delta=0.1)
        self.assertAlmostEqual(np.linalg.norm(obstacles[0].velocity), 0.0, delta=0.1)
        obstacle_id = obstacles[0].id

        # Frame 2: Obstacle moves
        self.mock_radar.set_targets([{'position': np.array([31, 0, 0]), 'velocity': np.zeros(3), 'confidence': 0.9}])
        time.sleep(0.1)
        obstacles = self.detector.get_obstacles()
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(obstacles[0].id, obstacle_id) # Should be the same obstacle
        self.assertAlmostEqual(obstacles[0].position[0], 31.0, delta=0.1)
        # Velocity should be estimated. The smoothing filter results in ~3 m/s.
        self.assertGreater(obstacles[0].velocity[0], 2.0)

    def test_obstacle_cleanup(self):
        """Test if old obstacles are removed after max_obstacle_age."""
        self.detector.max_obstacle_age = 0.1 # Set short age for testing
        self.detector.set_sensor_interfaces(radar=self.mock_radar)
        self.detector.start_detection()

        # Detect an obstacle
        self.mock_radar.set_targets([{'position': np.array([5, 5, 5]), 'velocity': np.zeros(3), 'confidence': 0.9}])
        time.sleep(0.05)
        self.assertEqual(self.detector.get_obstacle_count(), 1)

        # Stop detecting it
        self.mock_radar.set_targets([])
        time.sleep(0.15) # Wait longer than max_obstacle_age
        
        # Obstacle should be gone
        self.assertEqual(self.detector.get_obstacle_count(), 0)

    def test_path_clearance(self):
        """Test the is_path_clear method."""
        # Add a static obstacle
        points = np.random.rand(50, 3) + np.array([20, 0, 0])
        self.mock_lidar.set_point_cloud(points)
        self.detector.set_sensor_interfaces(lidar=self.mock_lidar)
        self.detector.start_detection()
        time.sleep(0.1)

        self.assertEqual(self.detector.get_obstacle_count(), 1)

        # Path 1: Clear path
        start_pos = np.array([0, 0, 0])
        end_pos = np.array([0, 10, 0])
        is_clear, blocking_obstacles = self.detector.is_path_clear(start_pos, end_pos)
        self.assertTrue(is_clear)
        self.assertEqual(len(blocking_obstacles), 0)

        # Path 2: Blocked path
        start_pos = np.array([0, 0, 0])
        end_pos = np.array([40, 0, 0])
        is_clear, blocking_obstacles = self.detector.is_path_clear(start_pos, end_pos)
        self.assertFalse(is_clear)
        self.assertEqual(len(blocking_obstacles), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)