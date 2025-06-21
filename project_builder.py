#!/usr/bin/env python3
"""
Autopilot Drone Project Builder
Creates the complete project structure based on the architecture diagram
"""

import os
import shutil
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure."""
    
    base_dir = Path("/home/p0wden/Documents/Autopilot drone")
    
    # Define the project structure
    structure = {
        "companion_computer": {
            "mission_planner": ["__init__.py", "mission_planner.py", "waypoint_manager.py", "flight_modes.py"],
            "trajectory_optimization": ["__init__.py", "trajectory_optimizer.py", "path_smoother.py"],
            "obstacle_avoidance": ["__init__.py", "obstacle_detector.py", "avoidance_planner.py", "sensor_fusion.py"],
            "mavlink_router": ["__init__.py", "mavlink_interface.py", "message_handler.py", "telemetry.py"],
            "sensors": ["__init__.py", "depth_camera.py", "lidar_interface.py", "radar_interface.py"]
        },
        "flight_controller": {
            "mission_handling": ["__init__.py", "mission_executor.py", "command_processor.py"],
            "state_estimation": ["__init__.py", "ekf2_interface.py", "sensor_manager.py", "kalman_filter.py"],
            "low_level_control": ["__init__.py", "attitude_controller.py", "rate_controller.py", "position_controller.py"],
            "sensors": ["__init__.py", "rtk_gps.py", "imu_interface.py", "barometer.py", "optical_flow.py"],
            "actuators": ["__init__.py", "motor_controller.py", "servo_controller.py"]
        },
        "integration": {
            "communication": ["__init__.py", "mavlink_bridge.py", "data_logger.py"],
            "testing": ["__init__.py", "system_tests.py", "integration_tests.py", "simulation_tests.py"],
            "configuration": ["px4_params.yaml", "system_config.yaml", "sensor_config.yaml"]
        },
        "simulation": {
            "models": ["__init__.py", "drone_dynamics.py", "sensor_models.py", "environment.py"],
            "scenarios": ["__init__.py", "test_scenarios.py", "benchmark_scenarios.py"]
        },
        "utils": {
            "common": ["__init__.py", "math_utils.py", "coordinate_transforms.py", "filters.py"],
            "visualization": ["__init__.py", "plot_tools.py", "3d_visualizer.py"],
            "logging": ["__init__.py", "logger_config.py", "data_recorder.py"]
        },
        "scripts": ["launch_system.py", "calibration.py", "system_check.py", "mission_upload.py"],
        "config": ["main_config.yaml", "calibration_data.yaml"],
        "docs": ["README.md", "API_REFERENCE.md", "INSTALLATION.md", "USER_GUIDE.md"]
    }
    
    # Create directories and files
    for main_dir, subdirs in structure.items():
        main_path = base_dir / main_dir
        main_path.mkdir(exist_ok=True)
        
        if isinstance(subdirs, dict):
            for subdir, files in subdirs.items():
                subdir_path = main_path / subdir
                subdir_path.mkdir(exist_ok=True)
                
                for file in files:
                    file_path = subdir_path / file
                    if not file_path.exists():
                        file_path.touch()
        elif isinstance(subdirs, list):
            for file in subdirs:
                file_path = main_path / file
                if not file_path.exists():
                    file_path.touch()
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()