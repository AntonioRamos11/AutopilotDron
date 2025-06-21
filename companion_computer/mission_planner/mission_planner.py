#!/usr/bin/env python3
"""
Mission Planner - High-level mission planning and coordination
"""

import numpy as np
import yaml
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import logging

class MissionType(Enum):
    WAYPOINT_NAVIGATION = "waypoint_navigation"
    SURVEY = "survey"
    INSPECTION = "inspection"
    SEARCH_RESCUE = "search_rescue"
    DELIVERY = "delivery"
    RETURN_TO_LAUNCH = "rtl"

class FlightMode(Enum):
    MANUAL = "manual"
    STABILIZED = "stabilized"
    ALTITUDE_HOLD = "altitude_hold"
    POSITION_HOLD = "position_hold"
    OFFBOARD = "offboard"
    AUTO_MISSION = "auto_mission"
    AUTO_RTL = "auto_rtl"
    AUTO_LAND = "auto_land"

@dataclass
class Waypoint:
    """Represents a single waypoint in the mission."""
    lat: float
    lon: float
    alt: float
    speed: float = 5.0
    hold_time: float = 0.0
    action: str = "waypoint"
    tolerance: float = 1.0

@dataclass
class Mission:
    """Represents a complete mission."""
    name: str
    mission_type: MissionType
    waypoints: List[Waypoint]
    safety_altitude: float = 50.0
    max_speed: float = 15.0
    return_to_launch: bool = True
    emergency_land_alt: float = 10.0

class MissionPlanner:
    """High-level mission planning and management."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the mission planner."""
        self.logger = logging.getLogger(__name__)
        self.current_mission: Optional[Mission] = None
        self.home_position: Optional[Tuple[float, float, float]] = None
        self.current_position: Optional[Tuple[float, float, float]] = None
        self.flight_mode = FlightMode.MANUAL
        
        # Load configuration
        if config_file:
            self.config = self._load_config(config_file)
        else:
            self.config = self._default_config()
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'safety': {
                'max_altitude': 120.0,
                'min_altitude': 5.0,
                'max_speed': 15.0,
                'geofence_radius': 1000.0,
                'battery_rtl_threshold': 25.0
            },
            'navigation': {
                'waypoint_tolerance': 2.0,
                'loiter_radius': 5.0,
                'climb_rate': 3.0,
                'descent_rate': 2.0
            }
        }
    
    def set_home_position(self, lat: float, lon: float, alt: float):
        """Set the home position for RTL operations."""
        self.home_position = (lat, lon, alt)
        self.logger.info(f"Home position set: {lat:.6f}, {lon:.6f}, {alt:.1f}")
    
    def update_position(self, lat: float, lon: float, alt: float):
        """Update current drone position."""
        self.current_position = (lat, lon, alt)
    
    def create_waypoint_mission(self, waypoints: List[Tuple[float, float, float]], 
                               speeds: Optional[List[float]] = None) -> Mission:
        """Create a waypoint navigation mission."""
        mission_waypoints = []
        
        for i, (lat, lon, alt) in enumerate(waypoints):
            speed = speeds[i] if speeds and i < len(speeds) else 5.0
            waypoint = Waypoint(
                lat=lat, lon=lon, alt=alt, speed=speed,
                tolerance=self.config['navigation']['waypoint_tolerance']
            )
            mission_waypoints.append(waypoint)
        
        mission = Mission(
            name=f"waypoint_mission_{len(mission_waypoints)}_points",
            mission_type=MissionType.WAYPOINT_NAVIGATION,
            waypoints=mission_waypoints,
            max_speed=self.config['safety']['max_speed']
        )
        
        return mission
    
    def create_survey_mission(self, survey_area: List[Tuple[float, float]], 
                             altitude: float, overlap: float = 0.8) -> Mission:
        """Create a survey mission with automatic grid pattern."""
        # Calculate survey grid
        waypoints = self._generate_survey_grid(survey_area, altitude, overlap)
        
        mission_waypoints = []
        for lat, lon, alt in waypoints:
            waypoint = Waypoint(
                lat=lat, lon=lon, alt=alt, speed=8.0,
                action="survey_point"
            )
            mission_waypoints.append(waypoint)
        
        mission = Mission(
            name=f"survey_mission_{len(mission_waypoints)}_points",
            mission_type=MissionType.SURVEY,
            waypoints=mission_waypoints,
            safety_altitude=altitude + 10.0
        )
        
        return mission
    
    def _generate_survey_grid(self, area: List[Tuple[float, float]], 
                             altitude: float, overlap: float) -> List[Tuple[float, float, float]]:
        """Generate a survey grid pattern within the specified area."""
        # This is a simplified grid generation
        # In practice, you'd use more sophisticated algorithms
        
        if len(area) < 3:
            raise ValueError("Survey area must have at least 3 points")
        
        # Find bounding box
        lats = [p[0] for p in area]
        lons = [p[1] for p in area]
        
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Calculate grid spacing based on overlap
        # Assuming camera with 50m ground coverage at this altitude
        ground_coverage = altitude * 0.8  # Simplified calculation
        spacing = ground_coverage * (1 - overlap)
        
        # Convert to approximate degree spacing
        lat_spacing = spacing / 111000  # ~111km per degree latitude
        lon_spacing = spacing / (111000 * np.cos(np.radians(np.mean(lats))))
        
        waypoints = []
        lat = min_lat
        row = 0
        
        while lat <= max_lat:
            if row % 2 == 0:  # Even rows: left to right
                lon = min_lon
                while lon <= max_lon:
                    waypoints.append((lat, lon, altitude))
                    lon += lon_spacing
            else:  # Odd rows: right to left
                lon = max_lon
                while lon >= min_lon:
                    waypoints.append((lat, lon, altitude))
                    lon -= lon_spacing
            
            lat += lat_spacing
            row += 1
        
        return waypoints
    
    def load_mission(self, mission: Mission) -> bool:
        """Load a mission for execution."""
        try:
            # Validate mission
            if not self._validate_mission(mission):
                return False
            
            self.current_mission = mission
            self.logger.info(f"Mission '{mission.name}' loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load mission: {e}")
            return False
    
    def _validate_mission(self, mission: Mission) -> bool:
        """Validate a mission for safety and feasibility."""
        if not mission.waypoints:
            self.logger.error("Mission has no waypoints")
            return False
        
        for i, wp in enumerate(mission.waypoints):
            # Check altitude limits
            if wp.alt > self.config['safety']['max_altitude']:
                self.logger.error(f"Waypoint {i} altitude too high: {wp.alt}m")
                return False
            
            if wp.alt < self.config['safety']['min_altitude']:
                self.logger.error(f"Waypoint {i} altitude too low: {wp.alt}m")
                return False
            
            # Check speed limits
            if wp.speed > self.config['safety']['max_speed']:
                self.logger.error(f"Waypoint {i} speed too high: {wp.speed}m/s")
                return False
        
        # Check geofence if home position is set
        if self.home_position:
            for i, wp in enumerate(mission.waypoints):
                distance = self._calculate_distance(
                    self.home_position[0], self.home_position[1],
                    wp.lat, wp.lon
                )
                if distance > self.config['safety']['geofence_radius']:
                    self.logger.error(f"Waypoint {i} outside geofence: {distance}m")
                    return False
        
        return True
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates (Haversine formula)."""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_current_mission_status(self) -> dict:
        """Get status of the current mission."""
        if not self.current_mission:
            return {"status": "no_mission"}
        
        return {
            "status": "mission_loaded",
            "mission_name": self.current_mission.name,
            "mission_type": self.current_mission.mission_type.value,
            "total_waypoints": len(self.current_mission.waypoints),
            "flight_mode": self.flight_mode.value
        }
    
    def emergency_rtl(self) -> bool:
        """Initiate emergency return to launch."""
        if not self.home_position:
            self.logger.error("Cannot RTL: No home position set")
            return False
        
        # Create emergency RTL mission
        rtl_waypoint = Waypoint(
            lat=self.home_position[0],
            lon=self.home_position[1],
            alt=self.home_position[2] + 20.0,  # RTL at +20m above home
            speed=self.config['safety']['max_speed'],
            action="rtl"
        )
        
        rtl_mission = Mission(
            name="emergency_rtl",
            mission_type=MissionType.RETURN_TO_LAUNCH,
            waypoints=[rtl_waypoint]
        )
        
        self.load_mission(rtl_mission)
        self.flight_mode = FlightMode.AUTO_RTL
        
        self.logger.warning("Emergency RTL initiated")
        return True
    
    def save_mission_to_file(self, mission: Mission, filename: str):
        """Save mission to file."""
        mission_data = {
            'name': mission.name,
            'type': mission.mission_type.value,
            'waypoints': [
                {
                    'lat': wp.lat,
                    'lon': wp.lon,
                    'alt': wp.alt,
                    'speed': wp.speed,
                    'hold_time': wp.hold_time,
                    'action': wp.action,
                    'tolerance': wp.tolerance
                }
                for wp in mission.waypoints
            ],
            'settings': {
                'safety_altitude': mission.safety_altitude,
                'max_speed': mission.max_speed,
                'return_to_launch': mission.return_to_launch,
                'emergency_land_alt': mission.emergency_land_alt
            }
        }
        
        with open(filename, 'w') as f:
            yaml.dump(mission_data, f, default_flow_style=False)
        
        self.logger.info(f"Mission saved to {filename}")
    
    def load_mission_from_file(self, filename: str) -> Optional[Mission]:
        """Load mission from file."""
        try:
            with open(filename, 'r') as f:
                mission_data = yaml.safe_load(f)
            
            waypoints = []
            for wp_data in mission_data['waypoints']:
                waypoint = Waypoint(
                    lat=wp_data['lat'],
                    lon=wp_data['lon'],
                    alt=wp_data['alt'],
                    speed=wp_data.get('speed', 5.0),
                    hold_time=wp_data.get('hold_time', 0.0),
                    action=wp_data.get('action', 'waypoint'),
                    tolerance=wp_data.get('tolerance', 1.0)
                )
                waypoints.append(waypoint)
            
            mission = Mission(
                name=mission_data['name'],
                mission_type=MissionType(mission_data['type']),
                waypoints=waypoints,
                safety_altitude=mission_data['settings'].get('safety_altitude', 50.0),
                max_speed=mission_data['settings'].get('max_speed', 15.0),
                return_to_launch=mission_data['settings'].get('return_to_launch', True),
                emergency_land_alt=mission_data['settings'].get('emergency_land_alt', 10.0)
            )
            
            self.logger.info(f"Mission loaded from {filename}")
            return mission
            
        except Exception as e:
            self.logger.error(f"Failed to load mission from {filename}: {e}")
            return None