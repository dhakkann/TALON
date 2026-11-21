"""Centralized configuration module for Real-Time Autonomous City Navigation."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AirSimConfig:
    """AirSim connection and camera setup."""
    ip: str = "127.0.0.1"
    port: int = 41451
    vehicle_name: str = ""
    camera_name: str = "front_center"
    fov_degrees: float = 90.0
    image_width: int = 640
    image_height: int = 480
    min_depth_meters: float = 0.3
    max_depth_meters: float = 80.0
    connect_timeout_sec: float = 3.0


@dataclass
class FlightConfig:
    """Continuous velocity flight controller and APF parameters."""
    control_rate_hz: float = 20.0
    cruise_speed: float = 4.0          # m/s
    max_speed: float = 6.0             # m/s
    takeoff_altitude: float = 5.0      # meters (NED z = -5.0)
    waypoint_arrival_radius: float = 1.8 # meters
    proportional_gain: float = 1.2
    
    # Reactive Artificial Potential Field (APF) obstacle avoidance
    obstacle_distance_threshold: float = 4.0  # meters
    obstacle_critical_ratio: float = 0.15     # trigger repulsion if >15% pixels close
    repulsion_lateral_speed: float = 2.0      # m/s slide
    repulsion_reverse_speed: float = 1.5      # m/s back off
    velocity_cmd_duration: float = 0.1        # seconds (duration for moveByVelocityAsync)


@dataclass
class PerceptionConfig:
    """YOLO-World v2 open-vocabulary perception settings."""
    model_name: str = "yolov8s-worldv2.pt"
    confidence_threshold: float = 0.25
    depth_patch_radius: int = 2        # 5x5 median patch (radius 2)
    device: str = "cuda"               # fallback to cpu if cuda unavailable
    default_classes: List[str] = field(
        default_factory=lambda: [
            "building", "car", "tree", "traffic light", "bus stop",
            "statue", "fountain", "bench", "bridge", "street lamp"
        ]
    )


@dataclass
class MemoryConfig:
    """Persistent 3D spatial memory registry settings."""
    cluster_threshold: float = 5.0     # meters Euclidean distance
    position_filter_alpha: float = 0.2  # running average weight for new observation
    max_retention_sec: float = 3600.0   # keep landmarks up to 1 hour


@dataclass
class CognitiveConfig:
    """Gemini 2.0 Flash client and dialogue settings."""
    model_name: str = "gemini-2.0-flash"
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    request_timeout_sec: float = 8.0
    narration_cadence_sec: float = 4.0  # cadence-based commentary interval


@dataclass
class DashboardConfig:
    """Web visualization dashboard settings."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    fps_stream_target: int = 25


@dataclass
class SystemConfig:
    """Master configuration encompassing all sub-configs."""
    airsim: AirSimConfig = field(default_factory=AirSimConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Fallback and simulation modes
    mock_airsim: bool = False
    mock_perception: bool = False
    mock_gemini: bool = False


# Default system configuration instance
DEFAULT_CONFIG = SystemConfig()
