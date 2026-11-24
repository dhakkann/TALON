"""Centralized configuration module for Real-Time Autonomous City Navigation."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


def _get_default_airsim_ip() -> str:
    return os.getenv("AIRSIM_IP", "127.0.0.1")


def _get_default_airsim_port() -> int:
    try:
        return int(os.getenv("AIRSIM_PORT", "41451"))
    except (ValueError, TypeError):
        return 41451


def _get_default_airsim_vehicle_name() -> str:
    return os.getenv("AIRSIM_VEHICLE_NAME", "")


@dataclass
class AirSimConfig:
    """AirSim connection and camera setup."""
    ip: str = field(default_factory=_get_default_airsim_ip)
    port: int = field(default_factory=_get_default_airsim_port)
    vehicle_name: str = field(default_factory=_get_default_airsim_vehicle_name)
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


def _get_default_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                key = winreg.QueryValueEx(k, "GEMINI_API_KEY")[0]
        except Exception:
            pass
    return key


@dataclass
class CognitiveConfig:
    """Gemini Flash client and dialogue settings."""
    model_name: str = "gemini-3.5-flash-lite"
    api_key: str = field(default_factory=_get_default_gemini_key)
    request_timeout_sec: float = 8.0
    narration_cadence_sec: float = 4.0  # cadence-based commentary interval


@dataclass
class DashboardConfig:
    """Web visualization dashboard settings."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    fps_stream_target: int = 25
    enabled: bool = True


@dataclass
class VoiceConfig:
    """Voice commands and speech feedback settings."""
    enabled: bool = True
    push_to_talk: bool = True
    auto_tts_feedback: bool = True
    tts_volume: float = 1.0
    emergency_fastpath: bool = True
    vad_energy_threshold: float = 500.0
    mic_device_index: Optional[int] = None
    mic_sample_rate: int = 16000
    model_name: str = "gemini-3.5-flash"


@dataclass
class EnvironmentConfig:
    """Simulation environment profile and presets."""
    name: str = "city"  # "city", "neighborhood", "blocks"
    description: str = "AirSim CityEnviron dense urban canyon"
    takeoff_altitude: float = 12.0
    cruise_speed: float = 4.0
    obstacle_distance_threshold: float = 4.0
    perception_engine: str = "auto"  # "auto", "coco", "color", "segmentation", "yolo_world"
    default_classes: List[str] = field(default_factory=lambda: [
        "building", "car", "tree", "traffic light", "fountain", "tower", "bridge"
    ])
    default_landmarks: Dict[str, List[float]] = field(default_factory=lambda: {
        "fountain": [32.0, 0.0, 0.0],
        "red building": [75.0, 12.0, 0.0],
        "blue building": [75.0, -14.0, 0.0],
        "tower": [130.0, -10.0, 0.0],
        "hospital": [60.0, -30.0, 0.0],
        "bridge": [85.0, -25.0, 0.0],
        "warehouse": [100.0, 40.0, 0.0],
    })


@dataclass
class SystemConfig:
    """Master configuration encompassing all sub-configs."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    airsim: AirSimConfig = field(default_factory=AirSimConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    
    # Fallback and simulation modes
    mock_airsim: bool = False
    mock_perception: bool = False
    mock_gemini: bool = False


def apply_environment_preset(cfg: SystemConfig, env_name: str, perception_engine: str = "auto") -> SystemConfig:
    """Apply customized environment and perception profile to SystemConfig."""
    normalized = env_name.lower().strip()
    engine = perception_engine.lower().strip()

    if normalized in ("neighborhood", "nh", "suburb", "airsimnh"):
        cfg.env.name = "neighborhood"
        cfg.env.description = "AirSim Neighborhood suburban residential map"
        cfg.flight.takeoff_altitude = 8.0
        cfg.env.takeoff_altitude = 8.0
        cfg.flight.cruise_speed = 3.5
        cfg.env.cruise_speed = 3.5
        cfg.flight.obstacle_distance_threshold = 3.0
        cfg.env.obstacle_distance_threshold = 3.0
        effective_engine = "coco" if engine in ("auto", "") else engine
        cfg.env.perception_engine = effective_engine
        cfg.perception.model_name = "yolov8n.pt"
        cfg.perception.default_classes = [
            "car", "truck", "bus", "stop sign", "traffic light", "bench", "tree", "house"
        ]
        cfg.env.default_classes = cfg.perception.default_classes
        cfg.env.default_landmarks = {
            "white car": [25.0, 8.0, 0.0],
            "stop sign": [40.0, 0.0, 0.0],
            "red car": [55.0, -12.0, 0.0],
            "house": [50.0, 15.0, 0.0],
            "tree": [30.0, -20.0, 0.0],
            "bench": [20.0, -5.0, 0.0],
        }

    elif normalized in ("blocks", "block", "geometric"):
        cfg.env.name = "blocks"
        cfg.env.description = "AirSim Blocks geometric benchmark arena"
        cfg.flight.takeoff_altitude = 5.0
        cfg.env.takeoff_altitude = 5.0
        cfg.flight.cruise_speed = 3.0
        cfg.env.cruise_speed = 3.0
        cfg.flight.obstacle_distance_threshold = 2.5
        cfg.env.obstacle_distance_threshold = 2.5
        effective_engine = "color" if engine in ("auto", "") else engine
        cfg.env.perception_engine = effective_engine
        cfg.perception.model_name = "yolov8n.pt"
        cfg.perception.default_classes = [
            "orange block", "blue cylinder", "green cube", "red pillar", "yellow pyramid", "cylinder", "block"
        ]
        cfg.env.default_classes = cfg.perception.default_classes
        cfg.env.default_landmarks = {
            "orange block": [25.0, 5.0, 0.0],
            "blue cylinder": [45.0, -10.0, 0.0],
            "red pillar": [60.0, 15.0, 0.0],
            "green cube": [35.0, -20.0, 0.0],
            "yellow pyramid": [50.0, 0.0, 0.0],
        }

    else:
        # Default CityEnviron
        cfg.env.name = "city"
        cfg.env.description = "AirSim CityEnviron dense urban canyon"
        cfg.flight.takeoff_altitude = 12.0
        cfg.env.takeoff_altitude = 12.0
        cfg.flight.cruise_speed = 4.0
        cfg.env.cruise_speed = 4.0
        cfg.flight.obstacle_distance_threshold = 4.0
        cfg.env.obstacle_distance_threshold = 4.0
        effective_engine = "yolo_world" if engine in ("auto", "") else engine
        cfg.env.perception_engine = effective_engine
        cfg.perception.model_name = "yolov8s-worldv2.pt" if effective_engine == "yolo_world" else "yolov8n.pt"
        cfg.perception.default_classes = [
            "building", "car", "tree", "traffic light", "fountain", "tower", "bridge"
        ]
        cfg.env.default_classes = cfg.perception.default_classes
        cfg.env.default_landmarks = {
            "fountain": [32.0, 0.0, 0.0],
            "red building": [75.0, 12.0, 0.0],
            "blue building": [75.0, -14.0, 0.0],
            "tower": [130.0, -10.0, 0.0],
            "hospital": [60.0, -30.0, 0.0],
            "bridge": [85.0, -25.0, 0.0],
            "warehouse": [100.0, 40.0, 0.0],
        }

    return cfg


# Default system configuration instance
DEFAULT_CONFIG = SystemConfig()
