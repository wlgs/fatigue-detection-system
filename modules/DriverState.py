from dataclasses import dataclass


@dataclass
class DriverState:
    # Simulation state
    rest_points: float = 100.0
    last_rest_tick: int = 0
    current_speed: float = 0.0
    last_drive_tick_time: int = 0
    
    # Environment state
    weather_condition: str = "clear"
    traffic_density: str = "low"
    time_of_day: str = "day"
    road_type: str = "highway"
    
    # Smartband metrics
    heart_rate: float = 75.0  # bpm (60-100 normal)
    hrv: float = 50.0  # ms (>30 normal)
    eda: float = 5.0  # μS (microsiemens)
    
    # Camera metrics
    perclos: float = 0.15  # percentage (0-1)
    blink_duration: float = 200.0  # ms
    blink_rate: float = 15.0  # blinks per minute