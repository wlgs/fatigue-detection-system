from dataclasses import dataclass


@dataclass
class DriverState:
    rest_points: float = 100.0
    last_rest_tick: int = 0
    current_speed: float = 0.0
    pulse: float = 70.0
    eyelid_movement: float = 1.0
    temperature: float = 36.6
    weather_condition: str = "clear"
    traffic_density: str = "low"
    time_of_day: str = "day"
    road_type: str = "highway"
    current_rest_loss: float = 0.0
