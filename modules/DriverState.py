from dataclasses import dataclass
from typing import Literal
from dataclasses import field

ScenarioType = Literal["normal", "bad", "worst"]


@dataclass
class DriverState:
    scenario: ScenarioType = "normal"

    # Simulation state
    rest_points: float = 100.0
    last_rest_tick: int = 0
    current_speed: float = 0.0
    last_drive_tick_time: int = 0

    # Environment state
    weather_condition: str = field(init=False)
    traffic_density: str = field(init=False)
    time_of_day: str = field(init=False)
    road_type: str = field(init=False)

    # Smartband metrics
    heart_rate: float = 75.0  # bpm (60-100 normal)
    hrv: float = 50.0  # ms (>30 normal)
    eda: float = 5.0  # μS (microsiemens)

    # Camera metrics
    perclos: float = 0.15  # percentage (0-1)
    blink_duration: float = 200.0  # ms
    blink_rate: float = 15.0  # blinks per minute

    def __post_init__(self):
        environment_scenarios = {
            "normal": {
                "weather_condition": "clear",
                "traffic_density": "low",
                "time_of_day": "day",
                "road_type": "highway"
            },
            "worst": {
                "weather_condition": "snow",
                "traffic_density": "high",
                "time_of_day": "night",
                "road_type": "rural"
            },
            "bad": {
                "weather_condition": "rain",
                "traffic_density": "high",
                "time_of_day": "day",
                "road_type": "city"
            },
        }

        # Set only environment attributes based on the selected scenario
        scenario_settings = environment_scenarios[self.scenario]
        for key, value in scenario_settings.items():
            setattr(self, key, value)
