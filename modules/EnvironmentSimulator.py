import random


class EnvironmentSimulator:
    def __init__(self):
        self.weather_change_prob = 1/72
        self.weather_probabilities = {
            "clear": 0.5,
            "rain": 0.1,
            "fog": 0.1,
            "snow": 0.1,
            "sunny": 0.2,
        }
        self.traffic_change_prob = 1/36
        self.traffic_probabilities = {
            "low": 0.4,
            "medium": 0.4,
            "high": 0.2
        }
        self.road_change_prob = 1/36
        self.road_probabilities = {
            "highway": 0.6,
            "city": 0.3,
            "rural": 0.1
        }

    def simulate_weather(self, current_weather):
        if random.random() < self.weather_change_prob:
            return random.choices(
                list(self.weather_probabilities.keys()),
                list(self.weather_probabilities.values())
            )[0]
        return current_weather

    def simulate_road_type(self, current_road_type):
        if random.random() < self.road_change_prob:
            return random.choices(
                list(self.road_probabilities.keys()),
                list(self.road_probabilities.values())
            )[0]
        return current_road_type

    def simulate_traffic(self, current_traffic):
        if random.random() < self.traffic_change_prob:
            return random.choices(
                list(self.traffic_probabilities.keys()),
                list(self.traffic_probabilities.values())
            )[0]
        return current_traffic

    def simulate_time_of_day(self,
                             current_time_of_day, tick_count):
        if tick_count % 288 == 0:
            return "night" if current_time_of_day == "day" else "day"
        return current_time_of_day
