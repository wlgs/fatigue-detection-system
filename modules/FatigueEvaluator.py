import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class FatigueEvaluator:
    def __init__(self):
        self.setup_bayesian_network()

    def setup_bayesian_network(self):
        self.model = BayesianNetwork([
            ('TimeOfDay', 'Fatigue'),
            ('TimeSinceRest', 'Fatigue'),
            ('Weather', 'Fatigue'),
            ('Traffic', 'Fatigue'),
            ('RoadType', 'Fatigue')
        ])

        # Basic CPDs
        cpd_time = TabularCPD('TimeOfDay', 2, [[0.2], [0.8]])  # Day, night
        cpd_time_since_rest = TabularCPD(
            'TimeSinceRest', 2, [[0.2], [0.8]])  # <2 hours, >2 hours

        # Weather impacts fatigue (5 states)
        cpd_weather = TabularCPD('Weather', 5, [
            [0],  # Clear
            [0.2],  # Rain
            [0.35],  # Fog
            [0.30],  # Snow
            [0.15]  # Sunny
        ])

        # Traffic impacts fatigue (3 states)
        cpd_traffic = TabularCPD('Traffic', 3, [
            [0.1],  # Low
            [0.3],  # Medium
            [0.6]  # High
        ])

        # Road type impacts fatigue (3 states)
        cpd_road_type = TabularCPD('RoadType', 3, [
            [0.1],  # Highway
            [0.3],  # City
            [0.6]   # Rural
        ])

        WEIGHTS = {
            'time_of_day': 0.25,
            'time_since_rest': 0.3,
            'weather': 0.15,
            'traffic': 0.15,
            'road': 0.15
        }

        fatigue_probs = np.zeros((2, 180))

        for i in range(180):
            # Get state indices for current combination
            road_state = i % 3
            traffic_state = (i // 3) % 3
            weather_state = (i // 9) % 5
            rest_state = (i // 45) % 2
            time_state = (i // 90) % 2

            # Calculate contribution from each factor using CPD values
            time_factor = float(cpd_time.get_values()[
                                time_state]) * WEIGHTS['time_of_day']
            rest_factor = float(cpd_time_since_rest.get_values()[
                                rest_state]) * WEIGHTS['time_since_rest']
            weather_factor = float(cpd_weather.get_values()[
                                   weather_state]) * WEIGHTS['weather']
            traffic_factor = float(cpd_traffic.get_values()[
                                   traffic_state]) * WEIGHTS['traffic']
            road_factor = float(cpd_road_type.get_values()[
                                road_state]) * WEIGHTS['road']

            fatigue_prob = min(1.0, (
                time_factor +
                rest_factor +
                weather_factor +
                traffic_factor +
                road_factor
            ))

            # Store probabilities for both states (not fatigued, fatigued)
            fatigue_probs[0, i] = 1 - fatigue_prob
            fatigue_probs[1, i] = fatigue_prob

        cpd_fatigue = TabularCPD(
            'Fatigue', 2,
            fatigue_probs,
            evidence=['TimeOfDay', 'TimeSinceRest',
                      'Weather', 'Traffic', 'RoadType'],
            evidence_card=[2, 2, 5, 3, 3]
        )

        self.model.add_cpds(cpd_time, cpd_time_since_rest,
                            cpd_weather, cpd_traffic, cpd_road_type, cpd_fatigue)
        self.inference = VariableElimination(self.model)

    def get_model(self):
        return self.model

    def evaluate_fatigue(self, driver_state, time_since_rest_hours):
        evidence = {
            'TimeOfDay': 1 if driver_state.time_of_day == "night" else 0,
            'TimeSinceRest': 1 if time_since_rest_hours > 2 else 0,
            'Weather': {
                'clear': 0,
                'rain': 1,
                'fog': 2,
                'snow': 3,
                'sunny': 4
            }[driver_state.weather_condition],
            'Traffic': {
                'low': 0,
                'medium': 1,
                'high': 2
            }[driver_state.traffic_density],
            'RoadType': {
                'highway': 0,
                'city': 1,
                'rural': 2
            }[driver_state.road_type]
        }

        result = self.inference.query(['Fatigue'], evidence=evidence)
        return result.values[1]  # Probability of high fatigue
