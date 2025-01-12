import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class FatigueEvaluator:
    def __init__(self):
        self.setup_bayesian_network()

    def setup_bayesian_network(self):
        # Simplified network focused only on fatigue
        self.model = BayesianNetwork([
            ('TimeOfDay', 'Fatigue'),
            ('TimeSinceRest', 'Fatigue'),
            ('Pulse', 'Fatigue'),
            ('EyelidMovement', 'Fatigue'),
            ('Weather', 'Fatigue'),
            ('Traffic', 'Fatigue'),
            ('RoadType', 'Fatigue')
        ])

        # Basic CPDs
        cpd_time = TabularCPD('TimeOfDay', 2, [[0.2], [0.8]])
        cpd_time_since_rest = TabularCPD('TimeSinceRest', 2, [[0.3], [0.7]])
        cpd_pulse = TabularCPD('Pulse', 2, [[0.3], [0.7]])
        cpd_eyelid = TabularCPD('EyelidMovement', 2, [[0.4], [0.6]])

        # Weather impacts fatigue (5 states)
        cpd_weather = TabularCPD('Weather', 5, [
            [0.5],  # Clear
            [0.2],  # Rain
            [0.15],  # Fog
            [0.1],  # Snow
            [0.05]  # Bad
        ])

        # Traffic impacts fatigue (3 states)
        cpd_traffic = TabularCPD('Traffic', 3, [
            [0.4],  # Low
            [0.4],  # Medium
            [0.2]  # High
        ])

        # Road type impacts fatigue (3 states)
        cpd_road_type = TabularCPD('RoadType', 3, [
            [0.6],  # Highway
            [0.3],  # City
            [0.1]   # Rural
        ])

        fatigue_probs = np.zeros((2, 720))

        for i in range(720):
            base_prob = 0.05

            time_of_day_factor = 0.1 if (i // 360) else 0
            time_since_rest_factor = 0.25 if ((i // 180) % 2) else 0
            pulse_factor = 0.1 if ((i // 90) % 2) else 0
            eyelid_factor = 0.15 if ((i // 45) % 2) else 0
            weather_factor = [0, 0.1, 0.25, 0.3, 0.4][(i // 15) % 5]
            traffic_factor = [0, 0.1, 0.3][(i // 5) % 3]
            road_type_factor = [0, 0.05, 0.2][i % 3]

            # Combine factors
            fatigue_prob = min(1, base_prob + time_of_day_factor + time_since_rest_factor +
                               pulse_factor + eyelid_factor + weather_factor + traffic_factor + road_type_factor)

            fatigue_probs[0, i] = 1 - fatigue_prob
            fatigue_probs[1, i] = fatigue_prob

        cpd_fatigue = TabularCPD(
            'Fatigue', 2,
            fatigue_probs,
            evidence=['TimeOfDay', 'TimeSinceRest', 'Pulse',
                      'EyelidMovement', 'Weather', 'Traffic', 'RoadType'],
            evidence_card=[2, 2, 2, 2, 5, 3, 3]
        )

        self.model.add_cpds(cpd_time, cpd_time_since_rest, cpd_pulse,
                            cpd_eyelid, cpd_weather, cpd_traffic, cpd_road_type, cpd_fatigue)
        self.inference = VariableElimination(self.model)

    def evaluate_fatigue(self, driver_state, time_since_rest_hours):
        evidence = {
            'TimeOfDay': 1 if driver_state.time_of_day == "night" else 0,
            'TimeSinceRest': 1 if time_since_rest_hours > 2 else 0,
            'Pulse': 1 if driver_state.pulse > 85 else 0,
            'EyelidMovement': 1 if driver_state.eyelid_movement < 0.6 else 0,
            'Weather': {
                'clear': 0,
                'rain': 1,
                'fog': 2,
                'snow': 3,
                'bad': 4
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
        print(result)
        return result.values[1]  # Probability of high fatigue
