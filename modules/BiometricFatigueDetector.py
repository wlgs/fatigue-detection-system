from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np


class BiometricFatigueDetector:
    """Detects driver fatigue based on biometric measurements using Bayesian networks"""

    def __init__(self):
        self.setup_bayesian_network()
        self.ALARM_THRESHOLD = 0.7  # Threshold for triggering fatigue alarm

    def setup_bayesian_network(self):
        self.model = BayesianNetwork([
            ('HeartRate', 'Fatigue'),
            ('HRV', 'Fatigue'),
            ('EDA', 'Fatigue'),
            ('PERCLOS', 'Fatigue'),
            ('BlinkDuration', 'Fatigue'),
            ('BlinkRate', 'Fatigue'),
            ('Fatigue', 'Alarm')
        ])

        # CPDs for biometric measurements
        cpd_heart_rate = TabularCPD(
            'HeartRate', 2, [[0.7], [0.3]])  # Normal, Low
        cpd_hrv = TabularCPD('HRV', 2, [[0.7], [0.3]])  # Normal, Low
        cpd_eda = TabularCPD('EDA', 2, [[0.7], [0.3]])  # Normal, Low
        cpd_perclos = TabularCPD('PERCLOS', 2, [[0.6], [0.4]])  # Normal, High
        cpd_blink_duration = TabularCPD(
            'BlinkDuration', 2, [[0.6], [0.4]])  # Normal, High
        cpd_blink_rate = TabularCPD(
            'BlinkRate', 2, [[0.6], [0.4]])  # Normal, High

        # Weights for biometric factors
        WEIGHTS = {
            'heart_rate': 0.15,
            'hrv': 0.20,
            'eda': 0.15,
            'perclos': 0.20,
            'blink_duration': 0.15,
            'blink_rate': 0.15
        }

        # Calculate fatigue probabilities
        fatigue_probs = self._calculate_fatigue_probabilities(
            [cpd_heart_rate, cpd_hrv, cpd_eda, cpd_perclos,
                cpd_blink_duration, cpd_blink_rate],
            WEIGHTS
        )

        cpd_fatigue = TabularCPD(
            'Fatigue', 2,
            fatigue_probs,
            evidence=['HeartRate', 'HRV', 'EDA',
                      'PERCLOS', 'BlinkDuration', 'BlinkRate'],
            evidence_card=[2, 2, 2, 2, 2, 2]
        )

        # Alarm CPD based on fatigue level
        cpd_alarm = TabularCPD(
            'Alarm', 2,
            [[0.9, 0.1],   # No alarm probabilities when not fatigued/fatigued
             [0.1, 0.9]],  # Alarm probabilities when not fatigued/fatigued
            evidence=['Fatigue'],
            evidence_card=[2]
        )

        self.model.add_cpds(cpd_heart_rate, cpd_hrv, cpd_eda, cpd_perclos,
                            cpd_blink_duration, cpd_blink_rate, cpd_fatigue, cpd_alarm)
        self.inference = VariableElimination(self.model)

    def _calculate_fatigue_probabilities(self, cpds, weights):
        """Calculate fatigue probabilities based on input CPDs and weights"""
        num_combinations = 2 ** len(cpds)
        probs = np.zeros((2, num_combinations))

        for i in range(num_combinations):
            # Convert i to binary array representing states
            states = [(i >> j) & 1 for j in range(len(cpds))]

            # Calculate weighted sum of probabilities
            fatigue_prob = sum(
                float(cpd.get_values()[state]) * weight
                for cpd, state, (_, weight) in zip(cpds, states, weights.items())
            )

            fatigue_prob = min(1.0, fatigue_prob)
            probs[0, i] = 1 - fatigue_prob  # Not fatigued
            probs[1, i] = fatigue_prob      # Fatigued

        return probs

    def _get_biometric_states(self, driver_state):
        """Convert biometric measurements to binary states"""
        return {
            'HeartRate': 1 if driver_state.heart_rate < 60 else 0,
            'HRV': 1 if driver_state.hrv < 30 else 0,
            'EDA': 1 if driver_state.eda < 3.0 else 0,
            'PERCLOS': 1 if driver_state.perclos > 0.4 else 0,
            'BlinkDuration': 1 if driver_state.blink_duration > 500 else 0,
            'BlinkRate': 1 if driver_state.blink_rate > 25 else 0
        }

    def detect_fatigue(self, driver_state):
        """
        Evaluates driver fatigue using Bayesian network
        Returns: (fatigue_level, alarm_triggered)
        """
        evidence = self._get_biometric_states(driver_state)

        # Query fatigue probability
        fatigue_result = self.inference.query(['Fatigue'], evidence=evidence)
        fatigue_level = fatigue_result.values[1]

        # Query alarm probability
        alarm_result = self.inference.query(['Alarm'], evidence=evidence)
        alarm_probability = alarm_result.values[1]

        return fatigue_level, alarm_probability, alarm_probability > self.ALARM_THRESHOLD
