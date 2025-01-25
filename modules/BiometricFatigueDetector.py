from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import math


class BiometricFatigueDetector:
    """Detects driver fatigue based on biometric measurements using Bayesian networks"""

    def __init__(self):
        self.tick_fatigue_contributors = {
            'HeartRate': 0,
            'HRV': 0,
            'EDA': 0,
            'PERCLOS': 0,
            'BlinkDuration': 0,
            'BlinkRate': 0
        }
        self.WEIGHTS = {
            'HeartRate': 1,
            'HRV': 1,
            'EDA': 1,
            'PERCLOS': 4,
            'BlinkDuration': 4,
            'BlinkRate': 2
        }
        self._normalize_weights()


        self.setup_bayesian_network()
        self.ALARM_THRESHOLD = 0.60

    def _normalize_weights(self):
        total_weight = sum(self.WEIGHTS.values())
        for key in self.WEIGHTS:
            self.WEIGHTS[key] /= total_weight

    def getBiometricFatigueContributors(self):
        return self.tick_fatigue_contributors

    def setup_bayesian_network(self):
        self.model = BayesianNetwork([
            ('HeartRate', 'Fatigue'),
            ('HRV', 'Fatigue'),
            ('EDA', 'Fatigue'),
            ('PERCLOS', 'Fatigue'),
            ('BlinkDuration', 'Fatigue'),
            ('BlinkRate', 'Fatigue'),
        ])

        # CPDs for biometric measurements with multiple states based on simulator ranges
        cpd_heart_rate = TabularCPD(
            'HeartRate', 4, [
                [0.1],  # Normal/Rested (70-80 bpm)
                [0.2],  # Slightly Fatigued (65-70 bpm)
                [0.3],  # Fatigued (60-65 bpm)
                [0.4]   # Very Fatigued (<60 bpm)
            ])

        cpd_hrv = TabularCPD(
            'HRV', 4, [
                [0.1],  # Normal/Rested (45-55 ms)
                [0.2],  # Slightly Fatigued (35-45 ms)
                [0.3],  # Fatigued (25-35 ms)
                [0.4]   # Very Fatigued (<25 ms)
            ])

        cpd_eda = TabularCPD(
            'EDA', 4, [
                [0.1],  # Normal/Rested (6-8 μS)
                [0.2],  # Slightly Fatigued (4-6 μS)
                [0.3],  # Fatigued (2-4 μS)
                [0.4]   # Very Fatigued (<2 μS)
            ])

        cpd_perclos = TabularCPD(
            'PERCLOS', 4, [
                [0.05],  # Normal/Rested (0.10-0.20)
                [0.1],  # Slightly Fatigued (0.20-0.30)
                [0.25],  # Fatigued (0.30-0.40)
                [0.6]   # Very Fatigued (>0.40)
            ])

        cpd_blink_duration = TabularCPD(
            'BlinkDuration', 4, [
                [0.05],  # Normal/Rested (150-250 ms)
                [0.15],  # Slightly Fatigued (250-350 ms)
                [0.25],  # Fatigued (350-450 ms)
                [0.55]   # Very Fatigued (>450 ms)
            ])

        cpd_blink_rate = TabularCPD(
            'BlinkRate', 4, [
                [0.05],  # Normal/Rested (10-14 bpm)
                [0.1],  # Slightly Fatigued (14-18 bpm)
                [0.3],  # Fatigued (18-22 bpm)
                [0.55]   # Very Fatigued (>22 bpm)
            ])

        # Weights for biometric factors - adjusted to emphasize important indicators

        # Calculate fatigue probabilities
        fatigue_probs = self._calculate_fatigue_probabilities(
            [cpd_heart_rate, cpd_hrv, cpd_eda, cpd_perclos,
                cpd_blink_duration, cpd_blink_rate],
            self.WEIGHTS
        )

        cpd_fatigue = TabularCPD(
            'Fatigue', 2,
            fatigue_probs,
            evidence=['HeartRate', 'HRV', 'EDA',
                      'PERCLOS', 'BlinkDuration', 'BlinkRate'],
            evidence_card=[4, 4, 4, 4, 4, 4]
        )

        self.model.add_cpds(cpd_heart_rate, cpd_hrv, cpd_eda, cpd_perclos,
                            cpd_blink_duration, cpd_blink_rate, cpd_fatigue)
        self.inference = VariableElimination(self.model)

    def _calculate_fatigue_probabilities(self, cpds, weights):
        """Calculate fatigue probabilities based on input CPDs and weights"""
        cards = [4, 4, 4, 4, 4, 4]
        num_combinations = np.prod(cards)  # Should be 4096 (4^6)
        probs = np.zeros((2, num_combinations))

        def index_to_states(idx, cards):
            states = []
            for card in reversed(cards):
                states.append(idx % card)
                idx //= card
            return list(reversed(states))

        FATIGUE_PROB_FACTOR_MULTIPLIER = 4.5
        for i in range(num_combinations):
            states = index_to_states(i, cards)
            fatigue_contrib = 0.0

            for j, cpd in enumerate(cpds):
                fatigue_value = float(cpd.get_values()[states[j]]) * weights[list(weights.keys())[j]]
                fatigue_contrib += fatigue_value * FATIGUE_PROB_FACTOR_MULTIPLIER
            fatigue_prob = min(1.0, fatigue_contrib)
            probs[0, i] = 1 - fatigue_prob
            probs[1, i] = fatigue_prob

        return probs

    def _get_biometric_states(self, driver_state):
        """Convert biometric measurements to discrete states"""
        states = {}

        if driver_state.heart_rate >= 72:
            states['HeartRate'] = 0  # Normal/Rested
        elif driver_state.heart_rate >= 67 and driver_state.heart_rate < 72:
            states['HeartRate'] = 1  # Slightly Fatigued
        elif driver_state.heart_rate >= 62 and driver_state.heart_rate < 67:
            states['HeartRate'] = 2  # Fatigued
        else:
            states['HeartRate'] = 3  # Very Fatigued

        if driver_state.hrv >= 45:
            states['HRV'] = 0  # Normal/Rested
        elif driver_state.hrv >= 35 and driver_state.hrv < 45:
            states['HRV'] = 1  # Slightly Fatigued
        elif driver_state.hrv >= 25 and driver_state.hrv < 35:
            states['HRV'] = 2  # Fatigued
        else:
            states['HRV'] = 3  # Very Fatigued

        if driver_state.eda >= 6 and driver_state.eda <= 8:
            states['EDA'] = 0  # Normal/Rested
        elif driver_state.eda >= 4 and driver_state.eda < 6:
            states['EDA'] = 1  # Slightly Fatigued
        elif driver_state.eda >= 2 and driver_state.eda < 4:
            states['EDA'] = 2  # Fatigued
        else:
            states['EDA'] = 3  # Very Fatigued

        if driver_state.perclos < 0.15:
            states['PERCLOS'] = 0  # Normal/Rested
        elif driver_state.perclos >= 0.15 and driver_state.perclos < 0.25:
            states['PERCLOS'] = 1  # Slightly Fatigued
        elif driver_state.perclos >= 0.25 and driver_state.perclos < 0.35:
            states['PERCLOS'] = 2  # Fatigued
        else:
            states['PERCLOS'] = 3  # Very Fatigued

        if driver_state.blink_duration >= 0 and driver_state.blink_duration <= 250:
            states['BlinkDuration'] = 0  # Normal/Rested
        elif driver_state.blink_duration > 250 and driver_state.blink_duration <= 350:
            states['BlinkDuration'] = 1  # Slightly Fatigued
        elif driver_state.blink_duration > 350 and driver_state.blink_duration <= 450:
            states['BlinkDuration'] = 2  # Fatigued
        else:
            states['BlinkDuration'] = 3  # Very Fatigued

        if driver_state.blink_rate >= 0 and driver_state.blink_rate <= 14:
            states['BlinkRate'] = 0  # Normal/Rested
        elif driver_state.blink_rate > 14 and driver_state.blink_rate <= 18:
            states['BlinkRate'] = 1  # Slightly Fatigued
        elif driver_state.blink_rate > 18 and driver_state.blink_rate <= 22:
            states['BlinkRate'] = 2  # Fatigued
        else:
            states['BlinkRate'] = 3  # Very Fatigued
        return states

    def detect_fatigue(self, driver_state):
        """
        Evaluates driver fatigue using Bayesian network and continuous probabilities
        Returns: (fatigue_level, alarm_probability, alarm_triggered)
        """
        states = self._get_biometric_states(driver_state)
        evidence = {
            metric: prob for metric, prob in states.items()
        }

        fatigue_result = self.inference.query(['Fatigue'], evidence=evidence)
        # get values of individual states and put them in a dictionary
        for metric, state in states.items():
            cpd = self.model.get_cpds(metric)
            self.tick_fatigue_contributors[metric] = float(cpd.values[state]) * self.WEIGHTS[metric] * 4.5
        fatigue_prob = fatigue_result.values[1]

        return fatigue_prob, fatigue_prob, fatigue_prob >= self.ALARM_THRESHOLD
