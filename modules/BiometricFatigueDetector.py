from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import math


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

        # CPDs for biometric measurements with multiple states based on simulator ranges
        cpd_heart_rate = TabularCPD(
            'HeartRate', 4, [
                [0.4],  # Normal/Rested (70-80 bpm)
                [0.3],  # Slightly Fatigued (65-70 bpm)
                [0.2],  # Fatigued (60-65 bpm)
                [0.1]   # Very Fatigued (<60 bpm)
            ])

        cpd_hrv = TabularCPD(
            'HRV', 4, [
                [0.4],  # Normal/Rested (45-55 ms)
                [0.3],  # Slightly Fatigued (35-45 ms)
                [0.2],  # Fatigued (25-35 ms)
                [0.1]   # Very Fatigued (<25 ms)
            ])

        cpd_eda = TabularCPD(
            'EDA', 4, [
                [0.4],  # Normal/Rested (6-8 μS)
                [0.3],  # Slightly Fatigued (4-6 μS)
                [0.2],  # Fatigued (2-4 μS)
                [0.1]   # Very Fatigued (<2 μS)
            ])

        cpd_perclos = TabularCPD(
            'PERCLOS', 4, [
                [0.4],  # Normal/Rested (0.10-0.20)
                [0.3],  # Slightly Fatigued (0.20-0.30)
                [0.2],  # Fatigued (0.30-0.40)
                [0.1]   # Very Fatigued (>0.40)
            ])

        cpd_blink_duration = TabularCPD(
            'BlinkDuration', 4, [
                [0.4],  # Normal/Rested (150-250 ms)
                [0.3],  # Slightly Fatigued (250-350 ms)
                [0.2],  # Fatigued (350-450 ms)
                [0.1]   # Very Fatigued (>450 ms)
            ])

        cpd_blink_rate = TabularCPD(
            'BlinkRate', 4, [
                [0.4],  # Normal/Rested (10-14 bpm)
                [0.3],  # Slightly Fatigued (14-18 bpm)
                [0.2],  # Fatigued (18-22 bpm)
                [0.1]   # Very Fatigued (>22 bpm)
            ])

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
            # All variables now have 4 states
            evidence_card=[4, 4, 4, 4, 4, 4]
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
        # All variables now have 4 states
        cards = [4, 4, 4, 4, 4, 4]  # Cardinality of each node
        num_combinations = np.prod(cards)  # Should be 4096 (4^6)
        probs = np.zeros((2, num_combinations))

        # Helper function to convert flat index to state indices
        def index_to_states(idx, cards):
            states = []
            for card in reversed(cards):
                states.append(idx % card)
                idx //= card
            return list(reversed(states))

        # Weights for each state (Normal->Very Fatigued)
        state_weights = [0.1, 0.4, 0.7, 0.9]

        for i in range(num_combinations):
            states = index_to_states(i, cards)

            # Calculate fatigue contribution from each metric
            fatigue_contrib = 0.0

            # Heart Rate contribution
            fatigue_contrib += state_weights[states[0]] * weights['heart_rate']

            # HRV contribution
            fatigue_contrib += state_weights[states[1]] * weights['hrv']

            # EDA contribution
            fatigue_contrib += state_weights[states[2]] * weights['eda']

            # PERCLOS contribution
            fatigue_contrib += state_weights[states[3]] * weights['perclos']

            # Blink Duration contribution
            fatigue_contrib += state_weights[states[4]
                                             ] * weights['blink_duration']

            # Blink Rate contribution
            fatigue_contrib += state_weights[states[5]] * weights['blink_rate']

            # Normalize and store probabilities
            fatigue_prob = min(1.0, fatigue_contrib)
            probs[0, i] = 1 - fatigue_prob  # Not fatigued
            probs[1, i] = fatigue_prob      # Fatigued

        return probs

        return probs

    def _get_biometric_states(self, driver_state):
        """Convert biometric measurements to discrete states"""
        states = {}

        # Heart Rate (4 states based on simulator ranges)
        if driver_state.heart_rate >= 70 and driver_state.heart_rate <= 80:
            states['HeartRate'] = 0  # Normal/Rested
        elif driver_state.heart_rate >= 65 and driver_state.heart_rate < 70:
            states['HeartRate'] = 1  # Slightly Fatigued
        elif driver_state.heart_rate >= 60 and driver_state.heart_rate < 65:
            states['HeartRate'] = 2  # Fatigued
        else:
            states['HeartRate'] = 3  # Very Fatigued

        # HRV (4 states based on simulator ranges)
        if driver_state.hrv >= 45 and driver_state.hrv <= 55:
            states['HRV'] = 0  # Normal/Rested
        elif driver_state.hrv >= 35 and driver_state.hrv < 45:
            states['HRV'] = 1  # Slightly Fatigued
        elif driver_state.hrv >= 25 and driver_state.hrv < 35:
            states['HRV'] = 2  # Fatigued
        else:
            states['HRV'] = 3  # Very Fatigued

        # EDA (4 states based on simulator ranges)
        if driver_state.eda >= 6 and driver_state.eda <= 8:
            states['EDA'] = 0  # Normal/Rested
        elif driver_state.eda >= 4 and driver_state.eda < 6:
            states['EDA'] = 1  # Slightly Fatigued
        elif driver_state.eda >= 2 and driver_state.eda < 4:
            states['EDA'] = 2  # Fatigued
        else:
            states['EDA'] = 3  # Very Fatigued

        # PERCLOS (4 states based on simulator ranges)
        if driver_state.perclos >= 0.10 and driver_state.perclos <= 0.20:
            states['PERCLOS'] = 0  # Normal/Rested
        elif driver_state.perclos > 0.20 and driver_state.perclos <= 0.30:
            states['PERCLOS'] = 1  # Slightly Fatigued
        elif driver_state.perclos > 0.30 and driver_state.perclos <= 0.40:
            states['PERCLOS'] = 2  # Fatigued
        else:
            states['PERCLOS'] = 3  # Very Fatigued

        # Blink Duration (4 states based on simulator ranges)
        if driver_state.blink_duration >= 150 and driver_state.blink_duration <= 250:
            states['BlinkDuration'] = 0  # Normal/Rested
        elif driver_state.blink_duration > 250 and driver_state.blink_duration <= 350:
            states['BlinkDuration'] = 1  # Slightly Fatigued
        elif driver_state.blink_duration > 350 and driver_state.blink_duration <= 450:
            states['BlinkDuration'] = 2  # Fatigued
        else:
            states['BlinkDuration'] = 3  # Very Fatigued

        # Blink Rate (4 states based on simulator ranges)
        if driver_state.blink_rate >= 10 and driver_state.blink_rate <= 14:
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

        # Set evidence states based on continuous probabilities
        evidence = {
            metric: 1 if prob > 0.5 else 0
            for metric, prob in states.items()
        }

        # Query the Bayesian network for fatigue and alarm probabilities
        # Query the Bayesian network for fatigue probability first
        fatigue_result = self.inference.query(['Fatigue'], evidence=evidence)
        fatigue_prob = fatigue_result.values[1]  # Probability of high fatigue
        print(fatigue_result)

        # Then query for alarm probability using fatigue as evidence
        alarm_evidence = evidence.copy()
        alarm_evidence['Fatigue'] = 1 if fatigue_prob > 0.5 else 0
        alarm_result = self.inference.query(['Alarm'], evidence=alarm_evidence)
        print(alarm_result)
        alarm_prob = alarm_result.values[1]    # Probability of alarm

        return fatigue_prob, alarm_prob, alarm_prob > self.ALARM_THRESHOLD
