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

        # CPDs for biometric measurements with multiple states
        cpd_heart_rate = TabularCPD(
            'HeartRate', 3, [
                [0.7],  # Normal (60-100)
                [0.2],  # Low (<60)
                [0.1]   # High (>100)
            ])

        cpd_hrv = TabularCPD(
            'HRV', 3, [
                [0.7],  # Normal (30-60)
                [0.2],  # Low (<30)
                [0.1]   # High (>60)
            ])

        cpd_eda = TabularCPD(
            'EDA', 3, [
                [0.7],  # Normal (2-12)
                [0.2],  # Low (<2)
                [0.1]   # High (>12)
            ])

        cpd_perclos = TabularCPD(
            'PERCLOS', 4, [
                [0.4],  # Low (<0.15)
                [0.3],  # Medium (0.15-0.25)
                [0.2],  # High (0.25-0.35)
                [0.1]   # Very High (>0.35)
            ])

        cpd_blink_duration = TabularCPD(
            'BlinkDuration', 4, [
                [0.4],  # Normal (100-300)
                [0.3],  # Slightly Long (300-400)
                [0.2],  # Long (400-500)
                [0.1]   # Very Long (>500)
            ])

        cpd_blink_rate = TabularCPD(
            'BlinkRate', 4, [
                [0.4],  # Normal (8-16)
                [0.3],  # Slightly High (16-20)
                [0.2],  # High (20-25)
                [0.1]   # Very High (>25)
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
            # Updated cardinality to match new states
            evidence_card=[3, 3, 3, 4, 4, 4]
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
        # Calculate the total number of combinations
        cards = [3, 3, 3, 4, 4, 4]  # Cardinality of each node
        num_combinations = np.prod(cards)
        probs = np.zeros((2, num_combinations))

        # Helper function to convert flat index to state indices
        def index_to_states(idx, cards):
            states = []
            for card in reversed(cards):
                states.append(idx % card)
                idx //= card
            return list(reversed(states))

        for i in range(num_combinations):
            states = index_to_states(i, cards)

            # Calculate baseline fatigue probability
            fatigue_contrib = 0.0

            # Heart Rate contribution (3 states)
            hr_weights = [0.1, 0.4, 0.3]  # Normal, Low, High
            fatigue_contrib += hr_weights[states[0]] * weights['heart_rate']

            # HRV contribution (3 states)
            hrv_weights = [0.1, 0.5, 0.2]  # Normal, Low, High
            fatigue_contrib += hrv_weights[states[1]] * weights['hrv']

            # EDA contribution (3 states)
            eda_weights = [0.1, 0.4, 0.3]  # Normal, Low, High
            fatigue_contrib += eda_weights[states[2]] * weights['eda']

            # PERCLOS contribution (4 states)
            perclos_weights = [0.1, 0.3, 0.6, 0.9]  # Low to Very High
            fatigue_contrib += perclos_weights[states[3]] * weights['perclos']

            # Blink Duration contribution (4 states)
            blink_dur_weights = [0.1, 0.3, 0.6, 0.9]  # Normal to Very Long
            fatigue_contrib += blink_dur_weights[states[4]
                                                 ] * weights['blink_duration']

            # Blink Rate contribution (4 states)
            blink_rate_weights = [0.1, 0.3, 0.6, 0.9]  # Normal to Very High
            fatigue_contrib += blink_rate_weights[states[5]
                                                  ] * weights['blink_rate']

            # Normalize and store probabilities
            fatigue_prob = min(1.0, fatigue_contrib)
            probs[0, i] = 1 - fatigue_prob  # Not fatigued
            probs[1, i] = fatigue_prob      # Fatigued

        return probs

    def _get_biometric_states(self, driver_state):
        """Convert biometric measurements to discrete states"""
        states = {}

        # Heart Rate (3 states)
        if driver_state.heart_rate < 60:
            states['HeartRate'] = 1  # Low
        elif driver_state.heart_rate > 100:
            states['HeartRate'] = 2  # High
        else:
            states['HeartRate'] = 0  # Normal

        # HRV (3 states)
        if driver_state.hrv < 30:
            states['HRV'] = 1  # Low
        elif driver_state.hrv > 60:
            states['HRV'] = 2  # High
        else:
            states['HRV'] = 0  # Normal

        # EDA (3 states)
        if driver_state.eda < 2:
            states['EDA'] = 1  # Low
        elif driver_state.eda > 12:
            states['EDA'] = 2  # High
        else:
            states['EDA'] = 0  # Normal

        # PERCLOS (4 states)
        if driver_state.perclos < 0.15:
            states['PERCLOS'] = 0  # Low
        elif driver_state.perclos < 0.25:
            states['PERCLOS'] = 1  # Medium
        elif driver_state.perclos < 0.35:
            states['PERCLOS'] = 2  # High
        else:
            states['PERCLOS'] = 3  # Very High

        # Blink Duration (4 states)
        if driver_state.blink_duration < 300:
            states['BlinkDuration'] = 0  # Normal
        elif driver_state.blink_duration < 400:
            states['BlinkDuration'] = 1  # Slightly Long
        elif driver_state.blink_duration < 500:
            states['BlinkDuration'] = 2  # Long
        else:
            states['BlinkDuration'] = 3  # Very Long

        # Blink Rate (4 states)
        if driver_state.blink_rate < 16:
            states['BlinkRate'] = 0  # Normal
        elif driver_state.blink_rate < 20:
            states['BlinkRate'] = 1  # Slightly High
        elif driver_state.blink_rate < 25:
            states['BlinkRate'] = 2  # High
        else:
            states['BlinkRate'] = 3  # Very High

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

        # Then query for alarm probability using fatigue as evidence
        alarm_evidence = evidence.copy()
        alarm_evidence['Fatigue'] = 1 if fatigue_prob > 0.5 else 0
        alarm_result = self.inference.query(['Alarm'], evidence=alarm_evidence)
        alarm_prob = alarm_result.values[1]    # Probability of alarm

        return fatigue_prob, alarm_prob, alarm_prob > self.ALARM_THRESHOLD
