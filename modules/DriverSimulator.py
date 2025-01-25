import random
from typing import Literal

DriverCharacteristic = Literal[
    "normal",
    "low_heart_rate",
    "fast_blinker",
    "eyes_wide_open",
    "sweaty_palms",
    "drowsy"
]


class DriverSimulator:
    """Simulates driver physiological factors based on fatigue level"""

    def __init__(self, characteristic: DriverCharacteristic = "normal"):
        # Variance settings for randomization
        self.heart_rate_variance = 2.0
        self.hrv_variance = 2.0
        self.eda_variance = 0.2
        self.perclos_variance = 0.02
        self.blink_duration_variance = 20.0
        self.blink_rate_variance = 1.0

        # Define characteristic biases for different driver types
        self.characteristic_biases = {
            "normal": {
                "heart_rate": {"base_offset": 0, "fatigue_resistance": 1.0},
                "hrv": {"base_offset": 0, "fatigue_resistance": 1.0},
                "eda": {"base_offset": 0, "fatigue_resistance": 1.0},
                "perclos": {"base_offset": 0, "fatigue_resistance": 1.0},
                "blink_duration": {"base_offset": 0, "fatigue_resistance": 1.0},
                "blink_rate": {"base_offset": 0, "fatigue_resistance": 1.0}
            },
            "low_heart_rate": {
                "heart_rate": {"base_offset": -30, "fatigue_resistance": 10},
                "hrv": {"base_offset": 5, "fatigue_resistance": 1.2},
                "eda": {"base_offset": 0, "fatigue_resistance": 1.0},
                "perclos": {"base_offset": 0, "fatigue_resistance": 1.0},
                "blink_duration": {"base_offset": 0, "fatigue_resistance": 1.0},
                "blink_rate": {"base_offset": 0, "fatigue_resistance": 1.0}
            },
            "fast_blinker": {
                "heart_rate": {"base_offset": 0, "fatigue_resistance": 1.0},
                "hrv": {"base_offset": 0, "fatigue_resistance": 1.0},
                "eda": {"base_offset": 0, "fatigue_resistance": 1.0},
                "perclos": {"base_offset": 0.05, "fatigue_resistance": 1.1},
                "blink_duration": {"base_offset": -50, "fatigue_resistance": 1.0},
                "blink_rate": {"base_offset": 5, "fatigue_resistance": 0.9}
            },
            "eyes_wide_open": {
                "heart_rate": {"base_offset": 0, "fatigue_resistance": 1.0},
                "hrv": {"base_offset": 0, "fatigue_resistance": 1.0},
                "eda": {"base_offset": 0, "fatigue_resistance": 1.0},
                "perclos": {"base_offset": -0.05, "fatigue_resistance": 3},
                "blink_duration": {"base_offset": -30, "fatigue_resistance": 3},
                "blink_rate": {"base_offset": -3, "fatigue_resistance": 3}
            },
            "sweaty_palms": {
                "heart_rate": {"base_offset": 5, "fatigue_resistance": 0.9},
                "hrv": {"base_offset": -5, "fatigue_resistance": 0.9},
                "eda": {"base_offset": 2, "fatigue_resistance": 0.8},
                "perclos": {"base_offset": 0, "fatigue_resistance": 1.0},
                "blink_duration": {"base_offset": 0, "fatigue_resistance": 1.0},
                "blink_rate": {"base_offset": 0, "fatigue_resistance": 1.0}
            },
            "drowsy": {
                "heart_rate": {"base_offset": -5, "fatigue_resistance": 0.8},
                "hrv": {"base_offset": -5, "fatigue_resistance": 0.8},
                "eda": {"base_offset": -1, "fatigue_resistance": 0.9},
                "perclos": {"base_offset": 0.1, "fatigue_resistance": 0.7},
                "blink_duration": {"base_offset": 50, "fatigue_resistance": 0.7},
                "blink_rate": {"base_offset": 3, "fatigue_resistance": 0.8}
            }
        }

        self.characteristic = characteristic
        self.biases = self.characteristic_biases[characteristic]

    def _apply_fatigue_effect(self, base_value, fatigue_level, min_val, max_val, inverse=False):
        """
        Adjusts a physiological value based on fatigue level
        inverse=True means the value decreases with fatigue (like HR and EDA)
        """
        if inverse:
            fatigue_effect = (1 - fatigue_level)
        else:
            fatigue_effect = fatigue_level

        range_size = max_val - min_val
        target_value = min_val + (range_size * fatigue_effect)
        return target_value

    def _update_metric(self, current_value, target_value, variance, min_val, max_val):
        """Helper method to update a physiological metric"""
        current_value += random.uniform(-variance, variance)
        return max(min_val, min(max_val, current_value * 0.95 + target_value * 0.05))

    def simulate_physiological(self, driver_state, energy_level):
        """
        Updates driver's physiological metrics based on energy level (100 = fully rested, 0 = maximum fatigue)
        """
        # Convert energy level to fatigue factor (0-1 scale)
        fatigue_bias = 25  # Bias to make fatigue more impactful
        fatigue_factor = (100 - fatigue_bias - energy_level) / 100.0

        # Define normal (rested) and fatigued values for each metric
        metrics = {
            'heart_rate': {
                'rested': 80 + self.biases['heart_rate']['base_offset'],
                'fatigued': 55 + self.biases['heart_rate']['base_offset'],
                'min': 45, 'max': 100,
                'variance': self.heart_rate_variance,
                'fatigue_resistance': self.biases['heart_rate']['fatigue_resistance']
            },
            'hrv': {
                'rested': 50 + self.biases['hrv']['base_offset'],
                'fatigued': 25 + self.biases['hrv']['base_offset'],
                'min': 15, 'max': 70,
                'variance': self.hrv_variance,
                'fatigue_resistance': self.biases['hrv']['fatigue_resistance']
            },
            'eda': {
                'rested': 7 + self.biases['eda']['base_offset'],
                'fatigued': 2 + self.biases['eda']['base_offset'],
                'min': 1, 'max': 12,
                'variance': self.eda_variance,
                'fatigue_resistance': self.biases['eda']['fatigue_resistance']
            },
            'perclos': {
                'rested': 0.15 + self.biases['perclos']['base_offset'],
                'fatigued': 0.40 + self.biases['perclos']['base_offset'],
                'min': 0.1, 'max': 0.5,
                'variance': self.perclos_variance,
                'fatigue_resistance': self.biases['perclos']['fatigue_resistance']
            },
            'blink_duration': {
                'rested': 200 + self.biases['blink_duration']['base_offset'],
                'fatigued': 500 + self.biases['blink_duration']['base_offset'],
                'min': 100, 'max': 600,
                'variance': self.blink_duration_variance,
                'fatigue_resistance': self.biases['blink_duration']['fatigue_resistance']
            },
            'blink_rate': {
                'rested': 12 + self.biases['blink_rate']['base_offset'],
                'fatigued': 27 + self.biases['blink_rate']['base_offset'],
                'min': 8, 'max': 30,
                'variance': self.blink_rate_variance,
                'fatigue_resistance': self.biases['blink_rate']['fatigue_resistance']
            }
        }

        # Update each metric
        for metric, values in metrics.items():
            # Calculate target value based on current fatigue, adjusted by fatigue resistance
            adjusted_fatigue = fatigue_factor / values['fatigue_resistance']

            if metric in ['perclos', 'blink_duration', 'blink_rate']:
                # These metrics increase with fatigue
                target = values['rested'] + \
                    (values['fatigued'] - values['rested']) * adjusted_fatigue
            else:
                # These metrics decrease with fatigue
                target = values['rested'] - \
                    (values['rested'] - values['fatigued']) * adjusted_fatigue

            # Update the metric
            current_value = getattr(driver_state, metric)
            new_value = self._update_metric(
                current_value, target,
                values['variance'],
                values['min'], values['max']
            )
            setattr(driver_state, metric, new_value)

        return driver_state

    def _reset_to_normal_state(self, driver_state):
        """Reset physiological values to normal after rest"""
        driver_state.heart_rate = 75.0 + \
            self.biases['heart_rate']['base_offset']
        driver_state.hrv = 50.0 + self.biases['hrv']['base_offset']
        driver_state.eda = 6.0 + self.biases['eda']['base_offset']
        driver_state.perclos = 0.10 + self.biases['perclos']['base_offset']
        driver_state.blink_duration = 200.0 + \
            self.biases['blink_duration']['base_offset']
        driver_state.blink_rate = 12.0 + \
            self.biases['blink_rate']['base_offset']
