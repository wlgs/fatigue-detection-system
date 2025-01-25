import random


class DriverSimulator:
    """Simulates driver physiological factors based on fatigue level"""

    def __init__(self):
        # Variance settings for randomization
        self.heart_rate_variance = 2.0
        self.hrv_variance = 2.0
        self.eda_variance = 0.2
        self.perclos_variance = 0.02
        self.blink_duration_variance = 20.0
        self.blink_rate_variance = 1.0

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
                'rested': 80, 'fatigued': 55,
                'min': 45, 'max': 100,
                'variance': self.heart_rate_variance
            },
            'hrv': {
                'rested': 50, 'fatigued': 25,
                'min': 15, 'max': 70,
                'variance': self.hrv_variance
            },
            'eda': {
                'rested': 7, 'fatigued': 2,
                'min': 1, 'max': 12,
                'variance': self.eda_variance
            },
            'perclos': {
                'rested': 0.15, 'fatigued': 0.40,
                'min': 0.1, 'max': 0.5,
                'variance': self.perclos_variance
            },
            'blink_duration': {
                'rested': 200, 'fatigued': 500,
                'min': 100, 'max': 600,
                'variance': self.blink_duration_variance
            },
            'blink_rate': {
                'rested': 12, 'fatigued': 27,
                'min': 8, 'max': 30,
                'variance': self.blink_rate_variance
            }
        }

        # Update each metric
        for metric, values in metrics.items():
            # Calculate target value based on current fatigue
            if metric in ['perclos', 'blink_duration', 'blink_rate']:
                # These metrics increase with fatigue
                target = values['rested'] + \
                    (values['fatigued'] - values['rested']) * fatigue_factor
            else:
                # These metrics decrease with fatigue
                target = values['rested'] - \
                    (values['rested'] - values['fatigued']) * fatigue_factor

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
        driver_state.heart_rate = 75.0
        driver_state.hrv = 50.0
        driver_state.eda = 6.0
        driver_state.perclos = 0.10
        driver_state.blink_duration = 200.0
        driver_state.blink_rate = 12.0
