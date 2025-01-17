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

    def simulate_physiological(self, driver_state, current_fatigue_level):
        """
        Updates driver's physiological metrics based on their current fatigue level
        """
        # Check if driver just rested
        if driver_state.last_rest_tick == driver_state.last_drive_tick_time:
            self._reset_to_normal_state(driver_state)
            return driver_state

        # Heart Rate (decreases with fatigue)
        target_hr = 75 - (current_fatigue_level * 20)  # 75 normal -> 55 when fatigued
        driver_state.heart_rate += random.uniform(-self.heart_rate_variance,
                                                self.heart_rate_variance)
        driver_state.heart_rate = max(45, min(100,
                                            driver_state.heart_rate * 0.95 + target_hr * 0.05))

        # Heart Rate Variability (decreases with fatigue)
        target_hrv = 50 - (current_fatigue_level * 25)  # 50 normal -> 25 when fatigued
        driver_state.hrv += random.uniform(-self.hrv_variance, self.hrv_variance)
        driver_state.hrv = max(15, min(70,
                                     driver_state.hrv * 0.95 + target_hrv * 0.05))

        # Electrodermal Activity (decreases with fatigue)
        target_eda = 7 - (current_fatigue_level * 5)  # 7 normal -> 2 when fatigued
        driver_state.eda += random.uniform(-self.eda_variance, self.eda_variance)
        driver_state.eda = max(1, min(12,
                                    driver_state.eda * 0.95 + target_eda * 0.05))

        # PERCLOS (increases with fatigue)
        target_perclos = 0.15 + (current_fatigue_level * 0.25)  # 0.15 normal -> 0.40 when fatigued
        driver_state.perclos += random.uniform(-self.perclos_variance,
                                             self.perclos_variance)
        driver_state.perclos = max(0.1, min(0.5,
                                          driver_state.perclos * 0.95 + target_perclos * 0.05))

        # Blink Duration (increases with fatigue)
        target_blink_duration = 200 + (current_fatigue_level * 300)  # 200 normal -> 500 when fatigued
        driver_state.blink_duration += random.uniform(-self.blink_duration_variance,
                                                    self.blink_duration_variance)
        driver_state.blink_duration = max(100, min(600,
                                                 driver_state.blink_duration * 0.95 + target_blink_duration * 0.05))

        # Blink Rate (increases with fatigue)
        target_blink_rate = 12 + (current_fatigue_level * 15)  # 12 normal -> 27 when fatigued
        driver_state.blink_rate += random.uniform(-self.blink_rate_variance,
                                                self.blink_rate_variance)
        driver_state.blink_rate = max(8, min(30,
                                           driver_state.blink_rate * 0.95 + target_blink_rate * 0.05))

        return driver_state

    def _reset_to_normal_state(self, driver_state):
        """Reset physiological values to normal after rest"""
        driver_state.heart_rate = 75.0
        driver_state.hrv = 50.0
        driver_state.eda = 7.0
        driver_state.perclos = 0.15
        driver_state.blink_duration = 200.0
        driver_state.blink_rate = 12.0