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
        # Heart Rate (decreases with fatigue)
        target_hr = self._apply_fatigue_effect(driver_state.heart_rate,
                                             current_fatigue_level,
                                             55, 90,
                                             inverse=True)
        driver_state.heart_rate += random.uniform(-self.heart_rate_variance,
                                                self.heart_rate_variance)
        driver_state.heart_rate = max(45, min(100,
                                            driver_state.heart_rate * 0.95 + target_hr * 0.05))

        # Heart Rate Variability (decreases with fatigue)
        target_hrv = self._apply_fatigue_effect(driver_state.hrv,
                                              current_fatigue_level,
                                              20, 60,
                                              inverse=True)
        driver_state.hrv += random.uniform(-self.hrv_variance, self.hrv_variance)
        driver_state.hrv = max(15, min(70,
                                     driver_state.hrv * 0.95 + target_hrv * 0.05))

        # Electrodermal Activity (decreases with fatigue)
        target_eda = self._apply_fatigue_effect(driver_state.eda,
                                              current_fatigue_level,
                                              2, 8,
                                              inverse=True)
        driver_state.eda += random.uniform(-self.eda_variance, self.eda_variance)
        driver_state.eda = max(1, min(10,
                                    driver_state.eda * 0.95 + target_eda * 0.05))

        # PERCLOS (increases with fatigue)
        target_perclos = self._apply_fatigue_effect(driver_state.perclos,
                                                  current_fatigue_level,
                                                  0.15, 0.60)
        driver_state.perclos += random.uniform(-self.perclos_variance,
                                             self.perclos_variance)
        driver_state.perclos = max(0.1, min(0.8,
                                          driver_state.perclos * 0.95 + target_perclos * 0.05))

        # Blink Duration (increases with fatigue)
        target_blink_duration = self._apply_fatigue_effect(driver_state.blink_duration,
                                                         current_fatigue_level,
                                                         200, 600)
        driver_state.blink_duration += random.uniform(-self.blink_duration_variance,
                                                    self.blink_duration_variance)
        driver_state.blink_duration = max(100, min(800,
                                                 driver_state.blink_duration * 0.95 + target_blink_duration * 0.05))

        # Blink Rate (increases with fatigue)
        target_blink_rate = self._apply_fatigue_effect(driver_state.blink_rate,
                                                     current_fatigue_level,
                                                     15, 30)
        driver_state.blink_rate += random.uniform(-self.blink_rate_variance,
                                                self.blink_rate_variance)
        driver_state.blink_rate = max(8, min(35,
                                           driver_state.blink_rate * 0.95 + target_blink_rate * 0.05))

        return driver_state