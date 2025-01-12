import random


class DriverSimulator:
    """Handles simulation of driver physiological factors"""

    def __init__(self):
        self.pulse_variance = 2.0
        self.eyelid_variance = 0.05
        self.speed_variance = 5.0

    def simulate_physiological(self, driver_state):
        # Simulate pulse changes
        driver_state.pulse += random.uniform(-self.pulse_variance,
                                             self.pulse_variance)
        driver_state.pulse = max(60, min(100, driver_state.pulse))

        # Simulate eyelid movement changes
        driver_state.eyelid_movement += random.uniform(-self.eyelid_variance,
                                                       self.eyelid_variance * 0.6)
        driver_state.eyelid_movement = max(
            0.3, min(1.0, driver_state.eyelid_movement))

        if driver_state.current_speed < 50:
            driver_state.current_speed += random.uniform(
                0, self.speed_variance)
        else:
            driver_state.current_speed += random.uniform(-self.speed_variance,
                                                         self.speed_variance)
        driver_state.current_speed = max(
            0, min(130, driver_state.current_speed))

        return driver_state
