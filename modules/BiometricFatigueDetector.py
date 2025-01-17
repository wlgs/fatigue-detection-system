class BiometricFatigueDetector:
    """Detects driver fatigue based on biometric measurements"""

    def __init__(self):
        # Thresholds for fatigue detection
        self.THRESHOLD_HR_LOW = 60  # bpm
        self.THRESHOLD_HRV_LOW = 30  # ms
        self.THRESHOLD_EDA_LOW = 3.0  # μS
        self.THRESHOLD_PERCLOS_HIGH = 0.40  # percentage
        self.THRESHOLD_BLINK_DURATION_HIGH = 500  # ms
        self.THRESHOLD_BLINK_RATE_HIGH = 25  # blinks per minute

        # Weights for different measurements
        self.weights = {
            'heart_rate': 0.15,
            'hrv': 0.20,
            'eda': 0.15,
            'perclos': 0.20,
            'blink_duration': 0.15,
            'blink_rate': 0.15
        }

    def _calculate_hr_fatigue(self, hr):
        """Heart rate fatigue score (higher score = more fatigue)"""
        if hr > self.THRESHOLD_HR_LOW:
            return 0.0
        return min(1.0, (self.THRESHOLD_HR_LOW - hr) / 15)

    def _calculate_hrv_fatigue(self, hrv):
        """HRV fatigue score"""
        if hrv > self.THRESHOLD_HRV_LOW:
            return 0.0
        return min(1.0, (self.THRESHOLD_HRV_LOW - hrv) / 15)

    def _calculate_eda_fatigue(self, eda):
        """EDA fatigue score"""
        if eda > self.THRESHOLD_EDA_LOW:
            return 0.0
        return min(1.0, (self.THRESHOLD_EDA_LOW - eda) / 2)

    def _calculate_perclos_fatigue(self, perclos):
        """PERCLOS fatigue score"""
        if perclos < self.THRESHOLD_PERCLOS_HIGH:
            return 0.0
        return min(1.0, (perclos - self.THRESHOLD_PERCLOS_HIGH) / 0.2)

    def _calculate_blink_duration_fatigue(self, duration):
        """Blink duration fatigue score"""
        if duration < self.THRESHOLD_BLINK_DURATION_HIGH:
            return 0.0
        return min(1.0, (duration - self.THRESHOLD_BLINK_DURATION_HIGH) / 300)

    def _calculate_blink_rate_fatigue(self, rate):
        """Blink rate fatigue score"""
        if rate < self.THRESHOLD_BLINK_RATE_HIGH:
            return 0.0
        return min(1.0, (rate - self.THRESHOLD_BLINK_RATE_HIGH) / 10)

    def detect_fatigue(self, driver_state):
        """
        Evaluates driver fatigue based on biometric measurements
        Returns: fatigue_level (0.0-1.0), where >0.7 indicates severe fatigue
        """
        fatigue_scores = {
            'heart_rate': self._calculate_hr_fatigue(driver_state.heart_rate),
            'hrv': self._calculate_hrv_fatigue(driver_state.hrv),
            'eda': self._calculate_eda_fatigue(driver_state.eda),
            'perclos': self._calculate_perclos_fatigue(driver_state.perclos),
            'blink_duration': self._calculate_blink_duration_fatigue(driver_state.blink_duration),
            'blink_rate': self._calculate_blink_rate_fatigue(driver_state.blink_rate)
        }

        # Calculate weighted average fatigue level
        total_fatigue = sum(score * self.weights[metric]
                            for metric, score in fatigue_scores.items())

        return total_fatigue
