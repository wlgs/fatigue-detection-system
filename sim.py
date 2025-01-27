import time
import pygame
import threading
from collections import deque
from modules.DriverState import DriverState
from modules.EnvironmentSimulator import EnvironmentSimulator
from modules.DriverSimulator import DriverSimulator
from modules.FatigueEvaluator import FatigueEvaluator
from modules.BiometricFatigueDetector import BiometricFatigueDetector

SIMULATION_DISPLAY_TARGET_FPS = 30
SIMULATION_TICK_RATE = 5000


class RestRecommendationSystem:
    def __init__(self, scenario=None, driver_scenario=None):
        self.driver_state = DriverState(scenario=scenario)
        self.environment_simulator = EnvironmentSimulator()
        self.driver_simulator = DriverSimulator(characteristic=driver_scenario)
        self.fatigue_evaluator = FatigueEvaluator()
        self.biometric_detector = BiometricFatigueDetector()
        self.alarm_probability = 0.0
        self.rest_threshold = 0
        self.valid_alarm_threshold = 20.0
        self.simulation_running = False
        self.deque_length = 200
        self.rest_points_history = deque(
            maxlen=self.deque_length)
        self.alarm_history = deque(maxlen=self.deque_length)
        self.paused = True
        self.simulators_running = False
        self.tick_count = 0
        self.current_alarm_state = False
        self.valid_alarms_count = 0
        self.total_alarms_count = 0
        self.missed_alarms_count = 0
        for _ in range(0, 200):
            self.rest_points_history.append(100.0)
            self.alarm_history.append([0.0, (0, 0, 0)])

    def handle_key_press(self, key):
        if key == pygame.K_w:
            weather_options = list(
                self.environment_simulator.weather_probabilities.keys())
            current_index = weather_options.index(
                self.driver_state.weather_condition)
            next_index = (current_index + 1) % len(weather_options)
            self.driver_state.weather_condition = weather_options[next_index]

        elif key == pygame.K_t:
            traffic_options = list(
                self.environment_simulator.traffic_probabilities.keys())
            current_index = traffic_options.index(
                self.driver_state.traffic_density)
            next_index = (current_index + 1) % len(traffic_options)
            self.driver_state.traffic_density = traffic_options[next_index]

        elif key == pygame.K_d:
            self.driver_state.time_of_day = "night" if self.driver_state.time_of_day == "day" else "day"

        elif key == pygame.K_f:
            self.simulators_running = not self.simulators_running

        elif key == pygame.K_r:
            road_options = list(
                self.environment_simulator.road_probabilities.keys())
            current_index = road_options.index(self.driver_state.road_type)
            next_index = (current_index + 1) % len(road_options)
            self.driver_state.road_type = road_options[next_index]

    def run_tick(self):
        time_since_rest_hours = (
            self.tick_count - self.driver_state.last_rest_tick) * 5 / 60
        fatigue_level = self.fatigue_evaluator.evaluate_fatigue(
            self.driver_state, time_since_rest_hours)
        self.driver_simulator.simulate_physiological(
            self.driver_state, self.driver_state.rest_points)
        if self.simulators_running:
            self.driver_state.weather_condition = self.environment_simulator.simulate_weather(
                self.driver_state.weather_condition)
            self.driver_state.traffic_density = self.environment_simulator.simulate_traffic(
                self.driver_state.traffic_density)
            self.driver_state.road_type = self.environment_simulator.simulate_road_type(
                self.driver_state.road_type)
            time_since_rest_hours = (
                self.tick_count - self.driver_state.last_rest_tick) * 5 / 60
            self.driver_state.time_of_day = self.environment_simulator.simulate_time_of_day(
                self.driver_state.time_of_day, self.tick_count)

        biometric_fatigue, alarm_probability, alarm_triggered = self.biometric_detector.detect_fatigue(
            self.driver_state)
        if alarm_triggered:
            self.total_alarms_count += 1
            if self._is_alarm_valid():
                self.valid_alarms_count += 1
        else:
            if self._is_alarm_valid():
                self.missed_alarms_count += 1

        self.current_alarm_state = alarm_triggered
        self.alarm_probability = alarm_probability

        combined_fatigue = (fatigue_level)
        rest_loss = combined_fatigue * 2.5

        self.driver_state.current_rest_loss = rest_loss
        self.driver_state.rest_points -= rest_loss
        self.rest_points_history.append(self.driver_state.rest_points)
        alarm_color = (0, 255, 0) if self._is_alarm_valid() else (255, 0, 0)
        self.alarm_history.append(
            [1.0, (alarm_color)] if alarm_triggered else [0.0, (alarm_color)])

        if self.driver_state.rest_points < self.rest_threshold:
            self.driver_state.rest_points = 100
            self.driver_state.last_drive_tick_time = self.tick_count - \
                self.driver_state.last_rest_tick
            self.driver_state.last_rest_tick = self.tick_count
            self.driver_simulator._reset_to_normal_state(self.driver_state)
        self.tick_count += 1

    def draw_graph(self, screen, start_x, start_y, width, height):

        pygame.draw.rect(screen, (240, 240, 240),
                         (start_x, start_y, width, height))

        threshold_y = start_y + height - \
            (height * self.valid_alarm_threshold / 100)
        pygame.draw.line(screen, (255, 0, 0), (start_x, threshold_y),
                         (start_x + width, threshold_y), 2)

        total_time_driven = self.tick_count * 5
        time_per_tick = 5
        num_ticks = len(self.rest_points_history)
        point_spacing = width / (num_ticks - 1)

        for i in range(0, num_ticks, 12):
            x_pos = start_x + (i * point_spacing)
            time_label = int(i * time_per_tick)  # Convert to minutes
            label_text = f"{time_label}m" if time_label < 60 else f"{time_label // 60}h"
            y_label_pos = start_y + height + 5

            pygame.draw.line(screen, (200, 200, 200), (x_pos, start_y),
                             (x_pos, start_y + height), 1)
            label_surface = pygame.font.Font(
                None, 24).render(label_text, True, (0, 0, 0))
            screen.blit(label_surface, (x_pos - 10, y_label_pos))

        for i in range(0, 101, 20):
            y_pos = start_y + height - (height * i / 100)
            pygame.draw.line(screen, (200, 200, 200), (start_x, y_pos),
                             (start_x + width, y_pos), 1)

        points = list(self.rest_points_history)
        coords = []
        for i, point in enumerate(points):
            x = start_x + (i * point_spacing)
            y = start_y + height - (height * point / 100)
            coords.append((x, y))

        if len(coords) >= 2:
            pygame.draw.lines(screen, (0, 128, 0), False, coords, 2)

        alarms = list(self.alarm_history)
        for i, alarm in enumerate(alarms):
            alarm_triggered, alarm_color = alarm
            if alarm_triggered > 0:
                x = start_x + (i * point_spacing)
                y = start_y + height - 20
                pygame.draw.circle(screen, alarm_color, (int(x), int(y)), 2)

    def draw_biometric_bars(self, screen, start_x, start_y, width, height):
        """Draw bars showing contribution of each biometric metric to fatigue detection"""
        pygame.draw.rect(screen, (240, 240, 240),
                         (start_x, start_y, width, height))

        fatigue_contributors = self.biometric_detector.getBiometricFatigueContributors()
        metrics = {
            'HeartRate': {
                'value': self.driver_state.heart_rate,
                'weight': fatigue_contributors['HeartRate'],
                'unit': 'BPM',
                'normal': (72, 86),
                'fatigued': (55, 62),
                'color': (0, 150, 255)
            },
            'HRV': {
                'value': self.driver_state.hrv,
                'weight': fatigue_contributors['HRV'],
                'unit': '',
                'normal': (45, 60),
                'fatigued': (15, 25),
                'color': (255, 150, 0)
            },
            'EDA': {
                'value': self.driver_state.eda,
                'weight': fatigue_contributors['EDA'],
                'unit': 'uS',
                'normal': (6, 10),
                'fatigued': (1, 2),
                'color': (150, 255, 0)
            },
            'PERCLOS': {
                'value': self.driver_state.perclos,
                'weight': fatigue_contributors['PERCLOS'],
                'unit': '',
                'normal': (0.05, 0.15),
                'fatigued': (0.35, 0.5),
                'color': (255, 0, 150)
            },
            'BlinkDuration': {
                'value': self.driver_state.blink_duration,
                'weight': fatigue_contributors['BlinkDuration'],
                'unit': 'ms',
                'normal': (50, 250),
                'fatigued': (450, 600),
                'color': (150, 0, 255)
            },
            'BlinkRate': {
                'value': self.driver_state.blink_rate,
                'weight': fatigue_contributors['BlinkRate'],
                'unit': '/min',
                'normal': (8, 14),
                'fatigued': (22, 30),
                'color': (255, 255, 0)
            }
        }

        bar_height = 25
        spacing = 10
        total_height = (bar_height + spacing) * len(metrics)
        start_y = start_y + (height - total_height) // 2

        font = pygame.font.SysFont("monospace", 14)

        for i, (name, metric) in enumerate(metrics.items()):
            y = start_y + i * (bar_height + spacing)

            # Draw metric name
            text = font.render(f"{name}", True, (0, 0, 0))
            metric_value = font.render(
                f"{metric['value']:.2f} {metric['unit']}", True, (0, 0, 0))
            screen.blit(text, (start_x, y))
            screen.blit(metric_value, (start_x + 125, y))

            # Calculate bar position and width
            bar_start_x = start_x + 220
            bar_width = width - 420

            # Draw background bar
            pygame.draw.rect(screen, (220, 220, 220),
                             (bar_start_x, y, bar_width, bar_height))

            # Calculate normalized value between 0 and 1
            if name == 'PERCLOS' or metric['fatigued'][1] > metric['normal'][1]:
                # Metrics that increase with fatigue
                norm_value = (metric['value'] - metric['normal'][0]) / \
                    (metric['fatigued'][1] - metric['normal'][0])
            else:
                # Metrics that decrease with fatigue
                norm_value = 1 - (metric['value'] - metric['fatigued'][0]) / \
                    (metric['normal'][1] - metric['fatigued'][0])

            norm_value = max(0, min(1, norm_value))

            value_width = int(bar_width * norm_value)
            pygame.draw.rect(screen, metric['color'],
                             (bar_start_x, y, value_width, bar_height))

            weight_start_x = bar_start_x + bar_width + 10
            weight_width = 180
            pygame.draw.rect(screen, (220, 220, 220),
                             (weight_start_x, y, weight_width, bar_height))
            # SCALED TIMES 2 FOR VISIBILITY
            weight_value_width = min(
                int(weight_width * metric['weight'] * 2), 180)
            pygame.draw.rect(screen, (0, 0, 255),
                             (weight_start_x, y, weight_value_width, bar_height))

            # value_text = f"{metric['value']:.2f} -> {metric['weight']:.2f}"
            # text = font.render(value_text, True, (0, 0, 0))
            # screen.blit(text, (weight_start_x + weight_width + 10, y))
        text = font.render(
            f"Calculated impact: {self.alarm_probability:.2f}/{self.biometric_detector.ALARM_THRESHOLD}", True, (0, 0, 0))
        screen.blit(text, (start_x+width-230, start_y+height-40))

    def _is_alarm_valid(self):
        if self.driver_state.rest_points <= self.valid_alarm_threshold:
            return True
        return False

    def update_simulation(self):
        while self.simulation_running:
            if not self.paused:
                self.run_tick()
            time.sleep(1/SIMULATION_TICK_RATE)

    def update_single_tick(self):
        if self.paused:
            self.run_tick()

    def toggle_simulation(self):
        self.paused = not self.paused

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((1200, 900))
        pygame.display.set_caption("Driver Rest Recommendation System")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 18, bold=True)

        self.simulation_running = True
        self.simulation_thread = threading.Thread(
            target=self.update_simulation)
        self.simulation_thread.start()

        try:
            while self.simulation_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.simulation_running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            self.toggle_simulation()
                        elif event.key == pygame.K_SPACE:
                            self.update_single_tick()
                        else:
                            self.handle_key_press(event.key)

                screen.fill((255, 255, 255))

                self.draw_graph(screen, 50, 50, 1100, 200)

                self.draw_biometric_bars(screen, 360, 300, 750, 250)

                bar_y = 300
                # pygame.draw.rect(screen, (200, 200, 200), (50, bar_y, 200, 30))
                # pygame.draw.rect(screen, (0, 255, 0), (50, bar_y, self.driver_state.rest_points * 2, 30))

                rest_info_texts = [
                    f"Rest Points: {self.driver_state.rest_points:.1f}",
                    f"Rest Threshold: {self.rest_threshold}",
                    f"Last drive time: {self.driver_state.last_drive_tick_time*5//60}h {self.driver_state.last_drive_tick_time*5%60}m",
                ]

                for i, text in enumerate(rest_info_texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (50, bar_y + i * 30))

                state_text = font.render(
                    f"{'Running' if not self.paused else 'Paused'}", True, (0, 0, 255))
                screen.blit(state_text, (50, bar_y + 100))

                texts = [
                    f"Tick: {self.tick_count}",
                    # f"{self.driver_state.heart_rate:.1f} BPM, {self.driver_state.hrv:.1f} HRV, {self.driver_state.eda:.1f} μS",
                    # f"{self.driver_state.perclos:.2f} PERCLOS, {self.driver_state.blink_duration:.1f} ms, {self.driver_state.blink_rate:.1f} blinks/min",
                    f"[D] Time of Day: {self.driver_state.time_of_day}",
                    f"[W] Weather: {self.driver_state.weather_condition}",
                    f"[T] Traffic: {self.driver_state.traffic_density}",
                    f"[R] Road Type: {self.driver_state.road_type}",
                    f"[F] Simulators: { 'Running' if self.simulators_running else 'Paused'}",
                    # f"Predicted Fatigue: {self.alarm_probability:.2f}/{self.biometric_detector.ALARM_THRESHOLD}",
                    f"FATIGUE ALARM: {'ACTIVE' if self.current_alarm_state else 'OFF'}",
                    f"Valid / Missed / Total: {self.valid_alarms_count} / {self.missed_alarms_count} / {self.total_alarms_count}",
                    f"ACC: {self.valid_alarms_count/((self.total_alarms_count if self.total_alarms_count > 0 else 1) + (self.missed_alarms_count if self.missed_alarms_count > 0 else 1))*100:.2f}%"
                ]

                # Render the texts
                for i, text in enumerate(texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (50, bar_y + 150 + i * 40))

                pygame.display.flip()
                clock.tick(SIMULATION_DISPLAY_TARGET_FPS)

        finally:
            self.simulation_running = False
            if self.simulation_thread:
                self.simulation_thread.join()
            pygame.quit()

    def start(self):
        self.run()

    def stop(self):
        self.simulation_running = False
        if self.simulation_thread:
            self.simulation_thread.join()


if __name__ == "__main__":
    system = RestRecommendationSystem("normal", "normal")
    system.start()
