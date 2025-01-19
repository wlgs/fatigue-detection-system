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
    def __init__(self, scenario=None):
        self.driver_state = DriverState(scenario=scenario)
        self.environment_simulator = EnvironmentSimulator()
        self.driver_simulator = DriverSimulator()
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
        if key == pygame.K_w:  # Cycle through weather
            weather_options = list(
                self.environment_simulator.weather_probabilities.keys())
            current_index = weather_options.index(
                self.driver_state.weather_condition)
            next_index = (current_index + 1) % len(weather_options)
            self.driver_state.weather_condition = weather_options[next_index]

        elif key == pygame.K_t:  # Cycle through traffic
            traffic_options = list(
                self.environment_simulator.traffic_probabilities.keys())
            current_index = traffic_options.index(
                self.driver_state.traffic_density)
            next_index = (current_index + 1) % len(traffic_options)
            self.driver_state.traffic_density = traffic_options[next_index]

        elif key == pygame.K_d:  # Cycle through day/night
            self.driver_state.time_of_day = "night" if self.driver_state.time_of_day == "day" else "day"

        elif key == pygame.K_f:
            self.simulators_running = not self.simulators_running

        elif key == pygame.K_r:  # Cycle through road types
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

        # Get biometric fatigue assessment
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

        # Combine environmental and biometric fatigue factors
        combined_fatigue = (fatigue_level)
        rest_loss = combined_fatigue * 2.5  # Scale fatigue to rest loss

        self.driver_state.current_rest_loss = rest_loss  # Store current rest loss
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
        x_tick_interval = max(1, int(total_time_driven / 10))
        time_per_tick = 5  # Minutes per tick
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

        # Draw Y-axis grid lines
        for i in range(0, 101, 20):
            y_pos = start_y + height - (height * i / 100)
            pygame.draw.line(screen, (200, 200, 200), (start_x, y_pos),
                             (start_x + width, y_pos), 1)

        # Draw rest points history
        points = list(self.rest_points_history)
        coords = []
        for i, point in enumerate(points):
            x = start_x + (i * point_spacing)
            y = start_y + height - (height * point / 100)
            coords.append((x, y))

        # Draw rest points line
        if len(coords) >= 2:
            pygame.draw.lines(screen, (0, 128, 0), False, coords, 2)

        alarms = list(self.alarm_history)
        for i, alarm in enumerate(alarms):
            alarm_triggered, alarm_color = alarm
            if alarm_triggered > 0:
                x = start_x + (i * point_spacing)
                y = start_y + height - 20  # Position at bottom of graph
                pygame.draw.circle(screen, alarm_color, (int(x), int(y)), 2)

    def _is_alarm_valid(self):
        if self.driver_state.rest_points < self.valid_alarm_threshold:
            return True
        return False

    def update_simulation(self):
        while self.simulation_running:
            if not self.paused:
                self.run_tick()
            time.sleep(1/SIMULATION_TICK_RATE)  # 60 FPS

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
        font = pygame.font.Font(None, 36)

        # Start simulation in a separate thread
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
                        if event.key == pygame.K_RETURN:  # Enter key
                            self.toggle_simulation()
                        elif event.key == pygame.K_SPACE:  # Space key
                            self.update_single_tick()
                        else:
                            # Handle custom key presses
                            self.handle_key_press(event.key)

                screen.fill((255, 255, 255))

                # Draw graph first
                self.draw_graph(screen, 50, 50, 1100, 200)

                # Draw rest points bar and related information
                bar_y = 300
                pygame.draw.rect(screen, (200, 200, 200), (50, bar_y, 200, 30))
                pygame.draw.rect(screen, (0, 255, 0), (50, bar_y,
                                                       self.driver_state.rest_points * 2, 30))

                # Display rest-related information
                rest_info_texts = [
                    f"Rest Points: {self.driver_state.rest_points:.1f}",
                    # f"Current Rest Loss: {self.driver_state.current_rest_loss:.3f}/tick",
                    f"Rest Threshold: {self.rest_threshold}",
                    f"Last drive time: {self.driver_state.last_drive_tick_time*5//60}h {self.driver_state.last_drive_tick_time*5%60}m",
                ]

                for i, text in enumerate(rest_info_texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (270, bar_y + i * 30))

                state_text = font.render(
                    f"{'Running' if not self.paused else 'Paused'}", True, (0, 0, 255))
                screen.blit(state_text, (50, bar_y + 100))

                # Display other information
                texts = [
                    f"Tick: {self.tick_count}",
                    f"{self.driver_state.heart_rate:.1f} BPM, {self.driver_state.hrv:.1f} HRV, {self.driver_state.eda:.1f} μS",
                    f"{self.driver_state.perclos:.2f} PERCLOS, {self.driver_state.blink_duration:.1f} ms, {self.driver_state.blink_rate:.1f} blinks/min",
                    f"[D] Time of Day: {self.driver_state.time_of_day} [W] Weather: {self.driver_state.weather_condition}",
                    f"[T] Traffic: {self.driver_state.traffic_density} [R] Road Type: {self.driver_state.road_type}",
                    f"[F] Simulators: { 'Running' if self.simulators_running else 'Paused'}",
                    f"Predicted Fatigue: {self.alarm_probability:.2f}/{self.biometric_detector.ALARM_THRESHOLD}",
                    f"FATIGUE ALARM: {'ACTIVE' if self.current_alarm_state else 'OFF'}",
                    f"Valid / Missed / Total: {self.valid_alarms_count} / {self.missed_alarms_count} / {self.total_alarms_count}  -  ACC: {self.valid_alarms_count/((self.total_alarms_count if self.total_alarms_count > 0 else 1) + (self.missed_alarms_count if self.missed_alarms_count > 0 else 1))*100:.2f}%",
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
    system = RestRecommendationSystem("worst")
    system.start()
