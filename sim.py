import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import datetime
import random
import time
import pygame
import threading
from dataclasses import dataclass
from collections import deque


@dataclass
class DriverState:
    rest_points: float = 100.0
    last_rest_tick: int = 0
    current_speed: float = 0.0
    pulse: float = 70.0
    eyelid_movement: float = 1.0
    temperature: float = 36.6
    weather_condition: str = "clear"
    traffic_density: str = "low"
    time_of_day: str = "day"
    road_type: str = "highway"
    current_rest_loss: float = 0.0  # Added to track current rest loss


class EnvironmentSimulator:
    """Handles simple probability-based simulation of environmental factors"""

    def __init__(self):
        self.weather_change_prob = 1/36
        self.weather_probabilities = {
            "clear": 0.5,
            "rain": 0.1,
            "fog": 0.1,
            "snow": 0.1,
            "bad": 0.2,
        }
        self.traffic_change_prob = 1/12
        self.traffic_probabilities = {
            "low": 0.4,
            "medium": 0.4,
            "high": 0.2
        }

    def simulate_weather(self, current_weather):
        if random.random() < self.weather_change_prob:
            return random.choices(
                list(self.weather_probabilities.keys()),
                list(self.weather_probabilities.values())
            )[0]
        return current_weather

    def simulate_traffic(self, current_traffic):
        if random.random() < self.traffic_change_prob:
            return random.choices(
                list(self.traffic_probabilities.keys()),
                list(self.traffic_probabilities.values())
            )[0]
        return current_traffic


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


class FatigueEvaluator:
    """Uses Bayesian Network for fatigue evaluation"""

    def __init__(self):
        self.setup_bayesian_network()

    def setup_bayesian_network(self):
        # Simplified network focused only on fatigue
        self.model = BayesianNetwork([
            ('TimeOfDay', 'Fatigue'),
            ('TimeSinceRest', 'Fatigue'),
            ('Pulse', 'Fatigue'),
            ('EyelidMovement', 'Fatigue'),
            ('Weather', 'Fatigue'),
            ('Traffic', 'Fatigue')
        ])

        # Basic CPDs
        cpd_time = TabularCPD('TimeOfDay', 2, [[0.7], [0.3]])
        cpd_time_since_rest = TabularCPD('TimeSinceRest', 2, [[0.8], [0.2]])
        cpd_pulse = TabularCPD('Pulse', 2, [[0.7], [0.3]])
        cpd_eyelid = TabularCPD('EyelidMovement', 2, [[0.6], [0.4]])

        # Weather impacts fatigue (5 states)
        cpd_weather = TabularCPD('Weather', 5, [
            [0.5],  # Clear
            [0.2],  # Rain
            [0.15],  # Fog
            [0.1],  # Snow
            [0.05]  # Bad
        ])

        # Traffic impacts fatigue (3 states)
        cpd_traffic = TabularCPD('Traffic', 3, [
            [0.4],  # Low
            [0.4],  # Medium
            [0.2]  # High
        ])

        fatigue_probs = np.zeros((2, 240))

        for i in range(240):
            base_prob = 0.05

            time_of_day_factor = 0.1 if (i // 120) else 0
            time_since_rest_factor = 0.25 if ((i // 60) % 2) else 0
            pulse_factor = 0.1 if ((i // 30) % 2) else 0
            eyelid_factor = 0.15 if ((i // 15) % 2) else 0
            weather_factor = [0, 0.1, 0.2, 0.2, 0.4][(i // 3) % 5]
            traffic_factor = [0, 0.1, 0.3][i % 3]

            # Combine factors
            fatigue_prob = min(1, base_prob + time_of_day_factor + time_since_rest_factor +
                               pulse_factor + eyelid_factor + weather_factor + traffic_factor)

            fatigue_probs[0, i] = 1 - fatigue_prob
            fatigue_probs[1, i] = fatigue_prob

        cpd_fatigue = TabularCPD(
            'Fatigue', 2,
            fatigue_probs,
            evidence=['TimeOfDay', 'TimeSinceRest', 'Pulse',
                      'EyelidMovement', 'Weather', 'Traffic'],
            evidence_card=[2, 2, 2, 2, 5, 3]
        )

        self.model.add_cpds(cpd_time, cpd_time_since_rest, cpd_pulse,
                            cpd_eyelid, cpd_weather, cpd_traffic, cpd_fatigue)
        self.inference = VariableElimination(self.model)

    def evaluate_fatigue(self, driver_state, time_since_rest_hours):
        evidence = {
            'TimeOfDay': 1 if driver_state.time_of_day == "night" else 0,
            'TimeSinceRest': 1 if time_since_rest_hours > 2 else 0,
            'Pulse': 1 if driver_state.pulse > 85 else 0,
            'EyelidMovement': 1 if driver_state.eyelid_movement < 0.6 else 0,
            'Weather': {
                'clear': 0,
                'rain': 1,
                'fog': 2,
                'snow': 3,
                'bad': 4
            }[driver_state.weather_condition],
            'Traffic': {
                'low': 0,
                'medium': 1,
                'high': 2
            }[driver_state.traffic_density]
        }

        result = self.inference.query(['Fatigue'], evidence=evidence)
        return result.values[1]  # Probability of high fatigue


class RestRecommendationSystem:
    def __init__(self):
        self.driver_state = DriverState()
        self.environment_simulator = EnvironmentSimulator()
        self.driver_simulator = DriverSimulator()
        self.fatigue_evaluator = FatigueEvaluator()

        self.rest_threshold = 30.0
        self.simulation_running = False
        self.rest_points_history = deque(maxlen=200)  # Store last 200 points
        self.paused = True
        self.tick_count = 0

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

        elif key == pygame.K_r:  # Cycle through road types
            road_options = ["highway", "city", "rural"]
            current_index = road_options.index(self.driver_state.road_type)
            next_index = (current_index + 1) % len(road_options)
            self.driver_state.road_type = road_options[next_index]

    def run_tick(self):
        self.driver_state.weather_condition = self.environment_simulator.simulate_weather(
            self.driver_state.weather_condition)
        self.driver_state.traffic_density = self.environment_simulator.simulate_traffic(
            self.driver_state.traffic_density)

        # Update driver state
        self.driver_state = self.driver_simulator.simulate_physiological(
            self.driver_state)

        # Update time of day
        current_hour = (self.tick_count * 5 // 60) % 24
        self.driver_state.time_of_day = "night" if current_hour < 6 or current_hour > 20 else "day"

        # Evaluate fatigue
        time_since_rest_hours = (
            self.tick_count - self.driver_state.last_rest_tick) * 5 / 60
        fatigue_level = self.fatigue_evaluator.evaluate_fatigue(
            self.driver_state, time_since_rest_hours)

        rest_loss = fatigue_level * 2  # Scale fatigue to rest loss
        self.driver_state.current_rest_loss = rest_loss  # Store current rest loss
        self.driver_state.rest_points -= rest_loss
        self.rest_points_history.append(self.driver_state.rest_points)

        if self.driver_state.rest_points < self.rest_threshold:
            self.driver_state.rest_points = 100
            self.driver_state.pulse = 70.0
            self.driver_state.eyelid_movement = 1.0
            self.driver_state.current_speed = 0.0
            self.driver_state.last_rest_tick = self.tick_count

        self.tick_count += 1

    def draw_graph(self, screen, start_x, start_y, width, height):
        if len(self.rest_points_history) < 2:
            return

        # Draw graph background
        pygame.draw.rect(screen, (240, 240, 240),
                         (start_x, start_y, width, height))

        # Draw threshold line
        threshold_y = start_y + height - (height * self.rest_threshold / 100)
        pygame.draw.line(screen, (255, 0, 0), (start_x, threshold_y),
                         (start_x + width, threshold_y), 2)

        # Draw grid lines
        for i in range(0, 101, 20):
            y_pos = start_y + height - (height * i / 100)
            pygame.draw.line(screen, (200, 200, 200), (start_x, y_pos),
                             (start_x + width, y_pos), 1)

        # Draw rest points history
        points = list(self.rest_points_history)
        point_spacing = width / (len(points) - 1)

        coords = []
        for i, point in enumerate(points):
            x = start_x + (i * point_spacing)
            y = start_y + height - (height * point / 100)
            coords.append((x, y))

        # Draw line graph
        if len(coords) >= 2:
            pygame.draw.lines(screen, (0, 128, 0), False, coords, 2)

    def update_simulation(self):
        while self.simulation_running:
            if not self.paused:
                self.run_tick()
            time.sleep(1/60)  # 60 FPS

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
                    f"Current Rest Loss: {self.driver_state.current_rest_loss:.3f}/tick",
                    f"Rest Threshold: {self.rest_threshold}",
                ]

                for i, text in enumerate(rest_info_texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (270, bar_y + i * 30))

                state_text = font.render(
                    f"Simulation {'Running' if not self.paused else 'Paused'}", True, (0, 0, 255))
                screen.blit(state_text, (50, bar_y + 100))

                # Display other information
                texts = [
                    f"Tick: {self.tick_count}",
                    f"Speed: {self.driver_state.current_speed:.1f} km/h",
                    f"Pulse: {self.driver_state.pulse:.1f}",
                    f"Eyelid Movement: {self.driver_state.eyelid_movement:.2f}",
                    f"Time of Day: {self.driver_state.time_of_day}",
                    f"Weather: {self.driver_state.weather_condition}",
                    f"Traffic: {self.driver_state.traffic_density}",
                    f"Road Type: {self.driver_state.road_type}",
                ]

                # Render the texts
                for i, text in enumerate(texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (50, bar_y + 150 + i * 40))

                pygame.display.flip()
                clock.tick(60)

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
    system = RestRecommendationSystem()
    system.start()
