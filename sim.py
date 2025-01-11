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
    last_rest_time: datetime.datetime = datetime.datetime.now()
    current_speed: float = 0.0
    pulse: float = 70.0
    eyelid_movement: float = 1.0
    temperature: float = 36.6
    weather_condition: str = "clear"
    traffic_density: str = "low"
    time_of_day: str = "day"
    road_type: str = "highway"


class RestRecommendationSystem:
    def __init__(self):
        self.driver_state = DriverState()
        self.setup_bayesian_network()
        self.rest_threshold = 30.0
        self.simulation_running = False
        self.simulation_thread = None
        self.rest_points_history = deque(maxlen=200)  # Store last 200 points
        self.paused = True  # New state to track if simulation is paused

    def setup_bayesian_network(self):
        # [Previous Bayesian Network setup code remains exactly the same]
        self.model = BayesianNetwork([
            ('TimeOfDay', 'Fatigue'),
            ('TimeSinceRest', 'Fatigue'),
            ('Speed', 'Risk'),
            ('Pulse', 'Fatigue'),
            ('EyelidMovement', 'Fatigue'),
            ('Weather', 'Risk'),
            ('Traffic', 'Risk'),
            ('Fatigue', 'Risk'),
            ('RoadType', 'Risk')
        ])

        cpd_time = TabularCPD('TimeOfDay', 2, [[0.7], [0.3]])
        cpd_time_since_rest = TabularCPD('TimeSinceRest', 2, [[0.8], [0.2]])
        cpd_speed = TabularCPD('Speed', 2, [[0.7], [0.3]])
        cpd_pulse = TabularCPD('Pulse', 2, [[0.7], [0.3]])
        cpd_eyelid = TabularCPD('EyelidMovement', 2, [[0.6], [0.4]])
        cpd_weather = TabularCPD('Weather', 2, [[0.8], [0.2]])
        cpd_traffic = TabularCPD('Traffic', 2, [[0.6], [0.4]])
        cpd_road = TabularCPD('RoadType', 2, [[0.7], [0.3]])

        fatigue_probs = np.array([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ])

        cpd_fatigue = TabularCPD(
            'Fatigue', 2,
            fatigue_probs,
            evidence=['TimeOfDay', 'TimeSinceRest', 'Pulse', 'EyelidMovement'],
            evidence_card=[2, 2, 2, 2]
        )

        risk_probs = np.zeros((2, 32))
        for i in range(32):
            risk_probs[0, i] = 0.9 - (i * 0.02)
            risk_probs[1, i] = 1 - risk_probs[0, i]

        cpd_risk = TabularCPD(
            'Risk', 2,
            risk_probs,
            evidence=['Speed', 'Fatigue', 'Weather', 'Traffic', 'RoadType'],
            evidence_card=[2, 2, 2, 2, 2]
        )

        self.model.add_cpds(cpd_time, cpd_time_since_rest, cpd_speed, cpd_pulse, cpd_eyelid,
                            cpd_weather, cpd_traffic, cpd_road, cpd_fatigue, cpd_risk)

        self.inference = VariableElimination(self.model)

    def calculate_rest_points_loss(self):
        evidence = {
            'TimeOfDay': 1 if self.driver_state.time_of_day == "night" else 0,
            'TimeSinceRest': 1 if (datetime.datetime.now() - self.driver_state.last_rest_time).seconds > 7200 else 0,
            'Speed': 1 if self.driver_state.current_speed > 100 else 0,
            'Pulse': 1 if self.driver_state.pulse > 85 else 0,
            'EyelidMovement': 1 if self.driver_state.eyelid_movement < 0.7 else 0,
            'Weather': 1 if self.driver_state.weather_condition == "bad" else 0,
            'Traffic': 1 if self.driver_state.traffic_density == "high" else 0,
            'RoadType': 1 if self.driver_state.road_type == "local" else 0
        }

        result = self.inference.query(['Risk'], evidence=evidence)
        risk_prob = result.values[1]

        base_loss = 0.5
        risk_multiplier = 1 + (risk_prob * 2)
        return base_loss * risk_multiplier

    def draw_graph(self, screen, x, y, width, height):
        """Draw the rest points history graph"""
        # Draw graph background
        pygame.draw.rect(screen, (240, 240, 240), (x, y, width, height))

        # Draw threshold line
        threshold_y = y + height - (height * self.rest_threshold / 100)
        pygame.draw.line(screen, (255, 0, 0), (x, threshold_y),
                         (x + width, threshold_y), 2)

        # Draw rest points history
        if len(self.rest_points_history) > 1:
            points = []
            for i, value in enumerate(self.rest_points_history):
                point_x = x + (i * width / self.rest_points_history.maxlen)
                point_y = y + height - (height * value / 100)
                points.append((point_x, point_y))

            # Draw lines connecting points
            pygame.draw.lines(screen, (0, 128, 0), False, points, 2)

    def simulate_sensor_data(self):
        self.driver_state.pulse += random.uniform(-2, 2)
        self.driver_state.pulse = max(60, min(100, self.driver_state.pulse))

        self.driver_state.eyelid_movement += random.uniform(-0.05, 0.03)
        self.driver_state.eyelid_movement = max(
            0.3, min(1.0, self.driver_state.eyelid_movement))

        self.driver_state.current_speed += random.uniform(-5, 5)
        self.driver_state.current_speed = max(
            0, min(130, self.driver_state.current_speed))

        current_hour = datetime.datetime.now().hour
        self.driver_state.time_of_day = "night" if current_hour < 6 or current_hour > 20 else "day"

    def update_simulation(self):
        while self.simulation_running:
            if not self.paused:  # Only update if not paused
                self.simulate_sensor_data()
            time.sleep(1/60)

    def update_single_tick(self):
        """Update simulation for a single tick when space is pressed"""
        if self.paused:
            self.simulate_sensor_data()
            self.update_display_data()

    def update_display_data(self):
        """Update display data without running simulation"""
        rest_loss = self.calculate_rest_points_loss()
        self.driver_state.rest_points -= rest_loss
        self.rest_points_history.append(self.driver_state.rest_points)

        if self.driver_state.rest_points < self.rest_threshold:
            self.driver_state.rest_points = 100
            self.driver_state.last_rest_time = datetime.datetime.now()

    def toggle_simulation(self):
        """Toggle simulation pause state"""
        self.paused = not self.paused

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Driver Rest Recommendation System")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)

        false_alarms = 0
        total_predictions = 0

        # Start simulation in separate thread
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

                # Only update display data if simulation is running
                if not self.paused:
                    self.update_display_data()

                # Rendering
                screen.fill((255, 255, 255))

                # Draw rest points bar
                pygame.draw.rect(screen, (200, 200, 200), (50, 50, 200, 30))
                pygame.draw.rect(screen, (0, 255, 0), (50, 50,
                                 self.driver_state.rest_points * 2, 30))

                # Draw historical graph
                self.draw_graph(screen, 300, 50, 450, 200)

                # Draw text for threshold and simulation state
                threshold_text = font.render(
                    f"Rest Threshold: {self.rest_threshold}", True, (255, 0, 0))
                screen.blit(threshold_text, (300, 260))

                # Add simulation state text
                state_text = font.render(
                    f"Simulation {'Running' if not self.paused else 'Paused'}", True, (0, 0, 255))
                screen.blit(state_text, (50, 100))

                texts = [
                    f"Rest Points: {self.driver_state.rest_points:.1f}",
                    f"Speed: {self.driver_state.current_speed:.1f} km/h",
                    f"Pulse: {self.driver_state.pulse:.1f}",
                    f"Eyelid Movement: {self.driver_state.eyelid_movement:.2f}",
                    f"Time of Day: {self.driver_state.time_of_day}",
                    f"False Alarms: {false_alarms}",
                    f"Accuracy: {(1 - false_alarms/max(1, total_predictions))*100:.1f}%",
                    "Press ENTER to toggle simulation",
                    "Press SPACE to step forward when paused"
                ]

                for i, text in enumerate(texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (50, 300 + i * 40))

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
