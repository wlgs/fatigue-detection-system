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


@dataclass
class DriverState:
    rest_points: float = 100.0
    last_rest_time: datetime.datetime = datetime.datetime.now()
    current_speed: float = 0.0
    pulse: float = 70.0
    eyelid_movement: float = 1.0  # 1.0 is normal, lower values indicate tiredness
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

    def setup_bayesian_network(self):
        # Define the structure of the Bayesian Network
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

        # Define CPDs (Conditional Probability Distributions)
        # Note: These probabilities are simplified for demonstration
        cpd_time = TabularCPD('TimeOfDay', 2, [[0.7], [0.3]])  # Day/Night

        cpd_time_since_rest = TabularCPD(
            'TimeSinceRest', 2, [[0.8], [0.2]])  # Short/Long

        # Added Speed CPD (Normal/High)
        cpd_speed = TabularCPD('Speed', 2, [[0.7], [0.3]])

        cpd_pulse = TabularCPD('Pulse', 2, [[0.7], [0.3]])  # Normal/High

        cpd_eyelid = TabularCPD('EyelidMovement', 2, [
                                [0.6], [0.4]])  # Normal/Slow

        cpd_weather = TabularCPD('Weather', 2, [[0.8], [0.2]])  # Clear/Bad

        cpd_traffic = TabularCPD('Traffic', 2, [[0.6], [0.4]])  # Light/Heavy

        cpd_road = TabularCPD('RoadType', 2, [[0.7], [0.3]])  # Highway/Local

        # Fixed CPD for fatigue - now with correct dimensions (2^4 = 16 combinations)
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

        # Risk CPD (2^4 = 16 combinations for evidence variables)
        risk_probs = np.array([
            [0.95, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        ])

        cpd_risk = TabularCPD(
            'Risk', 2,
            risk_probs,
            evidence=['Fatigue', 'Weather', 'Traffic', 'RoadType'],
            evidence_card=[2, 2, 2, 2]
        )

        # Add CPDs to the model
        self.model.add_cpds(cpd_time, cpd_time_since_rest, cpd_speed, cpd_pulse, cpd_eyelid,
                            cpd_weather, cpd_traffic, cpd_road, cpd_fatigue, cpd_risk)

        # Initialize inference
        self.inference = VariableElimination(self.model)

    def calculate_rest_points_loss(self):
        # Convert current state to evidence
        evidence = {
            'TimeOfDay': 1 if self.driver_state.time_of_day == "night" else 0,
            'TimeSinceRest': 1 if (datetime.datetime.now() - self.driver_state.last_rest_time).seconds > 7200 else 0,
            'Speed': 1 if self.driver_state.current_speed > 100 else 0,  # Added Speed evidence
            'Pulse': 1 if self.driver_state.pulse > 85 else 0,
            'EyelidMovement': 1 if self.driver_state.eyelid_movement < 0.7 else 0,
            'Weather': 1 if self.driver_state.weather_condition == "bad" else 0,
            'Traffic': 1 if self.driver_state.traffic_density == "high" else 0,
            'RoadType': 1 if self.driver_state.road_type == "local" else 0
        }

        # Query the Bayesian network
        result = self.inference.query(['Risk'], evidence=evidence)
        risk_prob = result.values[1]  # Probability of high risk

        # Calculate rest points loss based on risk
        base_loss = 0.5  # Base loss per tick
        risk_multiplier = 1 + (risk_prob * 2)  # Higher risk means faster loss
        return base_loss * risk_multiplier

    def simulate_sensor_data(self):
        # Simulate changes in driver state
        self.driver_state.pulse += random.uniform(-2, 2)
        self.driver_state.pulse = max(60, min(100, self.driver_state.pulse))

        self.driver_state.eyelid_movement += random.uniform(-0.05, 0.03)
        self.driver_state.eyelid_movement = max(
            0.3, min(1.0, self.driver_state.eyelid_movement))

        # Simulate speed changes
        self.driver_state.current_speed += random.uniform(-5, 5)
        self.driver_state.current_speed = max(
            0, min(130, self.driver_state.current_speed))

        # Simulate time of day changes
        current_hour = datetime.datetime.now().hour
        self.driver_state.time_of_day = "night" if current_hour < 6 or current_hour > 20 else "day"

    def run_simulation(self):
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Driver Rest Recommendation System")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)

        self.simulation_running = True
        false_alarms = 0
        total_predictions = 0

        while self.simulation_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.simulation_running = False

            # Update simulation
            self.simulate_sensor_data()
            rest_loss = self.calculate_rest_points_loss()
            self.driver_state.rest_points -= rest_loss

            # Check if rest is needed
            rest_needed = self.driver_state.rest_points < self.rest_threshold

            # Simulate rest if needed
            if rest_needed:
                self.driver_state.rest_points = 100
                self.driver_state.last_rest_time = datetime.datetime.now()
                false_alarms += random.random() < 0.1  # 10% chance of false alarm
                total_predictions += 1

            # Render visualization
            screen.fill((255, 255, 255))

            # Draw rest points bar
            pygame.draw.rect(screen, (200, 200, 200), (50, 50, 200, 30))
            pygame.draw.rect(screen, (0, 255, 0),
                             (50, 50, self.driver_state.rest_points * 2, 30))

            # Draw text information
            texts = [
                f"Rest Points: {self.driver_state.rest_points:.1f}",
                f"Speed: {self.driver_state.current_speed:.1f} km/h",
                f"Pulse: {self.driver_state.pulse:.1f}",
                f"Eyelid Movement: {self.driver_state.eyelid_movement:.2f}",
                f"Time of Day: {self.driver_state.time_of_day}",
                f"False Alarms: {false_alarms}",
                f"Accuracy: {(1 - false_alarms/max(1, total_predictions))*100:.1f}%"
            ]

            for i, text in enumerate(texts):
                surface = font.render(text, True, (0, 0, 0))
                screen.blit(surface, (50, 100 + i * 40))

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        pygame.quit()

    def start(self):
        simulation_thread = threading.Thread(target=self.run_simulation)
        simulation_thread.start()

    def stop(self):
        self.simulation_running = False


if __name__ == "__main__":
    system = RestRecommendationSystem()
    system.start()
