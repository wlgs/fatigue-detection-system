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


# WEATHER_CHANGE_PROB = 1 / 288  # 1 change per day on average
WEATHER_CHANGE_PROB = 1 / 36  # 1 change per day on average
SIMULATION_SPEED_TICKS_PER_SECOND = 60


@dataclass
class DriverState:
    rest_points: float = 100.0
    last_rest_tick: int = 0  # Track the tick when the last rest occurred
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
        self.tick_count = 0  # Simulation tick counter
        self.display_data = {
            'Rest Loss': 0,
            'Base Rest Loss': 0,
            'Risk Probability': 0
        }  # Display data dictionary

    def setup_bayesian_network(self):
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
        cpd_traffic = TabularCPD('Traffic', 2, [[0.6], [0.4]])
        cpd_road = TabularCPD('RoadType', 2, [[0.7], [0.3]])

        # Updated Weather CPD with five states
        weather_probs = np.array([
            [0.7],  # Clear (most likely)
            [0.1],  # Rain
            [0.1],  # Fog
            [0.05],  # Snow
            [0.05]  # Bad weather
        ])
        cpd_weather = TabularCPD('Weather', 5, weather_probs)

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

        risk_probs = np.zeros((2, 80))

        for i in range(80):
            base_risk = 0.7 - (i * 0.01)  # Slightly lower base risk

            risk_factor = base_risk
            risk_probs[0, i] = max(0, min(1, risk_factor))  # Lower risk state
            risk_probs[1, i] = 1 - risk_probs[0, i]  # Higher risk state

        print(risk_probs)

        # Add the updated CPDs to the model
        cpd_risk = TabularCPD(
            'Risk', 2,
            risk_probs,
            evidence=['Speed', 'Fatigue', 'Weather', 'Traffic', 'RoadType'],
            # Correct evidence_card for Weather (5 states)
            evidence_card=[2, 2, 5, 2, 2]
        )

        self.model.add_cpds(cpd_time, cpd_time_since_rest, cpd_speed, cpd_pulse, cpd_eyelid,
                            cpd_weather, cpd_traffic, cpd_road, cpd_fatigue, cpd_risk)

        self.inference = VariableElimination(self.model)

    def calculate_time_since_rest(self):
        ticks_since_rest = self.tick_count - self.driver_state.last_rest_tick
        total_minutes = ticks_since_rest * 5  # Each tick = 5 minutes
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m"

    def calculate_rest_points_loss(self):
        evidence = {
            'TimeOfDay': 1,  # Whether it's day or night, we assume it contributes positively
            # Assume that being rested for more than 2 hours always contributes to fatigue/risk
            'TimeSinceRest': 1,
            # Speed always contributes positively (e.g., speeding increases risk/fatigue)
            'Speed': 1,
            'Pulse': 1,  # High pulse rate always contributes to fatigue/risk
            # Reduced eyelid movement (tiredness) always contributes to fatigue/risk
            'EyelidMovement': 1,
            'Weather': 1,  # Assume bad weather conditions always contribute to risk/fatigue
            'Traffic': 1,  # Assume high traffic always contributes to risk/fatigue
            'RoadType': 1  # Assume local roads always contribute to risk/fatigue
        }

        # Query the Bayesian network
        result = self.inference.query(['Risk'], evidence=evidence)
        risk_prob = result.values[1]

        # Calculate rest points loss based on risk
        base_loss = 0.5
        risk_multiplier = 1 + (risk_prob * 2)
        return base_loss * risk_multiplier

    def draw_graph(self, screen, x, y, width, height):
        """Draw the rest points history graph with weather indication bars"""

        graph_background_color = (135, 206, 235)  # Light sky blue for day

        # Draw the background of the graph with the time-of-day color
        pygame.draw.rect(screen, graph_background_color, (x, y, width, height))

        # Draw the threshold line for rest points
        threshold_y = y + height - (height * self.rest_threshold / 100)
        pygame.draw.line(screen, (255, 0, 0), (x, threshold_y),
                         (x + width, threshold_y), 2)

        # Draw the line graph for the rest points history
        if len(self.rest_points_history) > 1:
            points = []
            for i, value in enumerate(self.rest_points_history):
                point_x = x + (i * width / self.rest_points_history.maxlen)
                point_y = y + height - (height * value / 100)
                points.append((point_x, point_y))

            # Draw lines connecting points (rest points over time)
            pygame.draw.lines(screen, (0, 128, 0), False, points, 2)

    def simulate_sensor_data(self):
        # Update pulse
        self.driver_state.pulse += random.uniform(-2, 2)
        self.driver_state.pulse = max(60, min(100, self.driver_state.pulse))

        # Update eyelid movement
        self.driver_state.eyelid_movement += random.uniform(-0.05, 0.03)
        self.driver_state.eyelid_movement = max(
            0.3, min(1.0, self.driver_state.eyelid_movement))

        # Update speed
        self.driver_state.current_speed += random.uniform(-5, 5)
        self.driver_state.current_speed = max(
            0, min(130, self.driver_state.current_speed))

        # Update time of day
        current_hour = (self.tick_count * 5 // 60) % 24  # Simulated hour
        self.driver_state.time_of_day = "night" if current_hour < 6 or current_hour > 20 else "day"

        # Randomly change weather with a very low probability (once per day on average)
        if random.random() < WEATHER_CHANGE_PROB:  # ~0.35% chance per tick
            self.driver_state.weather_condition = random.choice(
                ["clear", "rain", "fog", "snow", "bad"])

    def update_simulation(self):
        while self.simulation_running:
            if not self.paused:  # Only update if not paused
                self.simulate_sensor_data()
                self.tick_count += 1  # Increment tick count
            time.sleep(1/SIMULATION_SPEED_TICKS_PER_SECOND)

    def update_single_tick(self):
        """Update simulation for a single tick when space is pressed"""
        if self.paused:
            self.tick_count += 1  # Increment tick count
            self.simulate_sensor_data()
            self.update_display_data()

    def update_display_data(self):
        """Update display data without running simulation"""
        rest_loss = self.calculate_rest_points_loss()
        base_rest_loss = 0.5  # Base rest loss
        # The risk probability is derived from the rest_loss
        risk_prob = rest_loss / base_rest_loss

        # Update the rest points
        self.driver_state.rest_points -= rest_loss
        self.rest_points_history.append(self.driver_state.rest_points)

        if self.driver_state.rest_points < self.rest_threshold:
            self.driver_state.rest_points = 100
            self.driver_state.last_rest_tick = self.tick_count  # Update last rest tick

        # Add these values to the display data
        self.display_data = {
            'Rest Loss': rest_loss,
            'Base Rest Loss': base_rest_loss,
            'Risk Probability': risk_prob  # Directly showing risk probability
        }

    def toggle_simulation(self):
        """Toggle simulation pause state"""
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

                time_since_rest_formatted = self.calculate_time_since_rest()

                # Display key information
                texts = [
                    f"Tick: {self.tick_count}",
                    f"Rest Points: {self.driver_state.rest_points:.1f}",
                    f"Speed: {self.driver_state.current_speed:.1f} km/h",
                    f"Pulse: {self.driver_state.pulse:.1f}",
                    f"Eyelid Movement: {self.driver_state.eyelid_movement:.2f}",
                    f"Time of Day: {self.driver_state.time_of_day}",
                    f"Weather: {self.driver_state.weather_condition}",
                    f"Traffic: {self.driver_state.traffic_density}",
                    f"Road Type: {self.driver_state.road_type}",
                    f"Time Since Last Rest: {time_since_rest_formatted}",
                    f"Base Rest Loss: {self.display_data['Base Rest Loss']:.2f}",
                    f"Risk Probability: {self.display_data['Risk Probability']:.2f}",
                    f"Rest Loss: {self.display_data['Rest Loss']:.2f}"
                ]

                # Render the texts
                for i, text in enumerate(texts):
                    surface = font.render(text, True, (0, 0, 0))
                    screen.blit(surface, (50, 300 + i * 40))

                pygame.display.flip()
                clock.tick(SIMULATION_SPEED_TICKS_PER_SECOND)

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
