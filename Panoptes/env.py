"""Gymnasium environment for partially observed spatial survey."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .terrain import Terrain, generate_terrain


class PanoptesEnv(gym.Env):
    """Explore a hidden archaeological landscape under a finite survey budget."""

    metadata = {"render_modes": ["human"]}
    MOVE_DELTAS = np.array(
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    )

    def __init__(
        self,
        terrain: Terrain | None = None,
        size: tuple[int, int] = (24, 24),
        budget: float = 45.0,
        movement_cost: float = 1.0,
        sample_cost: float = 4.0,
        sensor_noise: float = 0.12,
        reward_weights: tuple[float, float, float] = (2.0, 0.08, 1.0),
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.terrain = terrain or generate_terrain(size=size)
        self.rows, self.columns = self.terrain.shape
        self.initial_budget = budget
        self.movement_cost = movement_cost
        self.sample_cost = sample_cost
        self.sensor_noise = sensor_noise
        self.yield_weight, self.cost_weight, self.info_weight = reward_weights
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "visited": spaces.Box(0.0, 1.0, (self.rows, self.columns), np.float32),
                "readings": spaces.Box(-1.0, 1.0, (self.rows, self.columns), np.float32),
                "belief": spaces.Box(0.0, 1.0, (self.rows, self.columns), np.float32),
                "uncertainty": spaces.Box(0.0, 1.0, (self.rows, self.columns), np.float32),
                "position": spaces.Box(0.0, 1.0, (2,), np.float32),
                "budget": spaces.Box(0.0, 1.0, (1,), np.float32),
            }
        )
        self.position = np.zeros(2, dtype=np.int32)
        self.visited = np.zeros((self.rows, self.columns), dtype=np.float32)
        self.sampled = np.zeros((self.rows, self.columns), dtype=np.float32)
        self.readings = np.full((self.rows, self.columns), -1.0, dtype=np.float32)
        self.belief = np.full((self.rows, self.columns), 0.5, dtype=np.float32)
        self.uncertainty = np.ones((self.rows, self.columns), dtype=np.float32)
        self.budget = budget
        self.discovered_yield = 0.0
        self.path: list[tuple[int, int]] = []

    def _observe_cell(self, intensive: bool = False) -> float:
        row, column = self.position
        noise = self.sensor_noise / (2.5 if intensive else 1.0)
        reading = float(np.clip(self.terrain.surface_signal[row, column] + self.np_random.normal(0, noise), 0, 1))
        self.visited[row, column] = 1.0
        self.readings[row, column] = reading
        confidence = 0.82 if intensive else 0.38
        old_belief = self.belief[row, column]
        self.belief[row, column] = (1 - confidence) * old_belief + confidence * reading
        self.uncertainty[row, column] *= 1.0 - confidence
        return reading

    def _update_spatial_uncertainty(self, radius: int = 3) -> None:
        row, column = self.position
        row_start, row_end = max(0, row - radius), min(self.rows, row + radius + 1)
        column_start, column_end = max(0, column - radius), min(self.columns, column + radius + 1)
        local_y, local_x = np.mgrid[row_start:row_end, column_start:column_end]
        distance = np.hypot(local_y - row, local_x - column)
        reduction = 0.22 * np.exp(-(distance**2) / 4.0)
        self.uncertainty[row_start:row_end, column_start:column_end] *= 1.0 - reduction

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "visited": self.visited.copy(),
            "readings": self.readings.copy(),
            "belief": self.belief.copy(),
            "uncertainty": self.uncertainty.copy(),
            "position": (self.position.astype(np.float32) / np.array([self.rows - 1, self.columns - 1])).astype(np.float32),
            "budget": np.array([self.budget / self.initial_budget], dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.position = np.array([self.rows // 2, self.columns // 2], dtype=np.int32)
        self.visited.fill(0.0)
        self.sampled.fill(0.0)
        self.readings.fill(-1.0)
        self.belief.fill(0.5)
        self.uncertainty.fill(1.0)
        self.budget = self.initial_budget
        self.discovered_yield = 0.0
        self.path = [tuple(self.position)]
        self._observe_cell()
        return self._get_obs(), {"uncertainty": float(self.uncertainty.sum())}

    def step(self, action: int):
        previous_uncertainty = float(self.uncertainty.sum())
        movement_cost = 0.0
        scientific_yield = 0.0
        action_name = "sample"

        if action < 8:
            action_name = "move"
            delta = self.MOVE_DELTAS[action]
            next_position = self.position + delta
            base_distance = 1.4142 if abs(delta).sum() == 2 else 1.0
            if 0 <= next_position[0] < self.rows and 0 <= next_position[1] < self.columns:
                terrain_multiplier = 1.0 + float(self.terrain.traversal_penalty[tuple(next_position)])
                proposed_cost = self.movement_cost * base_distance * terrain_multiplier
                if self.budget >= proposed_cost:
                    self.position = next_position
                    movement_cost = proposed_cost
                    self.budget -= movement_cost
                    self._observe_cell()
                else:
                    action_name = "insufficient_budget"
                    self.budget = 0.0
            else:
                action_name = "invalid_move"
        else:
            if self.budget >= self.sample_cost:
                self.budget -= self.sample_cost
                reading = self._observe_cell(intensive=True)
                self.sampled[tuple(self.position)] = 1.0
                self._update_spatial_uncertainty()
                scientific_yield = max(0.0, reading - 0.55)
                self.discovered_yield += scientific_yield
            else:
                action_name = "insufficient_budget"
                self.budget = 0.0

        information_gain = max(0.0, previous_uncertainty - float(self.uncertainty.sum()))
        reward = (
            self.yield_weight * scientific_yield
            - self.cost_weight * (movement_cost or (self.sample_cost if action_name == "sample" else 0.0))
            + self.info_weight * information_gain / (self.rows * self.columns)
        )
        self.path.append(tuple(self.position))
        terminated = self.budget <= 0.0
        truncated = len(self.path) >= 300
        info = {
            "action": action_name,
            "scientific_yield": scientific_yield,
            "information_gain": information_gain,
            "movement_cost": movement_cost,
            "budget": self.budget,
            "position": tuple(self.position),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        return {"position": tuple(self.position), "budget": self.budget}
