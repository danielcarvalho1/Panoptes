"""Synthetic multi-layer landscapes for the Panoptes prototype."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Terrain:
    """Hidden landscape layers used by the simulator."""

    elevation: np.ndarray
    slope: np.ndarray
    obstacle_field: np.ndarray
    anomaly: np.ndarray
    surface_signal: np.ndarray
    traversal_penalty: np.ndarray
    feature_types: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()
    return (values - minimum) / (maximum - minimum + 1e-8)


def _gaussian_field(
    rows: int, columns: int, rng: np.random.Generator, count: int
) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:columns]
    field = np.zeros((rows, columns), dtype=np.float32)
    for _ in range(count):
        center_y = rng.uniform(0, rows)
        center_x = rng.uniform(0, columns)
        width = rng.uniform(1.3, 4.5)
        amplitude = rng.uniform(0.65, 1.0)
        field += amplitude * np.exp(
            -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * width**2)
        )
    return _normalise(field).astype(np.float32)


def _line_feature(rows: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:columns]
    start = rng.uniform(0, columns)
    slope = rng.uniform(-0.7, 0.7)
    line = start + slope * y
    distance = np.abs(x - line)
    width = rng.uniform(0.7, 1.5)
    return np.exp(-(distance**2) / (2 * width**2)).astype(np.float32)


def generate_terrain(
    size: tuple[int, int] = (24, 24), seed: int = 7
) -> Terrain:
    """Generate terrain with interpretable environmental and hidden layers."""

    rows, columns = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:columns]

    elevation = (
        0.45 * np.sin(x / 3.8)
        + 0.3 * np.cos(y / 5.1)
        + 0.18 * np.sin((x + y) / 7.0)
        + rng.normal(0, 0.04, size)
    )
    elevation = _normalise(elevation).astype(np.float32)
    gradient_y, gradient_x = np.gradient(elevation)
    slope = _normalise(np.hypot(gradient_x, gradient_y)).astype(np.float32)

    obstacle_field = _gaussian_field(rows, columns, rng, count=5)

    cluster_field = _gaussian_field(rows, columns, rng, count=3)
    isolated_field = _gaussian_field(rows, columns, rng, count=4)
    diffuse_field = _gaussian_field(rows, columns, rng, count=2)
    linear_field = _line_feature(rows, columns, rng)
    anomaly = _normalise(
        0.48 * cluster_field
        + 0.18 * isolated_field**2
        + 0.18 * diffuse_field
        + 0.16 * linear_field
    ).astype(np.float32)
    environmental_prior = (
        0.7 * (1.0 - slope) + 0.3 * elevation
    )
    surface_signal = np.clip(
        0.72 * anomaly + 0.28 * environmental_prior, 0.0, 1.0
    ).astype(np.float32)
    start_row, start_column = rows // 2, columns // 2
    obstacle_field[start_row, start_column] = 0.0
    high_cost = (slope > 0.78) | (obstacle_field > 0.72)
    high_cost[start_row, start_column] = False
    traversal_penalty = (
        1.2 * slope + 2.5 * high_cost.astype(np.float32)
    ).astype(np.float32)

    return Terrain(
        elevation,
        slope,
        obstacle_field,
        anomaly,
        surface_signal,
        traversal_penalty,
        ("clustered sites", "linear features", "isolated anomalies", "diffuse areas"),
    )
