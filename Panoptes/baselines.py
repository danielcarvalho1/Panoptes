"""Simple interpretable survey policies."""

from __future__ import annotations

import numpy as np

from .env import PanoptesEnv


def _move_towards(env: PanoptesEnv, target: tuple[int, int]) -> int:
    delta = np.sign(np.asarray(target) - env.position)
    if delta[0] != 0 and delta[1] != 0:
        return int({(-1, -1): 4, (-1, 1): 5, (1, -1): 6, (1, 1): 7}[tuple(delta)])
    if delta[0] < 0:
        return 0
    if delta[0] > 0:
        return 1
    if delta[1] < 0:
        return 2
    if delta[1] > 0:
        return 3
    return 8


def random_policy(env: PanoptesEnv, rng: np.random.Generator) -> int:
    return int(rng.integers(0, 9))


def lawnmower_policy(env: PanoptesEnv, rng: np.random.Generator) -> int:
    current_row, current_column = env.position
    if env.sampled[current_row, current_column] == 0.0 and env.budget >= env.sample_cost:
        return 8
    row, column = env.position
    if row % 2 == 0 and column < env.columns - 1:
        preferred = (row, column + 1)
        if env.terrain.traversal_penalty[preferred] > env.terrain.traversal_penalty[row, column] + 0.8 and row < env.rows - 1:
            return 1
        return 3
    if row % 2 == 1 and column > 0:
        preferred = (row, column - 1)
        if env.terrain.traversal_penalty[preferred] > env.terrain.traversal_penalty[row, column] + 0.8 and row < env.rows - 1:
            return 1
        return 2
    if row < env.rows - 1:
        return 1
    return 8


def information_gain_policy(env: PanoptesEnv, rng: np.random.Generator) -> int:
    current_row, current_column = env.position
    if env.sampled[current_row, current_column] == 0.0 and env.budget >= env.sample_cost:
        return 8
    rows, columns = np.indices(env.uncertainty.shape)
    distance = np.hypot(rows - current_row, columns - current_column)
    travel_cost = (distance + 1.0) * (1.0 + env.terrain.traversal_penalty)
    candidate_value = env.uncertainty / travel_cost
    candidate_value[env.sampled > 0.0] = -1.0
    target = np.unravel_index(np.argmax(candidate_value), env.uncertainty.shape)
    return _move_towards(env, target)


def greedy_signal_policy(env: PanoptesEnv, rng: np.random.Generator) -> int:
    current_row, current_column = env.position
    current_belief = env.belief[current_row, current_column]
    if (
        env.sampled[current_row, current_column] == 0.0
        and current_belief >= 0.55
        and env.budget >= env.sample_cost
    ):
        return 8

    observed_candidates = (env.visited > 0.0) & (env.sampled == 0.0)
    promising_candidates = observed_candidates & (env.belief >= 0.55)
    if np.any(promising_candidates):
        rows, columns = np.indices(env.belief.shape)
        distance = np.hypot(rows - current_row, columns - current_column)
        travel_cost = (distance + 1.0) * (1.0 + env.terrain.traversal_penalty)
        candidate_belief = np.where(promising_candidates, env.belief / travel_cost, -1.0)
        target = np.unravel_index(np.argmax(candidate_belief), env.belief.shape)
    else:
        legal_actions = []
        action_costs = []
        for action, delta in enumerate(env.MOVE_DELTAS):
            next_position = env.position + delta
            if 0 <= next_position[0] < env.rows and 0 <= next_position[1] < env.columns:
                legal_actions.append(action)
                action_costs.append(1.0 + float(env.terrain.traversal_penalty[tuple(next_position)]))
        weights = 1.0 / np.asarray(action_costs)
        weights /= weights.sum()
        return int(rng.choice(legal_actions, p=weights))
    if tuple(env.position) == tuple(target) and env.budget >= env.sample_cost:
        return 8
    return _move_towards(env, target)


def run_policy(
    policy, env: PanoptesEnv, seed: int = 0
) -> dict[str, object]:
    observation, reset_info = env.reset(seed=seed)
    initial_uncertainty = reset_info["uncertainty"]
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    total_information_gain = 0.0
    total_yield = 0.0
    total_cost = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(env, rng)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_information_gain += info["information_gain"]
        total_yield += info["scientific_yield"]
        total_cost += info["movement_cost"]
        if info["action"] == "sample":
            total_cost += env.sample_cost
    return {
        "reward": total_reward,
        "information_gain": total_information_gain,
        "uncertainty_reduced_percent": 100.0 * total_information_gain / initial_uncertainty,
        "scientific_yield": total_yield,
        "cost": total_cost,
        "coverage": float(env.visited.mean()),
        "sampled_positions": [
            (int(row), int(column))
            for row, column in zip(*np.where(env.sampled > 0.0))
        ],
        "path": env.path.copy(),
        "terrain": env.terrain,
    }


POLICIES = {
    "Random": random_policy,
    "Lawnmower": lawnmower_policy,
    "Information gain": information_gain_policy,
    "Greedy signal": greedy_signal_policy,
}
