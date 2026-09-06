"""Optional PPO baseline for Panoptes.

Run this after installing requirements.txt. Training is intentionally separate
from benchmark.py so classical baselines remain quick and reproducible.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from Panoptes.env import PanoptesEnv


if __name__ == "__main__":
    environment = PanoptesEnv(size=(16, 16), budget=45.0)
    check_env(environment, warn=True)
    model = PPO(
        "MultiInputPolicy",
        environment,
        verbose=1,
        seed=7,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=25_000)
    output_path = Path("results")
    output_path.mkdir(exist_ok=True)
    model.save(output_path / "panoptes_ppo")
    print(f"Saved PPO model to {output_path / 'panoptes_ppo'}.zip")
