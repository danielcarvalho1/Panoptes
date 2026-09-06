"""Run reproducible comparisons between survey policies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Panoptes.baselines import POLICIES, run_policy
from Panoptes.env import PanoptesEnv
from Panoptes.terrain import generate_terrain


def run_benchmark(
    landscape_count: int = 100,
    runs_per_landscape: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    records = []
    for landscape_number in range(landscape_count):
        landscape_seed = seed + landscape_number
        terrain = generate_terrain(size=(24, 24), seed=landscape_seed)
        for policy_name, policy in POLICIES.items():
            for run_number in range(runs_per_landscape):
                sensor_seed = landscape_seed * 1000 + run_number
                env = PanoptesEnv(terrain=terrain, budget=45.0)
                result = run_policy(policy, env, seed=sensor_seed)
                records.append(
                    {
                        "policy": policy_name,
                        "landscape": landscape_number + 1,
                        "landscape_seed": landscape_seed,
                        "run": run_number + 1,
                        "sensor_seed": sensor_seed,
                        "reward": result["reward"],
                        "information_gain": result["information_gain"],
                        "uncertainty_reduced_percent": result["uncertainty_reduced_percent"],
                        "scientific_yield": result["scientific_yield"],
                        "cost": result["cost"],
                        "coverage": result["coverage"],
                    }
                )
    return pd.DataFrame(records)


def save_summary(results: pd.DataFrame, output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    results.to_csv(output_path / "episode_metrics.csv", index=False)
    results[["landscape", "landscape_seed", "run", "sensor_seed"]].drop_duplicates().to_csv(
        output_path / "seed_manifest.csv", index=False
    )
    metric_columns = [
        "reward",
        "information_gain",
        "uncertainty_reduced_percent",
        "scientific_yield",
        "cost",
        "coverage",
    ]
    summary = results.groupby("policy")[metric_columns].agg(["mean", "std"]).round(4)
    summary.to_csv(output_path / "summary_metrics.csv")
    confidence = results.groupby("policy")[metric_columns].agg(["count", "mean", "std"])
    for metric in metric_columns:
        count = confidence[(metric, "count")]
        mean = confidence[(metric, "mean")]
        standard_error = confidence[(metric, "std")] / np.sqrt(count)
        confidence[(metric, "ci95_low")] = mean - 1.96 * standard_error
        confidence[(metric, "ci95_high")] = mean + 1.96 * standard_error
    confidence.round(4).to_csv(output_path / "confidence_intervals.csv")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    results.boxplot(column="information_gain", by="policy", ax=axes[0], grid=False)
    axes[0].set_title("Uncertainty reduction")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Cumulative information gain")
    results.boxplot(column="scientific_yield", by="policy", ax=axes[1], grid=False)
    axes[1].set_title("Scientific yield")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Discovered signal")
    figure.suptitle("")
    figure.tight_layout()
    figure.savefig(output_path / "policy_comparison.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    save_summary(benchmark_results)
    print(benchmark_results.groupby("policy")["information_gain"].agg(["mean", "std"]).round(3))
