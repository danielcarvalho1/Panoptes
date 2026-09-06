# Panoptes

![logo](Panoptes_logo.png)

Panoptes is a small, reproducible research prototype for sequential survey strategy under uncertainty. It models an archaeological surveyor or drone exploring a 2D landscape with noisy observations and a finite movement and sampling budget.

The project is designed as a portfolio foundation for research at the intersection of archaeological prospection, spatial modelling, and reinforcement learning. Its first experiment is intentionally synthetic: the simulator knows the hidden archaeological anomaly field, while the agent only receives noisy local sensor readings and an evolving uncertainty map.

Panoptes is live! You can use it at https://panoptes-arch.streamlit.app/

## Research question

> Under a fixed survey budget and noisy partial observations, how do simple spatial survey policies differ in uncertainty reduction, scientific yield, cost, and coverage?

The primary outcome is **uncertainty reduction**. Scientific yield, cost, coverage, and path length are retained as separate metrics so that the trade-offs remain interpretable instead of being hidden inside a single scalar score.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python benchmark.py
streamlit run app.py
```

The benchmark writes episode-level metrics, summary tables, and a comparison plot to `results/`. The Streamlit app supports a single survey, repeated comparisons across all policies, and a plain-language guide to the metrics. Compare methods always evaluates multiple landscape seeds, with a configurable number of runs per landscape. Every policy uses the same landscape-seed set and budget. The hidden field is shown only as an evaluation aid; it is not part of the agent observation.

## Project structure

```text
Panoptes/
├── Panoptes/
│   ├── __init__.py
│   ├── terrain.py       # Synthetic elevation, slope, obstacles, and anomaly layers
│   ├── env.py           # Gymnasium-compatible partially observed environment
│   └── baselines.py     # Random, lawnmower, signal, and information-gain policies
├── benchmark.py         # Repeated seeded evaluation and plots
├── app.py               # Streamlit path visualisation
├── train_ppo.py         # Optional PPO reinforcement-learning baseline
├── requirements.txt
└── README.md
```

## Environment design

The hidden terrain contains elevation, slope, synthetic obstacle patches, an archaeological anomaly field, a noisy surface signal, and soft traversal penalties. Steep terrain and obstacle cells remain traversable but consume more budget. The anomaly field mixes clustered sites, linear features, isolated anomalies, and diffuse areas. The agent receives visited cells, observed readings, its current belief, uncertainty, position, and remaining budget.

Actions are eight-direction movement or intensive local sampling. Movement costs distance; intensive sampling costs more but reduces local uncertainty more strongly. The reward is a transparent proxy:

$$R_t = w_y \Delta Y_t - w_c C_t + w_i (H_{t-1} - H_t)$$

where $H_t$ is total uncertainty. The benchmark reports each component independently.

The budget is a hard constraint: an action is taken only when its full cost fits within the remaining budget, so reported survey cost is always less than or equal to the configured budget.

## Baselines

- **Random:** selects actions uniformly.
- **Lawnmower:** systematic coverage pattern.
- **Information gain:** targets uncertain, accessible cells using an estimated terrain-adjusted travel cost.
- **Greedy signal:** intensively samples promising observed signals and explores low-cost neighbors when evidence is weak.

The optional `train_ppo.py` script uses Stable-Baselines3 PPO with a multi-input policy over the dictionary observation. PPO should be compared against these interpretable baselines rather than replacing them. Training is kept separate from `benchmark.py` because it is slower and stochastic.

## Real data extension

The recommended open-data extension is Sentinel-2 Level-2A imagery paired with Copernicus DEM GLO-30 and, optionally, ESA WorldCover. These layers can supply environmental proxies such as vegetation, bare soil, elevation, slope, and land-cover masks. They should not be described as direct archaeological evidence without validation against field or heritage data.

A real-data adapter should preserve the same interface as `Terrain`, while keeping archaeological ground truth and evaluation data separate from agent observations. Potential sources include Copernicus Data Space, Microsoft Planetary Computer STAC, USGS EarthExplorer, and regional open archaeological registers. Always check the licence and spatial resolution for the selected site.

## Limitations and next research steps

This is a simulation, not a validated archaeological model. The anomaly generator is synthetic, the sensor likelihood is simplified, and the uncertainty map is a transparent heuristic rather than a calibrated posterior. Good next steps are calibrated Bayesian updates, multiple anomaly types including linear features, real seasonal imagery, historical site data, multi-objective policy evaluation, and PPO or other RL agents trained across landscape seeds.
