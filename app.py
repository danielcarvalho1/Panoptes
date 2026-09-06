"""Streamlit visualisation for Panoptes experiments."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from Panoptes.baselines import POLICIES, run_policy
from Panoptes.env import PanoptesEnv
from Panoptes.terrain import generate_terrain

st.set_page_config(page_title="Panoptes", page_icon="browser_logo.png", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Trebuchet MS', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif;
    }
    .panoptes-masthead {
        margin-top: -2.5rem;
        margin-bottom: 2rem;
    }
    .panoptes-masthead p {
        color: #475569;
        font-size: 1.05rem;
        margin: 0.35rem 0 0;
    }
    .panoptes-content-divider {
        border-top: 1px solid #cbd5e1;
        margin: 2.5rem 0 2rem;
    }
    .panoptes-footer {
        color: #6b7280;
        font-size: 0.8rem;
        text-align: right;
        margin-top: 3rem;
        padding: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="panoptes-masthead">', unsafe_allow_html=True)
st.image("Panoptes_logo.png", width=820)
st.markdown(
    '<p>An archaeological survey simulator that aims to reduce uncertainty and generate synthetic data for training AI models.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment")
    view = st.radio("View", ["Single survey", "Compare methods"], index=0)
    budget = st.slider("Survey budget", min_value=100.0, max_value=300.0, value=200.0, step=1.0)
    if view == "Single survey":
        terrain_seed = st.number_input("Landscape seed", min_value=0, value=7, step=1)


def show_metrics(result: dict[str, object]) -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Information gain", f"{result['information_gain']:.2f}")
    metric_columns[1].metric("Uncertainty reduced", f"{result.get('uncertainty_reduced_percent', 0.0):.1f}%")
    metric_columns[2].metric("Scientific yield", f"{result['scientific_yield']:.2f}")
    metric_columns[3].metric("Coverage", f"{result['coverage']:.1%}")
    metric_columns[4].metric("Survey cost", f"{result['cost']:.1f}")


def show_path(result: dict[str, object], policy_name: str) -> None:
    terrain = result["terrain"]
    path = result["path"]
    rows, columns = zip(*path)
    sampled_positions = result.get("sampled_positions", [])
    high_cost = (terrain.traversal_penalty >= 2.5).astype(float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(terrain.surface_signal, cmap="magma", vmin=0, vmax=1)
    axes[0].contour(high_cost, levels=[0.5], colors="#fb923c", linewidths=1.4, linestyles="--")
    axes[0].plot(columns, rows, color="white", linewidth=1.2, alpha=0.85)
    if sampled_positions:
        sampled_rows, sampled_columns = zip(*sampled_positions)
        axes[0].scatter(sampled_columns, sampled_rows, color="#ef4444", s=28, label="Intensive sample", zorder=3)
    axes[0].scatter(columns[0], rows[0], color="#4ade80", s=45, label="Start")
    axes[0].scatter(columns[-1], rows[-1], color="#38bdf8", s=45, label="End")
    axes[0].set_title(f"{policy_name}: signal + soft high-cost terrain")
    axes[0].legend(loc="upper right")
    axes[1].imshow(terrain.anomaly, cmap="viridis", vmin=0, vmax=1)
    axes[1].contour(high_cost, levels=[0.5], colors="#fb923c", linewidths=1.4, linestyles="--")
    axes[1].set_title("Hidden anomaly field + soft high-cost terrain")
    for axis in axes:
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
    figure.tight_layout()
    st.pyplot(figure)
    st.info("Orange dashed outlines mark soft high-cost terrain. The anomaly field is shown for evaluation only; a real agent would see noisy readings and its evolving belief, not the ground truth.")


def show_scenario_summary(terrain, landscape_count: int = 1, runs: int = 1) -> None:
    high_cost_cells = terrain.traversal_penalty >= 2.5
    summary = {
        "Landscapes": landscape_count,
        "Grid": f"{terrain.shape[0]} x {terrain.shape[1]} cells",
        "Runs per landscape": runs,
        "Average slope": f"{terrain.slope.mean():.2f}",
        "High-cost terrain": f"{high_cost_cells.mean():.1%} of cells",
        "Feature types": ", ".join(terrain.feature_types),
    }
    with st.expander("Scenario summary", expanded=True):
        st.write(summary)


if view == "Single survey":
    policy_name = st.sidebar.selectbox("Survey policy", list(POLICIES))
    run = st.sidebar.button("Run survey", type="primary")
    if run or "single_result" not in st.session_state:
        terrain = generate_terrain(size=(24, 24), seed=int(terrain_seed))
        environment = PanoptesEnv(terrain=terrain, budget=budget)
        st.session_state.single_result = run_policy(POLICIES[policy_name], environment, int(terrain_seed))
        st.session_state.single_policy = policy_name
    st.markdown('<div class="panoptes-content-divider"></div>', unsafe_allow_html=True)
    show_path(st.session_state.single_result, st.session_state.single_policy)
    show_metrics(st.session_state.single_result)
    show_scenario_summary(st.session_state.single_result["terrain"])

elif view == "Compare methods":
    landscape_count = st.sidebar.slider(
        "Number of landscapes", min_value=2, max_value=100, value=10, step=1
    )
    runs = st.sidebar.slider("Runs per landscape", min_value=1, max_value=100, value=5, step=1)
    landscape_seeds = [7 + index for index in range(landscape_count)]
    criterion = st.sidebar.selectbox(
        "Choose the best method by",
        [
            "Information gain",
            "Uncertainty reduced (%)",
            "Scientific yield",
            "Coverage",
            "Lowest cost",
        ],
    )
    compare = st.sidebar.button("Run comparison", type="primary")
    comparison_key = (landscape_count, runs, float(budget))
    if compare or st.session_state.get("comparison_key") != comparison_key:
        records = []
        total_runs = len(landscape_seeds) * runs * len(POLICIES)
        progress = st.progress(0, text="Running comparison...")
        completed_runs = 0
        for landscape_number, landscape_seed in enumerate(landscape_seeds, start=1):
            terrain = generate_terrain(size=(24, 24), seed=landscape_seed)
            for policy_name, policy in POLICIES.items():
                for run_number in range(runs):
                    environment = PanoptesEnv(terrain=terrain, budget=budget)
                    sensor_seed = landscape_seed * 1000 + run_number
                    result = run_policy(policy, environment, seed=sensor_seed)
                    records.append({
                        "Policy": policy_name,
                        "Landscape": landscape_number,
                        "Landscape seed": landscape_seed,
                        "Run": run_number + 1,
                        "Information gain": result["information_gain"],
                        "Uncertainty reduced (%)": result["uncertainty_reduced_percent"],
                        "Scientific yield": result["scientific_yield"],
                        "Coverage": result["coverage"],
                        "Cost": result["cost"],
                    })
                    completed_runs += 1
                    progress.progress(completed_runs / total_runs, text=f"Running comparison... {completed_runs}/{total_runs}")
        st.session_state.comparison = pd.DataFrame(records)
        st.session_state.comparison_key = comparison_key
        progress.empty()

    comparison = st.session_state.comparison
    summary = comparison.groupby("Policy").agg({
        "Information gain": ["mean", "std"],
        "Uncertainty reduced (%)": ["mean", "std"],
        "Scientific yield": ["mean", "std"],
        "Coverage": ["mean", "std"],
        "Cost": ["mean", "std"],
    }).round(3)
    ranking_metric = "Cost" if criterion == "Lowest cost" else criterion
    st.subheader("Average performance across repeated runs")
    st.dataframe(summary, use_container_width=True)
    show_scenario_summary(generate_terrain(size=(24, 24), seed=landscape_seeds[0]), len(landscape_seeds), runs)
    st.caption(f"Every method was tested on the same {len(landscape_seeds)} landscapes, with {runs} runs per landscape and the same budget.")
    st.caption("More runs improve the average estimate; they do not change the budget of an individual survey. Scientific yield and random-policy results should vary across landscapes, while coverage and cost are mainly controlled by the budget.")
    chart_data = comparison.groupby("Policy")[ranking_metric].mean().sort_values(
        ascending=criterion == "Lowest cost"
    )
    st.subheader(f"Average {criterion.lower()} by method")
    st.bar_chart(chart_data)

st.divider()
st.header("Glossary")
st.markdown(
    """
    **Survey policies**  
    Random chooses actions without a plan. Lawnmower systematically sweeps the grid. Information gain targets uncertain areas while considering terrain cost. Greedy signal targets promising observations.

    **Information gain**  
    How much the agent reduced uncertainty about the hidden landscape. The information-gain policy also considers the estimated cost of reaching a location. Higher means it learned more, not necessarily that it discovered a site.

    **Uncertainty reduced**  
    Information gain expressed as a percentage of the uncertainty present at the start of the survey. It measures learning progress, not the percentage of archaeological sites found.

    **Scientific yield**  
    The amount of strong sensor signal found during intensive sampling. Higher means more promising evidence was detected.

    **Coverage**  
    The fraction of grid cells visited at least once. Higher means broader geographic coverage.

    **Survey cost**  
    The budget spent on movement and intensive sampling. Lower is cheaper, but a cheap survey may learn less.

    **Soft terrain penalty**  
    Difficult terrain remains traversable, but movement costs more. Steep areas and synthetic obstacle patches receive larger penalties instead of becoming forbidden.

    **Synthetic feature types**  
    The hidden archaeological field combines clustered sites, linear features such as roads or walls, isolated anomalies, and diffuse areas. They are simulated patterns, not real archaeological classifications.

    **Reward**  
    A combined score balancing scientific yield, survey cost, and information gain. The separate metrics are easier to interpret.

    **Landscape seed**  
    The number used to generate the hidden terrain and archaeological anomaly pattern. The same landscape seed means every policy is tested on the same landscape.

    **Repeated runs**  
    One run is one possible noisy survey. Comparing averages and standard deviations across many runs shows which method is reliable rather than lucky.
    """
)
st.markdown('<div class="panoptes-footer">Made by Daniel Carvalho / 2026</div>', unsafe_allow_html=True)
