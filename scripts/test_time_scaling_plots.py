import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Name of your dataset
repo_ids = [
    "sibasmarakp/Llama-3.2-1B-Instruct-dvts-completions",
    "sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions",
    "sibasmarakp/Llama-3.2-1B-Instruct-best_of_n-completions",
    "sibasmarakp/Qwen2.5-1.5B-Instruct-best_of_n-completions",
]

# List of seed eval subset names
seed_subsets = [
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--seed-0--agg_strategy-last--evals",
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--seed-1--agg_strategy-last--evals",
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--seed-2--agg_strategy-last--evals"
]

map_repo_id_to_policy = {
    "sibasmarakp/Llama-3.2-1B-Instruct-dvts-completions": "Llama-3.2-1B-Instruct",
    "sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions": "Qwen2.5-1.5B-Instruct",
    "sibasmarakp/Llama-3.2-1B-Instruct-best_of_n-completions": "Llama-3.2-1B-Instruct",
    "sibasmarakp/Qwen2.5-1.5B-Instruct-best_of_n-completions": "Qwen2.5-1.5B-Instruct",
}

dfs = []
for repo_id in repo_ids:
    for seed_subset in seed_subsets:
        print(f"Loading {seed_subset} for {repo_id} …")
        ds = load_dataset(repo_id, name=seed_subset, split="train")
        df = ds.to_pandas()
        df["policy"] = map_repo_id_to_policy[repo_id]
        df["test_time_approach"] = repo_id.split("-")[-2]
        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


# Plot settings
plt.rcParams["font.family"] = "DejaVu Serif"
sns.set(style="whitegrid")
palette = {
    "Best-of-N (weighted)": "#2ca02c",
    "Best-of-N (majority)": "#1f77b4",
    "DVTS": "#ff7f0e",
}

# Helper to make the x-axis show powers of two nicely
def _format_pow_two_ticks(axis, values):
    axis.set_xscale("log", base=2)
    axis.set_xticks(values)
    axis.set_xticklabels([f"$2^{int(np.log2(v))}$" if v > 1 else "1" for v in values])
    axis.get_xaxis().set_major_formatter(ScalarFormatter())

# Zero-shot reference lines
zero_shot_refs = [
    ("Llama 3.1 8B", 49.8),
    ("Llama 3.2 1B", 26.0),
    ("Qwen2.5 1.5B", 54.4),
    ("Qwen2.5 14B", 80.2)
]

# Make one plot per policy
for policy in combined_df["policy"].unique():
    policy_df = combined_df[combined_df["policy"] == policy].copy()
    policy_df["n"] = policy_df["n"].astype(int)
    n_values = sorted(policy_df["n"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    best_of_n = policy_df[policy_df["test_time_approach"] == "best_of_n"]
    dvts = policy_df[policy_df["test_time_approach"] == "dvts"]

    # Best-of-N (weighted)
    if not best_of_n.empty:
        sns.lineplot(
            data=best_of_n,
            x="n",
            y="acc_weighted",
            marker="o",
            label="Best-of-N (weighted)",
            color=palette["Best-of-N (weighted)"],
            ax=ax,
        )
        # Best-of-N (majority)
        sns.lineplot(
            data=best_of_n,
            x="n",
            y="acc_maj",
            marker="o",
            label="Best-of-N (majority)",
            color=palette["Best-of-N (majority)"],
            ax=ax,
        )

    # DVTS (weighted) — ignore majority line
    if not dvts.empty:
        sns.lineplot(
            data=dvts,
            x="n",
            y="acc_weighted",
            marker="o",
            label="DVTS",
            color=palette["DVTS"],
            ax=ax,
        )

    # Zero-shot horizontal dotted lines
    for label, value in zero_shot_refs:
        ax.axhline(value, linestyle="--", color="black", linewidth=1, alpha=0.7)
        text_transform = ax.get_yaxis_transform()
        ax.text(
            0.02,
            value + 0.1,
            label,
            transform=text_transform,
            ha="left",
            va="bottom",
            fontsize=10,
        )

    _format_pow_two_ticks(ax, n_values)
    ax.set_xlabel("Number of generations per problem")
    ax.set_ylabel("MATH-500 accuracy (%)")
    ax.set_title(policy)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = f"{policy.replace(' ', '_')}_time_scaling.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

