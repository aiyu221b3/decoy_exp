from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE.parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ============================================================
# Project-local fonts
# ============================================================

FONT_DIR = HERE.parents[1] / "fonts"
for font_path in FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(font_path)

# ============================================================
# Visual language
# ============================================================

CREAM = "#FBF8F3"
INK = "#493F48"
GRID = "#D8D0C6"
MUTED = "#817782"

# Pastel, high-contrast-enough colors.
LAVENDER = "#9C8CC4"
PERIWINKLE = "#91B7D8"
ROSE = "#D895A0"
CORAL = "#E7A68F"
MINT = "#9FC8B2"
SAGE = "#A8BE9A"
PEACH = "#E8BC93"
SKY = "#9FC7D9"

ALGO_COLORS = {
    "Q-learning": LAVENDER,
    "Monte Carlo": ROSE,
}
ALGO_LIGHT = {
    "Q-learning": "#DDD5EA",
    "Monte Carlo": "#F0D5D9",
}

METRIC_COLORS = {
    "Fraudster caught": CORAL,
    "Innocent caught": MINT,
    "Trickster caught": LAVENDER,
    "Timeout": PEACH,
}

# Computer Modern / CMR10 typography, matching the reference figure.
# CMR10 is compact, elegant, and gives the labels/title the same visual
# character as the supplied reference without requiring LaTeX.
plt.rcParams.update({
    "font.family": "cmr10",
    "font.size": 10.5,

    "axes.titlesize": 15.0,
    "axes.labelsize": 11.0,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "axes.formatter.use_mathtext": True,

    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,

    "axes.edgecolor": INK,
    "axes.linewidth": 0.75,

    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,
    "savefig.facecolor": CREAM,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,

    "legend.frameon": True,
    "legend.fontsize": 10.0,

    # Embed TrueType fonts in PDFs.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Data
# ============================================================

def load_data():
    path = RESULTS / "rl_learning_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find:\n{path}")

    df = pd.read_csv(path)
    required = {
        "algorithm", "train_episodes", "seed", "mean_reward",
        "fraudster_catch_rate", "innocent_catch_rate",
        "trickster_catch_rate", "timeout_rate",
        "mean_observations", "mean_accusations", "mean_votes",
        "mean_episode_length", "mean_remaining_budget", "q_states",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns:\n" + "\n".join(sorted(missing)))

    df["algorithm"] = df["algorithm"].astype(str).str.strip()
    df["train_episodes"] = pd.to_numeric(df["train_episodes"])
    return df


def pretty_algorithm(name):
    return {
        "q_learning": "Q-learning",
        "monte_carlo": "Monte Carlo",
    }.get(name, name.replace("_", " ").title())


def aggregate(df, metric):
    out = (
        df.groupby(["algorithm", "train_episodes"], as_index=False)[metric]
        .agg(mean="mean", std="std")
    )
    out["std"] = out["std"].fillna(0.0)
    out["algorithm_label"] = out["algorithm"].map(pretty_algorithm)
    return out.sort_values(["algorithm", "train_episodes"])

# Continuous pastel maps used for gradient lines, uncertainty bands,
# densities, and heatmaps.
PASTEL_MAPS = {
    "lavender": LinearSegmentedColormap.from_list(
        "pastel_lavender",
        ["#F4F0FF", "#CFC4F2", "#9C8CC4"],
    ),
    "rose": LinearSegmentedColormap.from_list(
        "pastel_rose",
        ["#FFF1F3", "#F1C5CC", "#D895A0"],
    ),
    "mint": LinearSegmentedColormap.from_list(
        "pastel_mint",
        ["#EEF9F6", "#C4E7DC", "#9FC8B2"],
    ),
    "peach": LinearSegmentedColormap.from_list(
        "pastel_peach",
        ["#FFF6EC", "#F4D5B5", "#E8BC93"],
    ),
}


def gradient_line(ax, x, y, cmap, linewidth=2.8, alpha=0.98, zorder=4):
    """Draw a line whose colour gently progresses through a pastel map."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return ax.plot(
            x, y,
            color=cmap(0.65),
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )[0]

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(0, max(len(segments) - 1, 1)),
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)
    return lc


def gradient_band(ax, x, lower, upper, cmap, alpha=0.16, zorder=1):
    """Draw a softly varying uncertainty band using the same pastel map."""
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    for i in range(len(x) - 1):
        ax.fill_between(
            x[i:i + 2],
            lower[i:i + 2],
            upper[i:i + 2],
            color=cmap((i + 0.5) / max(len(x) - 1, 1)),
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )


# ============================================================
# Figure helpers
# ============================================================

def finish_axis(ax, *, xgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.grid(axis="y", color=GRID, alpha=0.32, linewidth=0.7)
    ax.grid(axis="x", color=GRID, alpha=0.18, linewidth=0.6 if xgrid else 0)
    ax.tick_params(length=3, width=0.7)


def save_figure(fig, name):
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor=CREAM)
    fig.savefig(pdf, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)
    print(f"  ✓ {png.name}")
    print(f"  ✓ {pdf.name}")


def add_subtitle(ax, text):
    ax.text(
        0.0, 1.010, text,
        transform=ax.transAxes,
        fontsize=9.0,
        color=MUTED,
        va="bottom",
        ha="left",
        fontweight=400,
        linespacing=1.15,
    )


def style_legend(ax, loc="upper right", **kwargs):
    """Reference-style translucent rounded legend card."""
    legend = ax.legend(
        loc=loc,
        handlelength=2.25,
        handletextpad=0.75,
        borderpad=0.6,
        labelspacing=0.58,
        columnspacing=1.0,
        borderaxespad=0.55,
        frameon=True,
        fancybox=True,
        framealpha=0.66,
        facecolor="#FFFFFF",
        edgecolor="#D9D1CB",
        shadow=False,
        alignment="center",
        **kwargs,
    )

    if legend:
        for text in legend.get_texts():
            text.set_color(INK)
            text.set_fontsize(10.0)
            text.set_fontweight("normal")

        frame = legend.get_frame()
        frame.set_linewidth(0.65)
        frame.set_alpha(0.66)

    return legend

# ============================================================
# Core learning curves
# ============================================================

def plot_metric(df, metric, title, ylabel, filename, *, ylim=None, log_x=True):
    stats = aggregate(df, metric)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    styles = {
        "Q-learning": ("-", "o"),
        "Monte Carlo": ("--", "D"),
    }

    for algorithm in stats["algorithm_label"].unique():
        sub = stats[stats["algorithm_label"] == algorithm].sort_values("train_episodes")
        x = sub["train_episodes"].to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].to_numpy()
        color = ALGO_COLORS[algorithm]
        linestyle, marker = styles.get(algorithm, ("-", "o"))

        cmap = PASTEL_MAPS["lavender" if algorithm == "Q-learning" else "rose"]
        gradient_band(ax, x, mean - std, mean + std, cmap, alpha=0.18, zorder=1)
        gradient_line(ax, x, mean, cmap, linewidth=2.8, zorder=3)
        ax.plot(x, mean, color="none", linewidth=0, linestyle="none",
                marker=marker, markersize=5.8, markerfacecolor=CREAM,
                markeredgewidth=1.4, markeredgecolor=color,
                label=algorithm, zorder=5)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Training episodes", labelpad=9)
    ax.set_ylabel(ylabel, labelpad=9)
    ax.set_title(title, loc="left", pad=19)
    add_subtitle(ax, r"Mean across three independent seeds")
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_legend(ax, loc="upper right")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_reward(df):
    plot_metric(df, "mean_reward", "Investigator performance across training", "Mean evaluation reward", "rl_reward")


def plot_fraudster_detection(df):
    plot_metric(df, "fraudster_catch_rate", "Fraudster detection", "Fraudster catch rate", "rl_fraudster_detection", ylim=(0, 1))


def plot_false_positive_innocent(df):
    plot_metric(df, "innocent_catch_rate", "Innocent false positives", "Innocent catch rate", "rl_innocent_false_positive", ylim=(0, 1))


def plot_false_positive_trickster(df):
    plot_metric(df, "trickster_catch_rate", "Trickster false positives", "Trickster catch rate", "rl_trickster_false_positive", ylim=(0, 1))


def plot_timeout(df):
    plot_metric(df, "timeout_rate", "Timeout behaviour", "Timeout rate", "rl_timeout", ylim=(0, 1))


def plot_observations(df):
    plot_metric(df, "mean_observations", "Investigation behaviour", "Mean observations per episode", "rl_observations")


def plot_accusations(df):
    plot_metric(df, "mean_accusations", "Accusation behaviour", "Mean accusations per episode", "rl_accusations")


def plot_votes(df):
    plot_metric(df, "mean_votes", "Voting behaviour", "Mean votes per episode", "rl_votes")


def plot_episode_length(df):
    plot_metric(df, "mean_episode_length", "Episode efficiency", "Mean episode length", "rl_episode_length")


def plot_budget(df):
    plot_metric(df, "mean_remaining_budget", "Budget usage", "Mean remaining budget", "rl_remaining_budget")


def plot_q_states(df):
    plot_metric(df, "q_states", "Learned state-space coverage", "Number of learned states", "rl_q_states")

# ============================================================
# Detection overview
# ============================================================

def plot_detection_overview(df):
    metrics = [
        ("fraudster_catch_rate", "Fraudster caught", CORAL),
        ("innocent_catch_rate", "Innocent caught", MINT),
        ("trickster_catch_rate", "Trickster caught", LAVENDER),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    styles = {"Q-learning": ("-", "o"), "Monte Carlo": ("--", "D")}

    for ax, (metric, ylabel, metric_color) in zip(axes, metrics):
        stats = aggregate(df, metric)
        for algorithm in stats["algorithm_label"].unique():
            sub = stats[stats["algorithm_label"] == algorithm].sort_values("train_episodes")
            x = sub["train_episodes"].to_numpy()
            mean = sub["mean"].to_numpy()
            std = sub["std"].to_numpy()
            line_color = ALGO_COLORS[algorithm]
            linestyle, marker = styles[algorithm]
            cmap = PASTEL_MAPS["lavender" if algorithm == "Q-learning" else "rose"]
            gradient_band(ax, x, mean - std, mean + std, cmap, alpha=0.16)
            gradient_line(ax, x, mean, cmap, linewidth=2.3)
            ax.plot(x, mean, color="none", linewidth=0, linestyle="none",
                    marker=marker, markersize=5, markerfacecolor=CREAM,
                    markeredgewidth=1.2, markeredgecolor=line_color, label=algorithm)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel(ylabel)
        finish_axis(ax)

    style_legend(axes[0], loc="upper right")
    fig.suptitle("Detection improves while false-positive behaviour remains visible", x=0.03, ha="left", fontsize=16)
    fig.text(0.03, 0.90, r"Mean across three independent seeds", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(fig, "rl_detection_overview")

# ============================================================
# Heatmaps
# ============================================================

def heatmap_matrix(df, metric, algorithm=None):
    sub = df if algorithm is None else df[df["algorithm"] == algorithm]
    pivot = sub.groupby("train_episodes")[metric].mean().to_frame().T
    pivot.columns = [int(x) for x in pivot.columns]
    return pivot


def plot_training_heatmap(df, metric, title, filename, *, vmin=None, vmax=None, fmt=".2f", cmap_colors=None):
    pivot = df.groupby(["algorithm", "train_episodes"])[metric].mean().unstack()
    pivot.index = [pretty_algorithm(x) for x in pivot.index]
    pivot = pivot.sort_index()
    pivot = pivot[[c for c in sorted(pivot.columns)]]

    if cmap_colors is None:
        cmap_colors = ["#F8F1F4", ROSE, LAVENDER]
    cmap = LinearSegmentedColormap.from_list("decoy_pastel", cmap_colors)

    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(x):,}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Training episodes", labelpad=9)
    ax.set_title(title, loc="left", pad=19)
    add_subtitle(ax, "Cell value = mean across three independent seeds")

    threshold = (vmin + vmax) / 2 if vmin is not None and vmax is not None else np.nanmean(pivot.to_numpy())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            text_color = "white" if value >= threshold else INK
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=text_color, fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, aspect=25)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_reward_heatmap(df):
    plot_training_heatmap(df, "mean_reward", "Reward landscape across training", "rl_reward_heatmap", fmt=".2f", cmap_colors=["#F7F3FF", "#C9B8DE", LAVENDER])


def plot_detection_heatmap(df):
    plot_training_heatmap(df, "fraudster_catch_rate", "Fraudster detection landscape", "rl_detection_heatmap", vmin=0, vmax=1, fmt=".2f", cmap_colors=["#FFF5F1", "#F3C2B6", CORAL])

# ============================================================
# Seed distributions / density
# ============================================================

def kde_curve(values, grid):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.std(values) == 0:
        return np.zeros_like(grid)
    bandwidth = 1.06 * np.std(values, ddof=1) * len(values) ** (-1 / 5)
    bandwidth = max(bandwidth, (grid.max() - grid.min()) / 100)
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return density


def plot_reward_density(df):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    global_min = df["mean_reward"].min()
    global_max = df["mean_reward"].max()
    pad = 0.08 * max(global_max - global_min, 1.0)
    grid = np.linspace(global_min - pad, global_max + pad, 500)

    for algorithm in ["Q-learning", "Monte Carlo"]:
        values = df.loc[df["algorithm"].map(pretty_algorithm) == algorithm, "mean_reward"].to_numpy()
        density = kde_curve(values, grid)
        color = ALGO_COLORS[algorithm]
        cmap = PASTEL_MAPS["lavender" if algorithm == "Q-learning" else "rose"]
        ax.fill_between(grid, density, color=cmap(0.38), alpha=0.55)
        ax.plot(grid, density, color=color, linewidth=2.3, label=algorithm)

        # Show every underlying experimental observation as a rug.
        rug_y = np.full(len(values), -0.012 * max(density.max(), 1))
        ax.scatter(values, rug_y, s=24, color=color, alpha=0.55, zorder=3)

    ax.set_xlabel("Mean evaluation reward")
    ax.set_ylabel("Estimated density")
    ax.set_title("Reward distribution across seeds and training budgets", loc="left", pad=18)
    add_subtitle(ax, "Each density contains the 15 seed $\times$ training observations for one algorithm")
    finish_axis(ax)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    style_legend(ax, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "rl_reward_density")


def plot_reward_seed_strip(df):
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    algorithms = ["Q-learning", "Monte Carlo"]
    offsets = {"Q-learning": -0.14, "Monte Carlo": 0.14}
    x_positions = np.arange(5)
    episodes = sorted(df["train_episodes"].unique())

    for algorithm in algorithms:
        for i, episodes_n in enumerate(episodes):
            values = df.loc[
                (df["algorithm"].map(pretty_algorithm) == algorithm) &
                (df["train_episodes"] == episodes_n), "mean_reward"
            ].to_numpy()
            x = np.full(len(values), x_positions[i] + offsets[algorithm])
            jitter = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(x + jitter, values, s=52, color=ALGO_COLORS[algorithm],
                       edgecolor=CREAM, linewidth=0.8, alpha=0.9, label=algorithm if i == 0 else None, zorder=3)
            ax.plot([x_positions[i] + offsets[algorithm]] * 2,
                    [values.min(), values.max()], color=ALGO_COLORS[algorithm], alpha=0.35, linewidth=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(x):,}" for x in episodes])
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Seed-level reward variability", loc="left", pad=18)
    add_subtitle(ax, "Individual points show the three independent seeds at each training budget")
    finish_axis(ax)
    style_legend(ax, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "rl_reward_seed_variability")

# ============================================================
# Behavioural fingerprint heatmap
# ============================================================

def plot_behavior_heatmap(df):
    metrics = {
        "Observations": "mean_observations",
        "Accusations": "mean_accusations",
        "Votes": "mean_votes",
        "Episode length": "mean_episode_length",
        "Remaining budget": "mean_remaining_budget",
    }
    rows = []
    for algorithm in ["Q-learning", "Monte Carlo"]:
        sub = df[df["algorithm"].map(pretty_algorithm) == algorithm]
        for metric_name, col in metrics.items():
            for episodes_n in sorted(df["train_episodes"].unique()):
                value = sub.loc[sub["train_episodes"] == episodes_n, col].mean()
                rows.append({"algorithm": algorithm, "metric": metric_name, "episodes": episodes_n, "value": value})

    long = pd.DataFrame(rows)
    # Normalize each behavioural metric independently so colors encode relative change.
    long["normalized"] = long.groupby("metric")["value"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    )
    pivot = long.pivot_table(index=["algorithm", "metric"], columns="episodes", values="normalized")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    cmap = LinearSegmentedColormap.from_list("behavior", ["#F7EEF5", "#C7DCE0", "#9C8CC4"])
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(x):,}" for x in pivot.columns])
    ax.set_xlabel("Training episodes")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{alg. replace('-', '-')} $\cdot$ {metric}" for alg, metric in pivot.index])
    ax.set_title("Behavioural fingerprint of the learned policies", loc="left", pad=18)
    add_subtitle(ax, "Each row is normalized independently; darker cells indicate higher relative values")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, aspect=28)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, "rl_behavior_heatmap")

# ============================================================
# Correlation matrix
# ============================================================

def plot_correlation_heatmap(df):
    cols = [
        "mean_reward", "fraudster_catch_rate", "innocent_catch_rate",
        "trickster_catch_rate", "timeout_rate", "mean_observations",
        "mean_accusations", "mean_votes", "mean_episode_length",
        "mean_remaining_budget", "q_states",
    ]
    labels = [
        "Reward", "Fraudster catch", "Innocent FP", "Trickster FP",
        "Timeout", "Observations", "Accusations", "Votes",
        "Episode length", "Remaining budget", "Q states",
    ]
    corr = df[cols].corr()
    corr.index = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    cmap = LinearSegmentedColormap.from_list("corr", ["#DCA0AA", "#FBF8F3", "#9CB9D2"])
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Relationships among learned-performance metrics", loc="left", pad=18)
    add_subtitle(ax, "Pearson correlations across the 30 algorithm $\times$ training $\times$ seed observations")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.55 else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.8, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, aspect=28)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, "rl_metric_correlation")

# ============================================================
# Reward vs detection
# ============================================================

def plot_reward_vs_detection(df):
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    marker_sizes = {10_000: 38, 50_000: 52, 100_000: 68, 250_000: 88, 500_000: 112}
    markers = {"Q-learning": "o", "Monte Carlo": "D"}

    for algorithm in ["Q-learning", "Monte Carlo"]:
        sub = df[df["algorithm"].map(pretty_algorithm) == algorithm]
        for _, row in sub.iterrows():
            episodes_n = int(row["train_episodes"])
            ax.scatter(
                row["fraudster_catch_rate"], row["mean_reward"],
                s=marker_sizes[episodes_n], marker=markers[algorithm],
                color=ALGO_COLORS[algorithm], alpha=0.78,
                edgecolor=CREAM, linewidth=0.9,
            )
        ax.scatter([], [], s=70, marker=markers[algorithm], color=ALGO_COLORS[algorithm], label=algorithm)

    ax.set_xlabel("Fraudster catch rate")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Reward and fraudster detection are coupled", loc="left", pad=18)
    add_subtitle(ax, "Point size increases with training budget; each point is one seed-level observation")
    finish_axis(ax)
    style_legend(ax, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "rl_reward_vs_detection")

# ============================================================
# Main
# ============================================================

def main():
    print("\nLoading RL results...")
    df = load_data()
    print(f"Rows: {len(df)}")
    print("Algorithms:", sorted(df["algorithm"].unique()))
    print("Training points:", sorted(df["train_episodes"].unique()))
    print("Seeds:", sorted(df["seed"].unique()))

    print("\nGenerating figures...")
    plot_reward(df)
    plot_fraudster_detection(df)
    plot_false_positive_innocent(df)
    plot_false_positive_trickster(df)
    plot_timeout(df)
    plot_observations(df)
    plot_accusations(df)
    plot_votes(df)
    plot_episode_length(df)
    plot_budget(df)
    plot_q_states(df)
    plot_detection_overview(df)

    # New analytical figures.
    plot_reward_heatmap(df)
    plot_detection_heatmap(df)
    plot_reward_density(df)
    plot_reward_seed_strip(df)
    plot_behavior_heatmap(df)
    plot_correlation_heatmap(df)
    plot_reward_vs_detection(df)

    print("\nDone.")
    print(f"Figures saved to:\n{FIGURES}")


if __name__ == "__main__":
    main()