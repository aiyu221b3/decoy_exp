from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).resolve().parents[2]

RESULTS = BASE / "results"
OUT = BASE / "figures" / "multi"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def find_result(candidates):
    """
    Locate a result CSV under the project.

    Searches:
        1. results/
        2. recursively under results/
        3. recursively under the whole project
    """

    # Direct results directory
    for name in candidates:

        path = RESULTS / name

        if path.exists():
            return path

    # Recursive results search
    for name in candidates:

        matches = list(
            RESULTS.rglob(name)
        )

        if matches:
            return matches[0]

    # Final fallback: whole project
    for name in candidates:

        matches = list(
            BASE.rglob(name)
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        "\nCould not locate any of:\n"
        + "\n".join(
            f"  - {name}"
            for name in candidates
        )
        + "\n\nProject searched:\n"
        + f"  {BASE}"
    )


LEARNING = find_result([
    "multi_learning_curve.csv",
    "multi_learning.csv",
    "joint_learning_curve.csv",
])

SUMMARY = find_result([
    "multi_summary.csv",
    "multi_summary(1).csv",
    "joint_summary.csv",
])

REGISTER = find_result([
    "trickster_register.csv",
    "trickster_register(1).csv",
    "multi_trickster_register.csv",
])


print("=" * 72)
print("MULTI-AGENT PLOT GENERATION")
print("=" * 72)

print("\nLocated files:")
print(f"  learning : {LEARNING}")
print(f"  summary  : {SUMMARY}")
print(f"  register : {REGISTER}")

df = pd.read_csv(LEARNING)
summary = pd.read_csv(SUMMARY)
register = pd.read_csv(REGISTER)

print("\nLoaded:")
print(f"  learning rows : {len(df):,}")
print(f"  summary rows  : {len(summary):,}")
print(f"  register rows : {len(register):,}")


CREAM = "#FBF8F3"
WHITE = "#FFFFFF"

INK = "#463F4A"
MUTED = "#8A818B"
GRID = "#E9E3DE"

PINK = "#DFA0AE"
LAV = "#9A86D4"
BLUE = "#8FA9C6"
MINT = "#9BC8BA"

PINK_LIGHT = "#F6DDE3"
LAV_LIGHT = "#E7E1F6"

ROSE_MAP = LinearSegmentedColormap.from_list(
    "rose_gradient",
    [
        PINK_LIGHT,
        PINK,
        "#C96E88",
    ],
)

LAV_MAP = LinearSegmentedColormap.from_list(
    "lav_gradient",
    [
        LAV_LIGHT,
        LAV,
        "#7660B7",
    ],
)


def format_budget_axis(ax, budgets):
    """Format joint-training budgets as clean linear x-axis labels."""
    budgets = sorted(budgets)

    ax.set_xscale("linear")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{int(b / 1000)}k" for b in budgets])

    ax.set_xlabel("Training episodes")

    ax.tick_params(
        axis="x",
        which="both",
        length=4,
        pad=8,
    )
plt.rcParams.update({
    "font.family": "cmr10",

    "mathtext.fontset": "cm",
    "mathtext.default": "regular",

    "font.size": 10.5,

    "axes.titlesize": 13.5,
    "axes.labelsize": 10.8,

    "axes.titleweight": "normal",
    "axes.labelweight": "normal",

    "axes.titlecolor": INK,
    "axes.labelcolor": INK,

    "text.color": INK,

    "xtick.color": INK,
    "ytick.color": INK,

    "axes.edgecolor": INK,

    "figure.facecolor": CREAM,
    "axes.facecolor": CREAM,

    "savefig.facecolor": CREAM,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.unicode_minus": False,
})


def finish_axis(ax):
    """
    Apply consistent journal-style axis formatting.
    """

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.75,
        alpha=0.78,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)

    ax.tick_params(
        length=3,
        width=0.7,
        labelsize=9.5,
    )


def add_title(ax, title, subtitle):
    """
    Add title + subtitle directly above an axes.

    Using axes coordinates keeps the text attached to the
    plot instead of letting tight/constrained layout move it.
    """

    ax.set_title(
        title,
        loc="left",
        pad=25,
        fontsize=14.5,
        color=INK,
    )

    ax.text(
        0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color=MUTED,
        clip_on=False,
    )


def add_figure_header(fig, title, subtitle, x=0.075):
    """
    Header for multi-panel figures.
    """

    fig.text(
        x,
        0.925,
        title,
        ha="left",
        va="center",
        fontsize=15,
        color=INK,
    )

    fig.text(
        x,
        0.885,
        subtitle,
        ha="left",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )


def add_legend(
    ax,
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=None,
):
    """
    Soft rounded translucent legend.

    No shadow. No heavy border.
    """

    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,

        frameon=True,
        fancybox=True,

        framealpha=0.78,
        facecolor=WHITE,
        edgecolor="#DDD6D0",

        borderpad=0.7,
        labelspacing=0.55,

        handlelength=2.0,
        handletextpad=0.7,

        fontsize=9.5,

        borderaxespad=0.7,
    )

    # Very subtle rounded border.
    legend.get_frame().set_linewidth(0.7)

    return legend


def save_figure(fig, filename):
    """
    Save PNG and PDF.
    """

    png = OUT / f"{filename}.png"
    pdf = OUT / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    plt.close(fig)

    print(f"  ✓ {png.name}")
    print(f"  ✓ {pdf.name}")


def set_training_ticks(ax, values):
    """
    Use only the actual training budgets as labelled major ticks.
    Minor log ticks are disabled so labels cannot overlap or collide
    with intermediate tick labels.
    """

    values = np.array(
        sorted(values),
        dtype=float,
    )

    ax.set_xticks(values)

    labels = []

    for value in values:

        if value >= 1_000_000:
            labels.append(
                f"{value / 1_000_000:g}M"
            )

        elif value >= 1_000:
            labels.append(
                f"{value / 1_000:g}k"
            )

        else:
            labels.append(
                str(int(value))
            )

    ax.set_xticklabels(
        labels,
        fontsize=9.2,
    )

    # No intermediate log labels/ticks.
    ax.xaxis.set_minor_locator(
        plt.NullLocator()
    )
    ax.tick_params(
        axis="x",
        which="minor",
        bottom=False,
        top=False,
        labelbottom=False,
    )


def gradient_color(value, low, high):
    """
    Interpolate between lavender and rose.
    """

    if high == low:
        frac = 0.5

    else:
        frac = (
            np.log10(value) - np.log10(low)
        ) / (
            np.log10(high) - np.log10(low)
        )

    frac = np.clip(
        frac,
        0,
        1,
    )

    lavender = np.array(
        mcolors.to_rgb(LAV)
    )

    pink = np.array(
        mcolors.to_rgb(PINK)
    )

    rgb = (
        lavender * (1 - frac)
        +
        pink * frac
    )

    return rgb



print("\n[1/10] Joint-training learning trajectories")

window = 5000

curves = []

for (
    train_eps,
    seed,
), group in df.groupby(
    [
        "train_episodes",
        "seed",
    ],
    sort=True,
):

    group = group.sort_values(
        "episode"
    ).copy()

    group["rolling_reward"] = (
        group["inv_rew"]
        .rolling(
            window,
            min_periods=window // 4,
        )
        .mean()
    )

    curves.append(
        group[
            [
                "train_episodes",
                "seed",
                "episode",
                "rolling_reward",
            ]
        ]
    )

curve = pd.concat(
    curves,
    ignore_index=True,
)

train_levels = sorted(
    curve["train_episodes"].unique()
)

fig, ax = plt.subplots(
    figsize=(9.4, 5.4),
)

# Explicit spacing.
fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.16,
    top=0.82,
)

for train_eps in train_levels:

    sub = curve[
        curve["train_episodes"] == train_eps
    ]

    # Faint individual seeds.
    for seed in sorted(
        sub["seed"].unique()
    ):

        seed_data = sub[
            sub["seed"] == seed
        ]

        ax.plot(
            seed_data["episode"],
            seed_data["rolling_reward"],
            color=LAV,
            alpha=0.12,
            linewidth=0.8,
        )

    pivot = sub.pivot(
        index="episode",
        columns="seed",
        values="rolling_reward",
    )

    mean = pivot.mean(
        axis=1
    )

    std = pivot.std(
        axis=1
    )

    x_full = mean.index.to_numpy()

    stride = max(
        1,
        len(x_full) // 1200,
    )

    x = x_full[::stride]
    m = mean.to_numpy()[::stride]
    s = std.to_numpy()[::stride]

    color = gradient_color(
        train_eps,
        min(train_levels),
        max(train_levels),
    )

    ax.plot(
        x,
        m,
        color=color,
        linewidth=1.9,
    )

    ax.fill_between(
        x,
        m - s,
        m + s,
        color=color,
        alpha=0.10,
    )


add_title(
    ax,
    "Joint training trajectories",
    r"Rolling investigator reward",
)

ax.set_xlabel(
    "Episode within training run"
)

ax.set_ylabel(
    "Rolling investigator reward"
)

finish_axis(ax)

handles = [
    Line2D(
        [0],
        [0],
        color=gradient_color(
            value,
            min(train_levels),
            max(train_levels),
        ),
        lw=2,
        label=(
            f"{int(value / 1000)}k"
            if value >= 1000
            else str(int(value))
        ),
    )
    for value in train_levels
]

add_legend(
    ax,
    handles,
    [
        h.get_label()
        for h in handles
    ],
)

save_figure(
    fig,
    "multi_learning_curve",
)



print("\n[2/10] Detection overview")

metrics = [
    (
        "fraudster_catch_rate",
        "Fraudster caught",
        PINK,
    ),
    (
        "innocent_catch_rate",
        "Innocent caught",
        LAV,
    ),
    (
        "trickster_catch_rate",
        "Trickster caught",
        BLUE,
    ),
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12.6, 4.3),
    sharex=True,
    sharey=True,
)


fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.20,
    top=0.74,
    wspace=0.035,
)

for ax, (
    metric,
    label,
    color,
) in zip(
    axes,
    metrics,
):

    stats = (
        summary
        .groupby("train_episodes")[metric]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    x = stats[
        "train_episodes"
    ].to_numpy()

    mean = stats[
        "mean"
    ].to_numpy()

    std = stats[
        "std"
    ].to_numpy()

    ax.plot(
        x,
        mean,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=5.2,
        markerfacecolor=CREAM,
        markeredgewidth=1.5,
        markeredgecolor=color,
    )

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.16,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_title(
        label,
        fontsize=13.5,
        pad=11,
        color=INK,
    )

    ax.set_xlabel(
        "Training episodes"
    )

    set_training_ticks(
        ax,
        x,
    )

    finish_axis(ax)


axes[0].set_ylabel(
    "Mean rate"
)

add_figure_header(
    fig,
    "Detection profile across joint-training budgets",
    r"Three independent seeds per training budget",
)

save_figure(
    fig,
    "multi_detection_overview",
)

print("\n[3/10] Episode outcome composition")

outcome_cols = [
    (
        "fraudster_catch_rate",
        "Fraudster",
        PINK,
    ),
    (
        "innocent_catch_rate",
        "Innocent",
        LAV,
    ),
    (
        "trickster_catch_rate",
        "Trickster",
        BLUE,
    ),
    (
        "timeout_rate",
        "Timeout",
        "#CFC8D2",
    ),
]

grouped = (
    summary
    .groupby("train_episodes")[
        [
            x[0]
            for x in outcome_cols
        ]
    ]
    .mean()
)

fig, ax = plt.subplots(
    figsize=(9.2, 5.3),
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    bottom=0.17,
    top=0.80,
)

x = np.arange(
    len(grouped)
)

bottom = np.zeros(
    len(grouped)
)

for (
    column,
    label,
    color,
) in outcome_cols:

    values = grouped[
        column
    ].to_numpy()

    ax.bar(
        x,
        values,
        bottom=bottom,
        width=0.66,
        color=color,
        alpha=0.86,
        edgecolor=CREAM,
        linewidth=0.8,
        label=label,
    )

    bottom += values


ax.set_xticks(
    x,
    [
        f"{int(value / 1000)}k"
        if value >= 1000
        else str(int(value))
        for value in grouped.index
    ],
)

ax.set_xlabel(
    "Training episodes"
)

ax.set_ylabel(
    "Mean episode outcome share"
)

ax.set_ylim(
    0,
    1,
)

add_title(
    ax,
    "Episode outcomes",
    "Mean across three independent seeds; categories sum to one",
)

finish_axis(ax)

ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=True,
    framealpha=0.78,
    facecolor=WHITE,
    edgecolor="#DDD6D0",
    borderpad=0.7,
    labelspacing=0.55,
    handlelength=1.7,
    handletextpad=0.7,
    fontsize=9.5,
)

save_figure(
    fig,
    "multi_outcome_composition",
)


print("\n[4/10] Behavioural fingerprint")

behavior = [
    (
        "mean_observations",
        "Observations",
        LAV,
    ),
    (
        "mean_accusations",
        "Accusations",
        PINK,
    ),
    (
        "mean_votes",
        "Votes",
        BLUE,
    ),
    (
        "mean_episode_length",
        "Episode length",
        MINT,
    ),
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.4, 7.0),
    sharex=True,
)

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    bottom=0.12,
    top=0.76,
    hspace=0.54,
    wspace=0.23,
)

for ax, (
    metric,
    label,
    color,
) in zip(
    axes.ravel(),
    behavior,
):

    stats = (
        summary
        .groupby("train_episodes")[metric]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    x = stats[
        "train_episodes"
    ].to_numpy()

    mean = stats[
        "mean"
    ].to_numpy()

    std = stats[
        "std"
    ].to_numpy()

    ax.plot(
        x,
        mean,
        color=color,
        linewidth=2.3,
        marker="o",
        markersize=4.8,
        markerfacecolor=CREAM,
        markeredgecolor=color,
        markeredgewidth=1.4,
    )

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.15,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_title(
        label,
        loc="left",
        pad=10,
        fontsize=12.5,
    )

    ax.set_xlabel(
        "Training episodes"
    )

    set_training_ticks(
        ax,
        x,
    )

    finish_axis(ax)


add_figure_header(
    fig,
    "Behavioural fingerprint of joint training",
    r"Mean across three independent seeds",
    x=0.085,
)

save_figure(
    fig,
    "multi_behaviour_overview",
)



print("\n[5/10] Behaviour heatmap")

heat_metrics = [
    (
        "mean_observations",
        "Observations",
    ),
    (
        "mean_accusations",
        "Accusations",
    ),
    (
        "mean_votes",
        "Votes",
    ),
    (
        "mean_episode_length",
        "Episode length",
    ),
    (
        "mean_remaining_budget",
        "Remaining budget",
    ),
    (
        "mean_focuses",
        "Focuses",
    ),
]

heat = (
    summary
    .groupby("train_episodes")[
        [
            metric[0]
            for metric in heat_metrics
        ]
    ]
    .mean()
    .T
)

heat.index = [
    metric[1]
    for metric in heat_metrics
]

normalized = heat.copy()

for index in normalized.index:

    low = normalized.loc[
        index
    ].min()

    high = normalized.loc[
        index
    ].max()

    if high == low:

        normalized.loc[
            index
        ] = 0.5

    else:

        normalized.loc[
            index
        ] = (
            normalized.loc[index] - low
        ) / (
            high - low
        )


fig, ax = plt.subplots(
    figsize=(10.0, 5.8),
)

fig.subplots_adjust(
    left=0.23,
    right=0.90,
    bottom=0.15,
    top=0.80,
)

image = ax.imshow(
    normalized.to_numpy(),
    aspect="auto",
    cmap=LAV_MAP,
    vmin=0,
    vmax=1,
)

ax.set_yticks(
    np.arange(
        len(normalized.index)
    ),
    normalized.index,
)

ax.set_xticks(
    np.arange(
        len(normalized.columns)
    ),
    [
        f"{int(value / 1000)}k"
        if value >= 1000
        else str(int(value))
        for value in normalized.columns
    ],
)

ax.set_xlabel(
    "Training episodes"
)

add_title(
    ax,
    "Behavioural fingerprint",
    "Each row is normalized independently; darker cells indicate relatively larger values",
)

for i in range(
    normalized.shape[0]
):

    for j in range(
        normalized.shape[1]
    ):

        ax.text(
            j,
            i,
            f"{heat.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=INK,
        )


ax.spines[:].set_visible(
    False
)

cbar = fig.colorbar(
    image,
    ax=ax,
    pad=0.025,
    aspect=28,
)

cbar.set_label(
    "Relative value within metric",
    fontsize=9.5,
)

cbar.outline.set_visible(
    False
)

save_figure(
    fig,
    "multi_behaviour_heatmap",
)



print("\n[6/10] Reward density")

sample_parts = []

for (
    train_eps,
    group,
) in df.groupby(
    "train_episodes"
):

    n = min(
        25000,
        len(group),
    )

    sample_parts.append(
        group.sample(
            n=n,
            random_state=int(train_eps),
        )
    )

sample = pd.concat(
    sample_parts,
    ignore_index=True,
)

density_colors = [
    LAV,
    "#B29AD0",
    "#C89BB7",
    PINK,
]

fig, ax = plt.subplots(
    figsize=(9.4, 5.5),
)

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.16,
    top=0.80,
)

for (
    train_eps,
    color,
) in zip(
    sorted(
        sample["train_episodes"].unique()
    ),
    density_colors,
):

    values = sample.loc[
        sample["train_episodes"] == train_eps,
        "inv_rew",
    ].to_numpy()

    counts, edges = np.histogram(
        values,
        bins=70,
        density=True,
    )

    centers = (
        edges[:-1]
        +
        edges[1:]
    ) / 2

    ax.plot(
        centers,
        counts,
        color=color,
        linewidth=2.0,
    )

    ax.fill_between(
        centers,
        counts,
        color=color,
        alpha=0.09,
    )


add_title(
    ax,
    "Reward distribution shifts with training",
    "Episode-level investigator rewards; density from 25k sampled episodes per budget",
)

ax.set_xlabel(
    "Investigator episode reward"
)

ax.set_ylabel(
    "Empirical density"
)

finish_axis(ax)

handles = [
    Line2D(
        [0],
        [0],
        color=color,
        lw=2,
        label=(
            f"{int(episodes / 1000)}k"
            if episodes >= 1000
            else str(int(episodes))
        ),
    )
    for episodes, color in zip(
        sorted(
            sample[
                "train_episodes"
            ].unique()
        ),
        density_colors,
    )
]

add_legend(
    ax,
    handles,
    [
        h.get_label()
        for h in handles
    ],
    loc="upper right",
)

save_figure(
    fig,
    "multi_reward_density",
)


print("\n[7/10] Seed variability")

fig, ax = plt.subplots(
    figsize=(9.2, 5.3),
)

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.16,
    top=0.80,
)

seed_colors = [
    PINK,
    LAV,
    BLUE,
]

seed_markers = [
    "o",
    "s",
    "D",
]

for (
    seed,
    color,
    marker,
) in zip(
    sorted(
        summary["seed"].unique()
    ),
    seed_colors,
    seed_markers,
):

    sub = (
        summary[
            summary["seed"] == seed
        ]
        .sort_values(
            "train_episodes"
        )
    )

    ax.plot(
        sub["train_episodes"],
        sub["inv_rew"],
        color=color,
        linewidth=1.8,
        marker=marker,
        markersize=5,
        markerfacecolor=CREAM,
        markeredgewidth=1.3,
        markeredgecolor=color,
    )


ax.set_xscale(
    "log"
)

set_training_ticks(
    ax,
    summary["train_episodes"].unique(),
)

add_title(
    ax,
    "Seed variability in investigator performance",
    "Each line is one independent training seed",
)

ax.set_xlabel(
    "Training episodes"
)

ax.set_ylabel(
    "Mean investigator reward"
)

finish_axis(ax)

handles = [
    Line2D(
        [0],
        [0],
        color=color,
        lw=2,
        marker=marker,
        label=f"Seed {seed}",
    )
    for seed, color, marker in zip(
        sorted(
            summary["seed"].unique()
        ),
        seed_colors,
        seed_markers,
    )
]

add_legend(
    ax,
    handles,
    [
        h.get_label()
        for h in handles
    ],
)

save_figure(
    fig,
    "multi_reward_seed_variability",
)

print("\n[8/10] Trickster win-register signal")

register_summary = (
    register
    .groupby("train_episodes")
    .agg({
        "investigator_all": [
            "mean",
            "std",
        ],
        "investigator_recent": [
            "mean",
            "std",
        ],
        "prediction_confidence": [
            "mean",
            "std",
        ],
    })
    .reset_index()
)

x = register_summary[
    "train_episodes"
].to_numpy()

fig, ax = plt.subplots(
    figsize=(9.2, 5.5),
)

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.16,
    top=0.80,
)

for (
    column,
    label,
    color,
) in [
    (
        "investigator_all",
        "Investigator, all history",
        LAV,
    ),
    (
        "investigator_recent",
        "Investigator, recent window",
        PINK,
    ),
]:

    mean = register_summary[
        (
            column,
            "mean",
        )
    ].to_numpy()

    std = register_summary[
        (
            column,
            "std",
        )
    ].to_numpy()

    ax.plot(
        x,
        mean,
        color=color,
        linewidth=2.3,
        marker="o",
        markersize=5,
    )

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.13,
    )


confidence = register_summary[
    (
        "prediction_confidence",
        "mean",
    )
].to_numpy()

confidence_std = register_summary[
    (
        "prediction_confidence",
        "std",
    )
].to_numpy()

ax.plot(
    x,
    confidence,
    color=BLUE,
    linewidth=2.1,
    marker="D",
    markersize=4.7,
)

ax.fill_between(
    x,
    confidence - confidence_std,
    confidence + confidence_std,
    color=BLUE,
    alpha=0.10,
)

ax.set_xscale(
    "log"
)

ax.set_ylim(
    0.35,
    0.85,
)

set_training_ticks(
    ax,
    x,
)

add_title(
    ax,
    "Trickster win-register signal",
    r"All-history and rolling-window estimates",
)

ax.set_xlabel(
    "Training episodes"
)

ax.set_ylabel(
    "Register probability / confidence"
)

finish_axis(ax)

handles = [
    Line2D(
        [0],
        [0],
        color=LAV,
        lw=2,
    ),
    Line2D(
        [0],
        [0],
        color=PINK,
        lw=2,
    ),
    Line2D(
        [0],
        [0],
        color=BLUE,
        lw=2,
    ),
]

add_legend(
    ax,
    handles,
    [
        "Investigator, all history",
        "Investigator, recent window",
        "Prediction confidence",
    ],
)

save_figure(
    fig,
    "multi_trickster_register",
)



print("\n[9/10] Reward vs fraudster detection")

fig, ax = plt.subplots(
    figsize=(8.2, 6.0),
)

fig.subplots_adjust(
    left=0.12,
    right=0.985,
    bottom=0.15,
    top=0.80,
)

minimum = summary[
    "train_episodes"
].min()

maximum = summary[
    "train_episodes"
].max()

for _, row in summary.iterrows():

    color = gradient_color(
        row["train_episodes"],
        minimum,
        maximum,
    )

    ax.scatter(
        row[
            "fraudster_catch_rate"
        ],
        row[
            "inv_rew"
        ],
        s=60,
        facecolor=color,
        edgecolor=INK,
        linewidth=0.6,
        alpha=0.85,
    )


means = (
    summary
    .groupby("train_episodes")[
        [
            "fraudster_catch_rate",
            "inv_rew",
        ]
    ]
    .mean()
)

for (
    train_eps,
    row,
) in means.iterrows():

    ax.annotate(
        (
            f"{int(train_eps / 1000)}k"
            if train_eps >= 1000
            else str(int(train_eps))
        ),
        (
            row[
                "fraudster_catch_rate"
            ],
            row[
                "inv_rew"
            ],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
    )


ax.axhline(
    0,
    color=GRID,
    linewidth=1,
)

add_title(
    ax,
    "Performance and fraudster detection",
    "Each point is one seed; labels mark the mean location of each training budget",
)

ax.set_xlabel(
    "Fraudster catch rate"
)

ax.set_ylabel(
    "Mean investigator reward"
)

finish_axis(ax)

save_figure(
    fig,
    "multi_reward_vs_detection",
)


print("\n[10/10] Joint-training summary heatmap")

final_metrics = [
    (
        "inv_rew",
        "Investigator reward",
    ),
    (
        "fraudster_catch_rate",
        "Fraudster caught",
    ),
    (
        "innocent_catch_rate",
        "Innocent caught",
    ),
    (
        "trickster_catch_rate",
        "Trickster caught",
    ),
    (
        "timeout_rate",
        "Timeout",
    ),
    (
        "mean_accusations",
        "Accusations",
    ),
    (
        "mean_episode_length",
        "Episode length",
    ),
    (
        "mean_remaining_budget",
        "Remaining budget",
    ),
]

final = (
    summary
    .groupby("train_episodes")[
        [
            metric[0]
            for metric in final_metrics
        ]
    ]
    .mean()
    .T
)

final.index = [
    metric[1]
    for metric in final_metrics
]

normalized_final = final.copy()

for index in normalized_final.index:

    low = normalized_final.loc[
        index
    ].min()

    high = normalized_final.loc[
        index
    ].max()

    if high == low:

        normalized_final.loc[
            index
        ] = 0.5

    else:

        normalized_final.loc[
            index
        ] = (
            normalized_final.loc[index] - low
        ) / (
            high - low
        )


fig, ax = plt.subplots(
    figsize=(10.2, 6.5),
)

fig.subplots_adjust(
    left=0.235,
    right=0.90,
    bottom=0.14,
    top=0.80,
)

image = ax.imshow(
    normalized_final.to_numpy(),
    aspect="auto",
    cmap=ROSE_MAP,
    vmin=0,
    vmax=1,
)

ax.set_yticks(
    np.arange(
        len(normalized_final.index)
    ),
    normalized_final.index,
)

ax.set_xticks(
    np.arange(
        len(normalized_final.columns)
    ),
    [
        (
            f"{int(value / 1000)}k"
            if value >= 1000
            else str(int(value))
        )
        for value in normalized_final.columns
    ],
)

ax.set_xlabel(
    "Training episodes"
)

add_title(
    ax,
    "Joint-training summary",
    "Rows normalized independently to emphasize how each metric changes with training",
)

for i in range(
    normalized_final.shape[0]
):

    for j in range(
        normalized_final.shape[1]
    ):

        ax.text(
            j,
            i,
            f"{final.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8.7,
            color=INK,
        )


ax.spines[:].set_visible(
    False
)

cbar = fig.colorbar(
    image,
    ax=ax,
    pad=0.025,
    aspect=28,
)

cbar.set_label(
    "Relative value within metric",
    fontsize=9.5,
)

cbar.outline.set_visible(
    False
)

save_figure(
    fig,
    "multi_summary_heatmap",
)



print("\nCreating archive...")

manifest = OUT / "README.txt"

manifest.write_text(
    "Multi-agent figures generated from:\n\n"
    f"- {LEARNING}\n"
    f"- {SUMMARY}\n"
    f"- {REGISTER}\n\n"
    "Each figure is provided as both PNG and PDF.\n",
    encoding="utf-8",
)


zip_path = BASE / "multi_figures.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED,
) as archive:

    for file in sorted(
        OUT.iterdir()
    ):

        if file.is_file():

            archive.write(
                file,
                arcname=(
                    f"multi_figures/{file.name}"
                ),
            )


print()
print("=" * 72)
print("DONE")
print("=" * 72)

print(
    f"\nFigures saved to:\n{OUT}"
)

print(
    f"\nZIP archive:\n{zip_path}"
)

print(
    "\nGenerated figures:"
)

for file in sorted(
    OUT.glob("*.png")
):

    print(
        f"  ✓ {file.name}"
    )

print(
    "\nNo training was performed."
)
print(
    "This script only reads the existing CSV results and generates figures."
)
