import numpy as np
import matplotlib.pyplot as plt

from decoy.ou_process import OUProcess


# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "CMU Serif",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.edgecolor": "#4A4036",
    "axes.labelcolor": "#3D352D",
    "xtick.color": "#4A4036",
    "ytick.color": "#4A4036",
    "text.color": "#3D352D",
    "legend.frameon": False,
    "figure.facecolor": "#FBF8F3",
    "axes.facecolor": "#FBF8F3",
    "grid.color": "#D8D0C5",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def main():
    rng = np.random.default_rng(3)

    theta_values = [0.1, 0.3, 0.8]

    mu = 0.0
    sigma = 1.0
    dt = 0.01

    x0 = 5.0
    n_steps = 5000

    time = np.arange(n_steps + 1) * dt

    pastel_colors = [
        "#9BB7D4",  # dusty blue
        "#B8C9A8",  # sage
        "#D9A6A6",  # dusty rose
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for theta, color in zip(theta_values, pastel_colors):
        process = OUProcess(
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            rng=rng,
        )

        trajectory = process.simulate(
            x0=x0,
            n_steps=n_steps,
        )

        stationary_std = sigma / np.sqrt(2 * theta)

        ax.plot(
            time,
            trajectory,
            color=color,
            linewidth=1.5,
            label=rf"$\theta={theta}$",
        )

        ax.axhspan(
            mu - stationary_std,
            mu + stationary_std,
            color=color,
            alpha=0.08,
        )

    ax.axhline(
        mu,
        color="#4A4036",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        label=r"$\mu$",
    )

    ax.set_title(
        "Ornstein–Uhlenbeck Dynamics",
        pad=14,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$X(t)$")

    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()