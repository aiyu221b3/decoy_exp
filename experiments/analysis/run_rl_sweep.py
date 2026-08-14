from pathlib import Path
import csv

from decoy.training import (
    eval_inv,
    train_mc_investigator,
    train_q_investigator,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

OUT.mkdir(exist_ok=True)

TRAIN_EPISODES = (
    10_000,
    50_000,
    100_000,
    250_000,
    500_000,
)

EVAL_EPISODES = 2_000
SEEDS = (11, 42, 99)

KEYS = (
    "mean_reward",
    "fraudster_catch_rate",
    "innocent_catch_rate",
    "trickster_catch_rate",
    "timeout_rate",
    "mean_observations",
    "mean_accusations",
    "mean_votes",
    "mean_episode_length",
    "mean_remaining_budget",
    "q_states",
)


def run_q(n_train, seed):
    agent, _, _ = train_q_investigator(
        n_episodes=n_train,
        seed=seed,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        max_steps=20,
        budget=10,
    )

    _, summary = eval_inv(
        agent,
        n_eps=EVAL_EPISODES,
        seed=10_000 + seed,
        max_steps=20,
        budget=10,
    )

    return summary


def run_mc(n_train, seed):
    agent, _, _ = train_mc_investigator(
        n_episodes=n_train,
        seed=seed,
        gamma=0.99,
        epsilon=0.1,
        max_steps=20,
        budget=10,
    )

    _, summary = eval_inv(
        agent,
        n_eps=EVAL_EPISODES,
        seed=10_000 + seed,
        max_steps=20,
        budget=10,
    )

    return summary


def main():
    rows = []

    for n_train in TRAIN_EPISODES:
        for seed in SEEDS:
            print(f"Q-learning | train={n_train:,} | seed={seed}")

            summary = run_q(n_train, seed)

            rows.append({
                "algorithm": "q_learning",
                "train_episodes": n_train,
                "seed": seed,
                **{
                    key: summary[key]
                    for key in KEYS
                },
            })

            print(f"Monte Carlo | train={n_train:,} | seed={seed}")

            summary = run_mc(n_train, seed)

            rows.append({
                "algorithm": "monte_carlo",
                "train_episodes": n_train,
                "seed": seed,
                **{
                    key: summary[key]
                    for key in KEYS
                },
            })

    path = OUT / "rl_learning_curve.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "train_episodes",
                "seed",
                *KEYS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()