from pathlib import Path
import csv
import json

from decoy.multi_training import tr_multi, ev_multi
from decoy.training import tr_trd, train_q_investigator


RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_EPISODES = (
    50_000,
    100_000,
    250_000,
    500_000,
)

EVAL_EPISODES = 2_000
SEEDS = (71, 74, 77)

HIST_KEYS = (
    "episode",
    "outcome",
    "inv_rew",
    "inn_rew",
    "fraud_rew",
    "trick_rew",
    "observations",
    "accusations",
    "votes",
    "length",
    "remaining_budget",
    "focuses",
)


def save_rows(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    learning_rows = []
    summary_rows = []
    register_rows = []

    for n_train in TRAIN_EPISODES:
        for seed in SEEDS:

            print(
                f"\nJoint training | "
                f"{n_train:,} episodes | seed {seed}"
            )

            (
                inv,
                mrk,
                soc,
                hist,
                summary,
                register,
            ) = tr_multi(
                n_eps=n_train,
                seed=seed,
                max_steps=20,
                budget=10,
            )

            for row in hist:
                learning_rows.append({
                    "train_episodes": n_train,
                    "seed": seed,
                    **{
                        key: row[key]
                        for key in HIST_KEYS
                    },
                })

            summary_rows.append({
                "train_episodes": n_train,
                "seed": seed,
                **summary,
            })

            rates = register.rates()
            prediction = register.predict()

            register_rows.append({
                "train_episodes": n_train,
                "seed": seed,
                "episodes_seen": register.episodes_seen,
                "investigator_all": rates["investigator"]["all"],
                "criminal_all": rates["criminal"]["all"],
                "investigator_recent": rates["investigator"]["recent"],
                "criminal_recent": rates["criminal"]["recent"],
                "predicted_side": prediction[0],
                "prediction_confidence": prediction[1],
            })

            print(
                f"  investigator reward: "
                f"{summary['inv_rew']:.3f}"
            )
            print(
                f"  fraudster catch: "
                f"{summary['fraudster_catch_rate']:.3f}"
            )
            print(
                f"  trickster catch: "
                f"{summary['trickster_catch_rate']:.3f}"
            )

    save_rows(
        RESULTS / "multi_learning_curve.csv",
        learning_rows,
        [
            "train_episodes",
            "seed",
            *HIST_KEYS,
        ],
    )

    summary_fields = (
        ["train_episodes", "seed"]
        + [
            key
            for key in summary_rows[0]
            if key not in {"train_episodes", "seed"}
        ]
    )

    save_rows(
        RESULTS / "multi_summary.csv",
        summary_rows,
        summary_fields,
    )

    save_rows(
        RESULTS / "trickster_register.csv",
        register_rows,
        [
            "train_episodes",
            "seed",
            "episodes_seen",
            "investigator_all",
            "criminal_all",
            "investigator_recent",
            "criminal_recent",
            "predicted_side",
            "prediction_confidence",
        ],
    )

    print("\nSaved multi-agent analysis data.")


if __name__ == "__main__":
    main()