import csv
from pathlib import Path

from decoy.multi_training import tr_multi


SEEDS = [11, 42, 99]

N_EPS = 1000
BUDGET = 10
MAX_STEPS = 20


def main():
    rows = []

    configs = [
        {
            "name": "fixed_alpha",
            "adaptive_inv_alpha": False,
        },
        {
            "name": "adaptive_alpha",
            "adaptive_inv_alpha": True,
        },
    ]

    total_runs = len(configs) * len(SEEDS)
    run_no = 0

    for config in configs:
        for seed in SEEDS:
            run_no += 1

            print(
                f"[{run_no}/{total_runs}] "
                f"{config['name']}, seed={seed}"
            )

            inv, _, _, _, summ, _ = tr_multi(
                n_eps=N_EPS,
                seed=seed,
                budget=BUDGET,
                max_steps=MAX_STEPS,
                adaptive_inv_alpha=config[
                    "adaptive_inv_alpha"
                ],
            )

            rows.append({
                "condition": config["name"],
                "seed": seed,
                "train_episodes": N_EPS,
                "inv_rew": summ["inv_rew"],
                "final_alpha": inv.alpha,
                "fraudster_catch_rate": (
                    summ["fraudster_catch_rate"]
                ),
                "innocent_catch_rate": (
                    summ["innocent_catch_rate"]
                ),
                "trickster_catch_rate": (
                    summ["trickster_catch_rate"]
                ),
                "timeout_rate": summ["timeout_rate"],
                "mean_observations": (
                    summ["mean_observations"]
                ),
                "mean_accusations": (
                    summ["mean_accusations"]
                ),
                "mean_votes": summ["mean_votes"],
                "mean_episode_length": (
                    summ["mean_episode_length"]
                ),
                "mean_remaining_budget": (
                    summ["mean_remaining_budget"]
                ),
                "mean_focuses": summ["mean_focuses"],
            })

            print(
                    f"    alpha={inv.alpha:.4f} | "
                    f"reward={summ['inv_rew']:.3f} | "
                    f"fraudster={summ['fraudster_catch_rate']:.3f} | "
                    f"innocent={summ['innocent_catch_rate']:.3f} | "
                    f"trickster={summ['trickster_catch_rate']:.3f} | "
                    f"timeout={summ['timeout_rate']:.3f}"
                )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "adaptive_alpha_sweep.csv"

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()