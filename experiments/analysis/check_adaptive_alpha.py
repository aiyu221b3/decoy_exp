import numpy as np

from decoy.multi_training import tr_multi


SEEDS = [11, 42, 99]

N_EPS = 1000
BUDGET = 10
MAX_STEPS = 20


def q_table_difference(q_a, q_b):
    """Compare two Q-tables over their shared states."""

    shared_states = set(q_a) & set(q_b)

    if not shared_states:
        return {
            "shared_states": 0,
            "different_states": 0,
            "max_difference": 0.0,
            "mean_difference": 0.0,
        }

    differences = []

    for state in shared_states:
        a = np.asarray(q_a[state])
        b = np.asarray(q_b[state])

        differences.append(
            float(np.max(np.abs(a - b)))
        )

    different_states = sum(
        diff > 1e-12
        for diff in differences
    )

    return {
        "shared_states": len(shared_states),
        "different_states": different_states,
        "max_difference": max(differences),
        "mean_difference": float(np.mean(differences)),
    }


def policy_disagreement(q_a, q_b):
    """Measure disagreement between greedy policies."""

    shared_states = set(q_a) & set(q_b)

    if not shared_states:
        return {
            "shared_states": 0,
            "different_actions": 0,
            "disagreement_rate": 0.0,
        }

    different_actions = 0

    for state in shared_states:
        action_a = int(np.argmax(q_a[state]))
        action_b = int(np.argmax(q_b[state]))

        if action_a != action_b:
            different_actions += 1

    return {
        "shared_states": len(shared_states),
        "different_actions": different_actions,
        "disagreement_rate": (
            different_actions / len(shared_states)
        ),
    }


def trajectory_difference(hist_a, hist_b):
    """Compare episode-level outcomes and summary behavior."""

    n = min(len(hist_a), len(hist_b))

    if n == 0:
        return {
            "episodes_compared": 0,
            "different_outcomes": 0,
            "outcome_disagreement_rate": 0.0,
            "different_rewards": 0,
            "reward_disagreement_rate": 0.0,
        }

    different_outcomes = 0
    different_rewards = 0

    for row_a, row_b in zip(
        hist_a[:n],
        hist_b[:n],
    ):
        if row_a["outcome"] != row_b["outcome"]:
            different_outcomes += 1

        if row_a["inv_rew"] != row_b["inv_rew"]:
            different_rewards += 1

    return {
        "episodes_compared": n,
        "different_outcomes": different_outcomes,
        "outcome_disagreement_rate": (
            different_outcomes / n
        ),
        "different_rewards": different_rewards,
        "reward_disagreement_rate": (
            different_rewards / n
        ),
    }


def run_condition(
    seed,
    adaptive,
):
    return tr_multi(
        n_eps=N_EPS,
        seed=seed,
        budget=BUDGET,
        max_steps=MAX_STEPS,
        adaptive_inv_alpha=adaptive,
    )


def main():
    print("=" * 72)
    print("ADAPTIVE INVESTIGATOR ALPHA DIAGNOSTIC")
    print("=" * 72)
    print()

    for seed in SEEDS:
        print(f"SEED {seed}")
        print("-" * 72)

        fixed = run_condition(
            seed,
            adaptive=False,
        )

        adaptive = run_condition(
            seed,
            adaptive=True,
        )

        fixed_inv = fixed[0]
        adaptive_inv = adaptive[0]

        fixed_hist = fixed[3]
        adaptive_hist = adaptive[3]

        fixed_summ = fixed[4]
        adaptive_summ = adaptive[4]

        # --------------------------------------------------
        # 1. Final alpha
        # --------------------------------------------------

        print("Alpha:")
        print(
            f"  fixed:    {fixed_inv.alpha:.6f}"
        )
        print(
            f"  adaptive: {adaptive_inv.alpha:.6f}"
        )

        # --------------------------------------------------
        # 2. Performance
        # --------------------------------------------------

        print()
        print("Performance:")

        print(
            f"  fixed reward:    "
            f"{fixed_summ['inv_rew']:.6f}"
        )

        print(
            f"  adaptive reward: "
            f"{adaptive_summ['inv_rew']:.6f}"
        )

        print(
            f"  reward delta:    "
            f"{adaptive_summ['inv_rew'] - fixed_summ['inv_rew']:.6f}"
        )

        # --------------------------------------------------
        # 3. Q-table difference
        # --------------------------------------------------

        q_diff = q_table_difference(
            fixed_inv.q,
            adaptive_inv.q,
        )

        print()
        print("Q-table difference:")
        print(
            f"  shared states:    "
            f"{q_diff['shared_states']}"
        )
        print(
            f"  changed states:   "
            f"{q_diff['different_states']}"
        )
        print(
            f"  mean difference:  "
            f"{q_diff['mean_difference']:.12g}"
        )
        print(
            f"  max difference:   "
            f"{q_diff['max_difference']:.12g}"
        )

        # --------------------------------------------------
        # 4. Greedy policy difference
        # --------------------------------------------------

        policy_diff = policy_disagreement(
            fixed_inv.q,
            adaptive_inv.q,
        )

        print()
        print("Greedy-policy difference:")
        print(
            f"  shared states:    "
            f"{policy_diff['shared_states']}"
        )
        print(
            f"  changed actions:  "
            f"{policy_diff['different_actions']}"
        )
        print(
            f"  disagreement:    "
            f"{policy_diff['disagreement_rate']:.6%}"
        )

        # --------------------------------------------------
        # 5. Actual trajectory difference
        # --------------------------------------------------

        trajectory_diff = trajectory_difference(
            fixed_hist,
            adaptive_hist,
        )

        print()
        print("Episode trajectory difference:")
        print(
            f"  episodes:         "
            f"{trajectory_diff['episodes_compared']}"
        )
        print(
            f"  different outcomes:"
            f" {trajectory_diff['different_outcomes']}"
        )
        print(
            f"  outcome disagreement:"
            f" {trajectory_diff['outcome_disagreement_rate']:.6%}"
        )
        print(
            f"  different rewards: "
            f"{trajectory_diff['different_rewards']}"
        )
        print(
            f"  reward disagreement:"
            f" {trajectory_diff['reward_disagreement_rate']:.6%}"
        )

        print()
        print("=" * 72)
        print()


if __name__ == "__main__":
    main()