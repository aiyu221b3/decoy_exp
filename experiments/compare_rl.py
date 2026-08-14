from decoy.training import eval_inv
from decoy.training import train_mc_investigator as tr_mc
from decoy.training import train_q_investigator as tr_q

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

def avg(rows):
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in KEYS
    }

def run_q(seeds, n_tr, n_ev):
    rows = []
    for seed in seeds:
        agent, _, _ = tr_q(
            n_episodes=n_tr,
            seed=seed,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1,
            max_steps=20,
            budget=10,
        )
        _, summ = eval_inv(
            agent,
            n_eps=n_ev,
            seed=10_000 + seed,
            max_steps=20,
            budget=10,
        )
        rows.append(summ)
    return rows

def run_mc(seeds, n_tr, n_ev):
    rows = []
    for seed in seeds:
        agent, _, _ = tr_mc(
            n_episodes=n_tr,
            seed=seed,
            gamma=0.99,
            epsilon=0.1,
            max_steps=20,
            budget=10,
        )
        _, summ = eval_inv(
            agent,
            n_eps=n_ev,
            seed=10_000 + seed,
            max_steps=20,
            budget=10,
        )
        rows.append(summ)
    return rows

def show(name, rows):
    print(f"\n{name}")
    for i, row in enumerate(rows, start=1):
        print(f"\nseed run {i}")
        for key in KEYS:
            print(f"{key}: {row[key]:.3f}")
    print("\naverage")
    for key, val in avg(rows).items():
        print(f"{key}: {val:.3f}")

def main():
    seeds = (11, 42, 99)
    n_tr = 10_000
    n_ev = 2_000
    show("Q-learning", run_q(seeds, n_tr, n_ev))
    show("Monte Carlo", run_mc(seeds, n_tr, n_ev))

if __name__ == "__main__":
    main()