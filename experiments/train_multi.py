from decoy.multi_training import ev_multi, tr_multi
from decoy.training import tr_trd, train_q_investigator

def show(name, summ):
    print(f"\n{name}")
    for key, val in summ.items():
        print(f"{key}: {val:.3f}")

def main():
    TRAIN_EPISODES = 10_000
    EVAL_EPISODES = 2_000
    inv, _, _ = train_q_investigator(
        n_episodes=TRAIN_EPISODES,
        seed=71,
        max_steps=20,
        budget=10,
    )
    trds, _, trd_summ = tr_trd(
        n_eps=TRAIN_EPISODES,
        inv=inv,
        seed=72,
        max_steps=20,
        budget=10,
    )
    _, staged_summ = ev_multi(
        inv,
        trds,
        n_eps=EVAL_EPISODES,
        seed=73,
        max_steps=20,
        budget=10,
    )
    (
        joint_inv,
        joint_mrk,
        joint_soc,
        _,
        joint_summ,
        win_register,
    ) = tr_multi(
        n_eps=TRAIN_EPISODES,
        seed=74,
        max_steps=20,
        budget=10,
    )
    _, joint_ev_summ = ev_multi(
        joint_inv,
        joint_mrk,
        joint_soc,
        n_eps=EVAL_EPISODES,
        seed=75,
        max_steps=20,
        budget=10,
        win_register=win_register,
    )
    _, transfer_ev_summ = ev_multi(
        inv,
        joint_mrk,
        joint_soc,
        n_eps=EVAL_EPISODES,
        seed=76,
        max_steps=20,
        budget=10,
        win_register=win_register,
    )
    show(
        "Traders versus frozen investigator",
        trd_summ,
    )
    show(
        "Frozen investigator versus learned traders",
        staged_summ,
    )
    show(
        "Joint training",
        joint_summ,
    )
    show(
        "Jointly trained system",
        joint_ev_summ,
    )
    show(
        "Frozen investigator versus jointly trained traders",
        transfer_ev_summ,
    )
    print("\nTrickster win-history prediction")
    print(win_register.rates())
    print(win_register.predict())

if __name__ == "__main__":
    main()