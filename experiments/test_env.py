import numpy as np

from decoy.environment import DecoyEnv, Role, TraderAction, TRADER_IDS, StatType, Statement


def main():
    rng = np.random.default_rng(42)
    env = DecoyEnv(rng = rng)
    env.reset()
    actions = {
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    }

    inv_obs, done = env.step(actions)

    env.accuse(1)

    reaction = env.react_acc()

    print("ROLES:", env.roles)
    print("ACCUSATION:", env.accus)
    print("REACTION:", reaction)



if __name__ == "__main__":
    main()