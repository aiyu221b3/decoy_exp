import numpy as np

from decoy.q_learning import QLearning
from decoy.toy_env import ToyEnv

def main():
    rng = np.random.default_rng(42)

    env = ToyEnv(rng = rng)
    agent = QLearning(
        actions = [0, 1],
        alpha = 0.1,
        gamma = 0.99,
        epsilon = 0.1,
        rng = rng,
    )

    for _ in range(5000):
        state = env.reset()
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.update(
            state,
            action,
            reward,
            next_state,
            done,
        )

    print("State 0:", agent.q[0])
    print("State 1:", agent.q[1])

if __name__ == "__main__":
    main()