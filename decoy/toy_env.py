import numpy as np

class ToyEnv:
    def __init__(self, rng = None):
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng()
        )
        self.state = None

    def reset(self):
        self.state = int(self.rng.integers(0, 2))
        return self.state

    def step(self, action):
        reward = int(action == self.state)
        done = True

        next_state = None

        return next_state, reward, done