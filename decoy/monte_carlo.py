from collections import defaultdict
import numpy as np

class MonteCarlo:
    def __init__(self, actions, gamma=0.99, rng=None):
        self.actions = list(actions)
        self.gamma = gamma
        self.rng = rng if rng is not None else np.random.default_rng()
        self.q = defaultdict(self._new_q)
        self.n = defaultdict(self._new_n)

    def _new_q(self):
        return np.zeros(len(self.actions), dtype=float)

    def _new_n(self):
        return np.zeros(len(self.actions), dtype=int)

    def _action_ids(self, available_actions=None):
        if available_actions is None:
            return np.arange(len(self.actions))
        action_ids = np.array(
            [self.actions.index(action) for action in available_actions],
            dtype=int,
        )
        if len(action_ids) == 0:
            raise ValueError("no legal actions on the menu, tragic")
        return action_ids

    def act(self, state, epsilon=0.1, available_actions=None):
        action_ids = self._action_ids(available_actions)
        if self.rng.random() < epsilon:
            action_id = int(self.rng.choice(action_ids))
            return self.actions[action_id]
        values = self.q[state][action_ids]
        best_ids = action_ids[np.flatnonzero(values == np.max(values))]
        action_id = int(self.rng.choice(best_ids))
        return self.actions[action_id]

    def update(self, episode):
        G = 0.0
        for transition in reversed(episode.transitions):
            G = transition.reward + self.gamma * G
            state = transition.observation
            action_id = self.actions.index(transition.action)
            self.n[state][action_id] += 1
            self.q[state][action_id] += (
                G - self.q[state][action_id]
            ) / self.n[state][action_id]