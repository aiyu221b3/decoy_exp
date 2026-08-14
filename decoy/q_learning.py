from collections import defaultdict
import numpy as np

class QLearning:

    def __init__(
            self,
            actions,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1,
            rng=None,
    ):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.q = defaultdict(self._new_q)

    def _new_q(self):
        return np.zeros(len(self.actions), dtype=float)

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

    def act(self, state, available_actions=None, epsilon=None):
        action_ids = self._action_ids(available_actions)
        epsilon = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < epsilon:
            action_id = int(self.rng.choice(action_ids))
            return self.actions[action_id]
        values = self.q[state][action_ids]
        best_ids = action_ids[np.flatnonzero(values == np.max(values))]
        action_id = int(self.rng.choice(best_ids))
        return self.actions[action_id]

    def update(
            self,
            state,
            action,
            reward,
            next_state,
            done,
            next_actions=None,
    ):
        action_id = self.actions.index(action)
        current = self.q[state][action_id]

        if done:
            target = reward
        else:
            next_ids = self._action_ids(next_actions)
            target = reward + self.gamma * np.max(
                self.q[next_state][next_ids]
            )

        self.q[state][action_id] += self.alpha * (target - current)