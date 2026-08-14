import numpy as np

from decoy.environment import InvAction
from decoy.q_learning import QLearning


ACTIONS = [
    (InvAction.OBSERVE, 0),
    (InvAction.ACCUSE, 0),
    (InvAction.VOTE, 0),
]


def test_q_learning_never_selects_an_illegal_action():
    learner = QLearning(
        actions=ACTIONS,
        epsilon=0.0,
        rng=np.random.default_rng(1),
    )
    state = ("state",)

    learner.q[state][:] = [1.0, 2.0, 100.0]

    action = learner.act(
        state,
        available_actions=ACTIONS[:2],
    )

    assert action == (InvAction.ACCUSE, 0)


def test_q_learning_bootstraps_only_from_legal_next_actions():
    learner = QLearning(
        actions=ACTIONS,
        alpha=1.0,
        gamma=0.5,
        epsilon=0.0,
        rng=np.random.default_rng(2),
    )

    state = ("current",)
    next_state = ("next",)

    learner.q[next_state][:] = [2.0, 4.0, 100.0]

    learner.update(
        state=state,
        action=(InvAction.OBSERVE, 0),
        reward=1.0,
        next_state=next_state,
        done=False,
        next_actions=ACTIONS[:2],
    )

    assert np.isclose(learner.q[state][0], 3.0)

def test_q_learning_can_turn_off_exploration_for_eval():
    learner = QLearning(
        actions=ACTIONS,
        epsilon=1.0,
        rng=np.random.default_rng(4),
    )
    state = ("state",)
    learner.q[state][:] = [1.0, 2.0, 100.0]
    action = learner.act(
        state,
        epsilon=0.0,
        available_actions=ACTIONS[:2],
    )
    assert action == (InvAction.ACCUSE, 0)