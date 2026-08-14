import numpy as np
from decoy.environment import InvAction
from decoy.monte_carlo import MonteCarlo

ACTIONS = [
    (InvAction.OBSERVE, 0),
    (InvAction.ACCUSE, 0),
    (InvAction.VOTE, 0),
]

def test_monte_carlo_never_selects_an_illegal_action():
    learner = MonteCarlo(
        actions=ACTIONS,
        rng=np.random.default_rng(3),
    )
    state = ("state",)
    learner.q[state][:] = [1.0, 2.0, 100.0]
    action = learner.act(
        state,
        epsilon=0.0,
        available_actions=ACTIONS[:2],
    )
    assert action == (InvAction.ACCUSE, 0)