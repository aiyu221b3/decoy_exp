import numpy as np

from decoy.environment import DecoyEnv, InvAction, TraderAction
from decoy.runner import (
    default_trader_actions,
    run_investigator_episode,
    trd_acts,
)
from decoy.agent import FixAgent


def accuse_then_vote_policy(state, valid_actions):
    vote_actions = [
        action
        for action in valid_actions
        if action[0] == InvAction.VOTE
    ]

    if vote_actions:
        return vote_actions[0]

    return (InvAction.ACCUSE, 0)


def observe_policy(state, valid_actions):
    return next(
        action
        for action in valid_actions
        if action[0] == InvAction.OBSERVE
    )


def test_default_trader_actions_follow_roles():
    env = DecoyEnv(rng=np.random.default_rng(4))
    env.reset()
    actions = default_trader_actions(env)
    assert set(actions) == {0, 1, 2}
    assert all(action is not None for action in actions.values())


def test_runner_records_an_accuse_then_vote_episode():
    env = DecoyEnv(
        rng=np.random.default_rng(11),
        budget=3,
        max_steps=10,
    )
    episode, outcome = run_investigator_episode(
        env,
        accuse_then_vote_policy,
    )
    assert len(episode) == 2
    assert episode.transitions[0].action == (InvAction.ACCUSE, 0)
    assert episode.transitions[0].reward == 0
    assert episode.transitions[0].done is False
    assert len(episode.transitions[0].observation) == 32
    final_transition = episode.transitions[-1]
    assert final_transition.action == (InvAction.VOTE, 0)
    assert final_transition.done is True
    assert final_transition.next_observation is None
    assert final_transition.reward == env.inv_reward(outcome)
    assert env.done is True


def test_runner_ends_when_the_budget_is_gone():
    env = DecoyEnv(
        rng=np.random.default_rng(12),
        budget=1,
        max_steps=10,
    )
    episode, outcome = run_investigator_episode(
        env,
        observe_policy,
    )
    assert outcome is None
    assert len(episode) == 1
    assert episode.transitions[0].action[0] == InvAction.OBSERVE
    assert episode.transitions[0].reward == -1
    assert episode.transitions[0].done is True
    assert episode.transitions[0].next_observation is None
    assert env.done is True

def test_trd_acts_use_the_agent_interface():
    env = DecoyEnv(rng=np.random.default_rng(8))
    env.reset()
    agents = {
        0: FixAgent(TraderAction.SIGNAL),
        1: FixAgent(TraderAction.CONCEAL),
        2: FixAgent(TraderAction.NORMAL),
    }
    assert trd_acts(env, agents) == {
        0: TraderAction.SIGNAL,
        1: TraderAction.CONCEAL,
        2: TraderAction.NORMAL,
    }