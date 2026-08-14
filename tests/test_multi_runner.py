import numpy as np

from decoy.agent import FixAgent, TrickAgent
from decoy.environment import DecoyEnv, InvAction, TraderAction
from decoy.multi_runner import run_multi
from decoy.q_learning import QLearning

def acc_vote(state, acts):
    votes = [
        act
        for act in acts
        if act[0] == InvAction.VOTE
    ]
    if votes:
        return votes[0]
    return (InvAction.ACCUSE, 0)

def test_multi_records_all_agents():
    env = DecoyEnv(
        rng=np.random.default_rng(31),
        budget=3,
        max_steps=10,
    )
    trd_agents = {
        0: FixAgent(TraderAction.NORMAL),
        1: FixAgent(TraderAction.CONCEAL),
        2: FixAgent(TraderAction.SIGNAL),
    }
    inv_ep, trd_eps, outcome = run_multi(
        env,
        acc_vote,
        trd_agents,
    )
    assert len(inv_ep) == 2
    assert set(trd_eps) == {0, 1, 2}
    assert all(len(ep) == 2 for ep in trd_eps.values())
    assert all(
        ep.transitions[-1].done
        for ep in trd_eps.values()
    )
    assert all(
        ep.transitions[-1].reward == env.trd_rew(i, outcome)
        for i, ep in trd_eps.items()
    )

def test_multi_timeout_rewards_every_trader():
    env = DecoyEnv(
        rng=np.random.default_rng(32),
        budget=1,
        max_steps=10,
    )
    trd_agents = {
        0: FixAgent(TraderAction.NORMAL),
        1: FixAgent(TraderAction.NORMAL),
        2: FixAgent(TraderAction.NORMAL),
    }
    inv_ep, trd_eps, outcome = run_multi(
        env,
        lambda state, acts: (InvAction.OBSERVE, 0),
        trd_agents,
    )
    assert outcome is None
    assert len(inv_ep) == 1
    assert all(len(ep) == 1 for ep in trd_eps.values())
    assert all(
        ep.transitions[-1].reward == env.trd_rew(i, None)
        for i, ep in trd_eps.items()
    )

def test_trick_agent_commits_to_predicted_faction():
    from decoy.agent import TrickAgent
    from decoy.environment import StatType
    social_actions = [
        (StatType.NEUTRAL, None),
        (StatType.DEFEND, 0),
        (StatType.DEFEND, 1),
        (StatType.ACCUSE, 0),
        (StatType.ACCUSE, 1),
        (StatType.SUPPORT, 0),
        (StatType.SUPPORT, 1),
    ]
    social = QLearning(
        actions=social_actions,
        epsilon=0.0,
        rng=np.random.default_rng(81),
    )
    market = QLearning(
        actions=list(TraderAction),
        epsilon=0.0,
        rng=np.random.default_rng(82),
    )
    from decoy.win_register import (
        WinRegister,
        INVESTIGATOR,
    )
    register = WinRegister(
        min_episodes=0,
        min_probability=0.60,
        min_expected_margin=0,
    )
    for _ in range(8):
        register.record("fraudster_caught")
    for _ in range(2):
        register.record("innocent_caught")
    agent = TrickAgent(
        market,
        social,
        eps=0.0,
        advisor=register,
        episode_no=20,
        total_episodes=100,
        faction_targets={
            "investigator": (0, 1),
            "criminal": (1, 0),
        },
    )
    state = (0, 0, 0, 2, -1, -1)
    social.q[state][:] = 0.0
    # Investigator side:
    # support/defend trader 0, accuse trader 1.
    social.q[state][social_actions.index(
        (StatType.DEFEND, 0)
    )] = 10.0

    first = agent.soc_act(
        state,
        social_actions,
    )

    assert agent.side == INVESTIGATOR
    assert first == (StatType.DEFEND, 0)

    second = agent.soc_act(
        state,
        social_actions,
    )

    assert second[1] in (0, 1)