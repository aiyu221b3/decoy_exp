from decoy.environment import InvAction, Role, TraderAction, DecoyEnv
from decoy.training import (
    inv_action_space,
    train_q_investigator, 
    train_mc_investigator,
    eval_inv,
    mk_role, 
    trd_space
)
from decoy.agent import FixAgent
import numpy as np
import numpy as np
from decoy.environment import Role
from decoy.training import tr_trd


def test_investigator_action_space_has_twelve_actions():
    actions = inv_action_space()
    assert len(actions) == 12
    assert set(action[0] for action in actions) == {
        InvAction.OBSERVE,
        InvAction.FOCUS,
        InvAction.ACCUSE,
        InvAction.VOTE,
    }


def test_q_training_returns_metrics_for_every_episode():
    learner, history, summary = train_q_investigator(
        n_episodes=10,
        seed=21,
        budget=3,
        max_steps=10,
    )

    assert len(learner.actions) == 12
    assert len(history) == 10

    expected_row_keys = {
        "episode",
        "reward",
        "outcome",
        "observations",
        "accusations",
        "votes",
        "length",
        "remaining_budget",
    }
    assert set(history[0]) == expected_row_keys

    assert (
        summary["fraudster_catch_rate"]
        + summary["innocent_catch_rate"]
        + summary["trickster_catch_rate"]
        + summary["timeout_rate"]
        == 1.0
    )


def test_q_training_is_reproducible_for_a_seed():
    settings = {
        "n_episodes": 10,
        "seed": 99,
        "budget": 3,
        "max_steps": 10,
    }

    _, first_history, first_summary = train_q_investigator(
        **settings,
    )
    _, second_history, second_summary = train_q_investigator(
        **settings,
    )

    assert first_history == second_history
    assert first_summary == second_summary

def test_mc_training_returns_metrics_for_every_episode():
    learner, history, summary = train_mc_investigator(
        n_episodes=10,
        seed=22,
        budget=3,
        max_steps=10,
    )
    assert len(learner.actions) == 12
    assert len(history) == 10
    assert (
        summary["fraudster_catch_rate"]
        + summary["innocent_catch_rate"]
        + summary["trickster_catch_rate"]
        + summary["timeout_rate"]
        == 1.0
    )

def test_eval_inv_returns_metrics_for_every_episode():
    learner, _, _ = train_q_investigator(
        n_episodes=10,
        seed=25,
        budget=3,
        max_steps=10,
    )
    hist, summ = eval_inv(
        learner,
        n_eps=10,
        seed=26,
        budget=3,
        max_steps=10,
    )
    assert len(hist) == 10
    assert (
        summ["fraudster_catch_rate"]
        + summ["innocent_catch_rate"]
        + summ["trickster_catch_rate"]
        + summ["timeout_rate"]
        == 1.0
    )

def test_trd_space_has_three_market_actions():
    assert trd_space() == [
        TraderAction.NORMAL,
        TraderAction.CONCEAL,
        TraderAction.SIGNAL,
    ]

def test_mk_role_maps_agents_after_roles_are_assigned():
    env = DecoyEnv(rng=np.random.default_rng(41))
    env.reset()
    env.roles = {
        0: Role.TRICKSTER,
        1: Role.INNOCENT,
        2: Role.FRAUDSTER,
    }
    agents = {
        Role.INNOCENT: FixAgent(TraderAction.NORMAL),
        Role.FRAUDSTER: FixAgent(TraderAction.CONCEAL),
        Role.TRICKSTER: FixAgent(TraderAction.SIGNAL),
    }
    trds = mk_role(env, agents)
    assert trds[0].action == TraderAction.SIGNAL
    assert trds[1].action == TraderAction.NORMAL
    assert trds[2].action == TraderAction.CONCEAL

def test_tr_trd_trains_each_role_without_updating_the_investigator():
    inv, _, _ = train_q_investigator(
        n_episodes=10,
        seed=52,
        budget=3,
        max_steps=10,
    )
    q_before = {
        state: vals.copy()
        for state, vals in inv.q.items()
    }
    learners, hist, summ = tr_trd(
        n_eps=10,
        inv=inv,
        seed=53,
        budget=3,
        max_steps=10,
    )
    assert set(learners) == set(Role)
    assert len(hist) == 10
    assert set(summ) == {
        "inn_rew",
        "fraud_rew",
        "trick_rew",
        "fraudster_catch_rate",
        "innocent_catch_rate",
        "trickster_catch_rate",
        "timeout_rate",
    }
    assert all(
        np.array_equal(inv.q[state], vals)
        for state, vals in q_before.items()
    )

def test_q_training_uses_the_compact_learning_state():
    learner, _, _ = train_q_investigator(
        n_episodes=10,
        seed=81,
        budget=3,
        max_steps=10,
    )
    assert all(len(state) == 16 for state in learner.q)

def test_tr_trd_uses_compact_trader_states():
    inv, _, _ = train_q_investigator(
        n_episodes=10,
        seed=91,
        budget=3,
        max_steps=10,
    )
    learners, _, _ = tr_trd(
        n_eps=10,
        inv=inv,
        seed=92,
        budget=3,
        max_steps=10,
    )
    assert all(
        len(state) == 6
        for learner in learners.values()
        for state in learner.q
    )

