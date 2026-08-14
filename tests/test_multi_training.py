import numpy as np
from decoy.multi_training import ev_multi, tr_multi
from decoy.agent import FixAgent, DualAgent
from decoy.environment import (
    DecoyEnv,
    InvAction,
    TraderAction,
    StatType,
    Role,
)
from decoy.q_learning import QLearning
from decoy.training import inv_action_space, trd_space
from decoy.multi_runner import run_multi
import pytest

def test_ev_multi_returns_one_row_per_episode():
    inv = QLearning(
        actions=inv_action_space(),
        rng=np.random.default_rng(61),
    )
    trds = {
        role: QLearning(
            actions=trd_space(),
            rng=np.random.default_rng(62 + role.value),
        )
        for role in Role
    }
    hist, summ = ev_multi(
        inv,
        trds,
        n_eps=5,
        seed=63,
        budget=3,
        max_steps=10,
    )
    assert len(hist) == 5
    assert summ["fraudster_catch_rate"] + summ["innocent_catch_rate"] + summ["trickster_catch_rate"] + summ["timeout_rate"] == 1.0

def test_tr_multi_trains_every_agent():
    inv, mrk, soc, hist, summ, register = tr_multi(
        n_eps=10,
        seed=64,
        budget=3,
        max_steps=10,
    )
    assert len(inv.actions) == 12
    assert set(mrk) == set(Role)
    assert set(soc) == set(Role)
    assert register.episodes_seen == 10

    assert all(len(agent.actions) == 3 for agent in mrk.values())
    assert all(len(agent.actions) == 10 for agent in soc.values())
    assert len(hist) == 10
    assert summ["fraudster_catch_rate"] + summ["innocent_catch_rate"] + summ["trickster_catch_rate"] + summ["timeout_rate"] == 1.0

def test_tr_multi_trains_both_market_and_social_learners():
    _, mrk, soc, hist, summ, _ = tr_multi(
        n_eps=100,
        seed=65,
        budget=5,
        max_steps=10,
    )

    assert set(mrk) == set(Role)
    assert set(soc) == set(Role)

    assert all(len(learner.q) > 0 for learner in mrk.values())
    assert all(len(learner.q) > 0 for learner in soc.values())

    assert all(
        set(learner.actions) == {
            TraderAction.NORMAL,
            TraderAction.CONCEAL,
            TraderAction.SIGNAL,
        }
        for learner in mrk.values()
    )

    expected_social = {
        (StatType.NEUTRAL, None),
        (StatType.DEFEND, 0),
        (StatType.DEFEND, 1),
        (StatType.DEFEND, 2),
        (StatType.ACCUSE, 0),
        (StatType.ACCUSE, 1),
        (StatType.ACCUSE, 2),
        (StatType.SUPPORT, 0),
        (StatType.SUPPORT, 1),
        (StatType.SUPPORT, 2),
    }

    assert all(
        set(learner.actions) == expected_social
        for learner in soc.values()
    )

def test_social_action_space_has_expected_actions():
    expected = {
        (StatType.NEUTRAL, None),
        (StatType.DEFEND, 0),
        (StatType.DEFEND, 1),
        (StatType.DEFEND, 2),
        (StatType.ACCUSE, 0),
        (StatType.ACCUSE, 1),
        (StatType.ACCUSE, 2),
        (StatType.SUPPORT, 0),
        (StatType.SUPPORT, 1),
        (StatType.SUPPORT, 2),
    }

    _, _, soc, _, _, _ = tr_multi(
        n_eps=1,
        seed=66,
        budget=3,
        max_steps=5,
    )

    assert all(set(learner.actions) == expected for learner in soc.values())

def test_ev_multi_accepts_dual_trader_learners():
    inv = QLearning(
        actions=inv_action_space(),
        rng=np.random.default_rng(67),
    )

    mrk = {
        role: QLearning(
            actions=trd_space(),
            rng=np.random.default_rng(68 + role.value),
        )
        for role in Role
    }

    social_actions = [
        (StatType.NEUTRAL, None),
        (StatType.DEFEND, 0),
        (StatType.DEFEND, 1),
        (StatType.DEFEND, 2),
        (StatType.ACCUSE, 0),
        (StatType.ACCUSE, 1),
        (StatType.ACCUSE, 2),
        (StatType.SUPPORT, 0),
        (StatType.SUPPORT, 1),
        (StatType.SUPPORT, 2),
    ]

    soc = {
        role: QLearning(
            actions=social_actions,
            rng=np.random.default_rng(69 + role.value),
        )
        for role in Role
    }

    hist, summ = ev_multi(
        inv,
        mrk,
        soc,
        n_eps=5,
        seed=70,
        budget=3,
        max_steps=10,
    )

    assert len(hist) == 5
    assert (
        summ["fraudster_catch_rate"]
        + summ["innocent_catch_rate"]
        + summ["trickster_catch_rate"]
        + summ["timeout_rate"]
        == 1.0
    )

def test_multi_records_social_transitions_for_dual_agents():
    env = DecoyEnv(
        rng=np.random.default_rng(33),
        budget=3,
        max_steps=10,
    )

    social_actions = [
        (StatType.NEUTRAL, None),
        (StatType.DEFEND, 0),
        (StatType.DEFEND, 1),
        (StatType.DEFEND, 2),
        (StatType.ACCUSE, 0),
        (StatType.ACCUSE, 1),
        (StatType.ACCUSE, 2),
        (StatType.SUPPORT, 0),
        (StatType.SUPPORT, 1),
        (StatType.SUPPORT, 2),
    ]

    agents = {
        i: DualAgent(
            QLearning(
                actions=list(TraderAction),
                epsilon=0.0,
                rng=np.random.default_rng(40 + i),
            ),
            QLearning(
                actions=social_actions,
                epsilon=0.0,
                rng=np.random.default_rng(50 + i),
            ),
            eps=0.0,
        )
        for i in range(3)
    }

    def accuse_policy(state, acts):
        return (InvAction.ACCUSE, 1)

    inv_ep, trd_eps, outcome = run_multi(
        env,
        accuse_policy,
        agents,
        state_fn=env.inv_q_state,
        trd_state_fn=env.trd_q_state,
    )

    social_transitions = [
        trans
        for ep in trd_eps.values()
        for trans in ep.transitions
        if isinstance(trans.action, tuple)
        and isinstance(trans.action[0], StatType)
    ]

    assert social_transitions
    assert all(trans.done for trans in social_transitions)
    assert all(trans.next_observation is None for trans in social_transitions)

def test_tr_multi_is_reproducible():
    result_a = tr_multi(
        n_eps=20,
        seed=83,
        budget=3,
        max_steps=10,
    )

    result_b = tr_multi(
        n_eps=20,
        seed=83,
        budget=3,
        max_steps=10,
    )

    inv_a, mrk_a, soc_a, hist_a, summ_a, register_a = result_a
    inv_b, mrk_b, soc_b, hist_b, summ_b, register_b = result_b

    assert hist_a == hist_b
    assert summ_a == summ_b
    assert register_a.episodes_seen == register_b.episodes_seen
    assert register_a.all_history == register_b.all_history
    assert list(register_a.recent_history) == list(
    register_b.recent_history
    )

    assert inv_a.q.keys() == inv_b.q.keys()

    for state in inv_a.q:
        assert np.array_equal(
            inv_a.q[state],
            inv_b.q[state],
        )

    for role in Role:
        assert mrk_a[role].q.keys() == mrk_b[role].q.keys()
        assert soc_a[role].q.keys() == soc_b[role].q.keys()

        for state in mrk_a[role].q:
            assert np.array_equal(
                mrk_a[role].q[state],
                mrk_b[role].q[state],
            )

        for state in soc_a[role].q:
            assert np.array_equal(
                soc_a[role].q[state],
                soc_b[role].q[state],
            )

def test_tr_multi_register_is_reproducible():
    result_a = tr_multi(
        n_eps=20,
        seed=84,
        budget=3,
        max_steps=10,
    )

    result_b = tr_multi(
        n_eps=20,
        seed=84,
        budget=3,
        max_steps=10,
    )

    register_a = result_a[-1]
    register_b = result_b[-1]

    assert register_a.episodes_seen == register_b.episodes_seen
    assert register_a.all_history == register_b.all_history
    assert list(register_a.recent_history) == list(
        register_b.recent_history
    )

def test_adaptive_inv_alpha_stays_within_bounds():
    inv, _, _, _, _, _ = tr_multi(
        n_eps=250,
        seed=91,
        budget=3,
        max_steps=10,
        adaptive_inv_alpha=True,
        alpha=0.1,
        alpha_inv_min=0.03,
        alpha_inv_max=0.12,
        alpha_inv_step=0.005,
        performance_window=10,
        performance_tolerance=0.1,
    )

    assert 0.03 <= inv.alpha <= 0.12

def test_nonadaptive_inv_alpha_stays_fixed():
    inv, _, _, _, _, _ = tr_multi(
        n_eps=100,
        seed=92,
        budget=3,
        max_steps=10,
        adaptive_inv_alpha=False,
        alpha=0.1,
    )

    assert inv.alpha == pytest.approx(0.1)

def test_adaptive_inv_alpha_is_reproducible():
    result_a = tr_multi(
        n_eps=250,
        seed=93,
        budget=3,
        max_steps=10,
        adaptive_inv_alpha=True,
        performance_window=10,
    )

    result_b = tr_multi(
        n_eps=250,
        seed=93,
        budget=3,
        max_steps=10,
        adaptive_inv_alpha=True,
        performance_window=10,
    )

    assert result_a[-2]["inv_rew"] == pytest.approx(
        result_b[-2]["inv_rew"]
    )
    assert result_a[0].q.keys() == result_b[0].q.keys()

    for state in result_a[0].q:
        assert np.array_equal(
            result_a[0].q[state],
            result_b[0].q[state],
        )

    assert result_a[0].alpha == pytest.approx(
        result_b[0].alpha
    )
    assert result_a[0].alpha == pytest.approx(result_b[0].alpha)

def test_warmup_does_not_change_episode_count():
    _, _, _, hist, _, register = tr_multi(
        n_eps=20,
        seed=94,
        budget=3,
        max_steps=10,
        warmup_episodes=10,
    )

    assert len(hist) == 20
    assert register.episodes_seen == 20

def test_zero_warmup_preserves_reproducibility():
    result_a = tr_multi(
        n_eps=20,
        seed=95,
        budget=3,
        max_steps=10,
        warmup_episodes=0,
    )

    result_b = tr_multi(
        n_eps=20,
        seed=95,
        budget=3,
        max_steps=10,
        warmup_episodes=0,
    )

    assert result_a[3] == result_b[3]
    assert result_a[4] == result_b[4]

def test_warmup_changes_investigator_learning():
    cold, _, _, _, _, _ = tr_multi(
        n_eps=20,
        seed=96,
        budget=3,
        max_steps=10,
        warmup_episodes=0,
    )

    warm, _, _, _, _, _ = tr_multi(
        n_eps=20,
        seed=96,
        budget=3,
        max_steps=10,
        warmup_episodes=20,
    )

    assert len(warm.q) > 0
    assert warm.q != cold.q

