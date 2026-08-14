import numpy as np
import pytest

from decoy.environment import (
    DecoyEnv,
    InvAction,
    Role,
    StatType,
    TraderAction,
    TRADER_IDS,
)


def make_env(budget=5, max_steps=4):
    env = DecoyEnv(
        rng=np.random.default_rng(7),
        budget=budget,
        max_steps=max_steps,
    )
    env.reset()

    # Fix roles so the episode outcome is deterministic.
    env.roles = {
        0: Role.INNOCENT,
        1: Role.FRAUDSTER,
        2: Role.TRICKSTER,
    }
    return env


def test_complete_accusation_reaction_vote_episode():
    env = make_env()

    normal_actions = {
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    }

    _, done = env.step(normal_actions)
    assert not done

    observed_values = env.inv_act((InvAction.OBSERVE, 0))
    assert len(observed_values) == 1
    assert env.rem_budget == 4

    accused_target = env.inv_act((InvAction.ACCUSE, 1))
    assert accused_target == 1
    assert env.rem_budget == 3

    reaction_actions = env.react_acc()
    assert reaction_actions == {
        0: TraderAction.NORMAL,
        1: TraderAction.CONCEAL,
        2: TraderAction.NORMAL,
    }

    statement = env.soc_reac()
    assert statement.speaker == 1
    assert statement.kind == StatType.ACCUSE
    assert statement in env.history
    assert env.get_prof(1)["acc_rate"] == 1.0

    _, done = env.step(reaction_actions)
    assert not done

    outcome = env.inv_act((InvAction.VOTE, 1))
    assert outcome == "fraudster_caught"
    assert env.inv_reward(outcome) == 10
    assert env.done
    assert env.rem_budget == 2


def test_vote_must_target_the_accused_trader():
    env = make_env()
    env.accuse(0)
    remaining_budget = env.rem_budget

    with pytest.raises(ValueError, match="needs to match"):
        env.vote(1)

    assert env.rem_budget == remaining_budget
    assert not env.done


@pytest.mark.parametrize(
    ("outcome", "expected_reward"),
    [
        ("fraudster_caught", 10),
        ("innocent_caught", -10),
        ("trickster_caught", -3),
                (None, -1),
    ],
)
def test_inv_reward_mapping(outcome, expected_reward):
    env = make_env()

    assert env.inv_reward(outcome) == expected_reward

def test_observe_saves_a_focused_history_summary():
    env = make_env()

    normal_actions = {
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    }

    inv_obs, done = env.step(normal_actions)
    assert not done

    history = env.inv_observe(0)
    refreshed_obs = env.get_inv_obs(inv_obs["market"])
    inspection = refreshed_obs["inspections"][0]

    assert env.obs_hist[0][-5:] == history
    assert inspection["seen"] is True
    assert inspection["mean"] == pytest.approx(np.mean(history))
    assert inspection["trend"] == pytest.approx(
        history[-1] - history[0]
    )
    assert refreshed_obs["inspections"][1]["seen"] is False
    assert len(env.inv_state(refreshed_obs)) == 32

def test_trd_obs_is_role_blind():
    env = make_env()
    env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })
    obs = env.trd_obs(1)
    assert "roles" not in obs
    assert "role" not in obs
    assert obs["own_mrk"] == obs["market"][1]
    assert set(obs["market"]) == {0, 1, 2}
    assert len(env.trd_state(1)) == 15

def test_trd_rew_has_the_approved_values():
    env = make_env()
    env.roles = {
        0: Role.INNOCENT,
        1: Role.FRAUDSTER,
        2: Role.TRICKSTER,
    }
    assert env.trd_rew(0, "fraudster_caught") == 5
    assert env.trd_rew(0, "innocent_caught") == -10
    assert env.trd_rew(0, "trickster_caught") == 1
    assert env.trd_rew(0, None) == 0
    assert env.trd_rew(1, "fraudster_caught") == -10
    assert env.trd_rew(1, "innocent_caught") == 10
    assert env.trd_rew(1, "trickster_caught") == 2
    assert env.trd_rew(1, None) == 1
    assert env.trd_rew(2, "fraudster_caught") == -3
    assert env.trd_rew(2, "innocent_caught") == 5
    assert env.trd_rew(2, "trickster_caught") == -10
    assert env.trd_rew(2, None) == 1

def test_inv_q_state_is_compact():
    env = make_env()
    inv_obs, _ = env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })
    env.inv_observe(0)
    inv_obs = env.get_inv_obs(inv_obs["market"])
    assert len(env.inv_state(inv_obs)) == 32
    assert len(env.inv_q_state(inv_obs)) == 16

def test_inv_q_state_keeps_the_last_social_event():
    env = make_env()
    inv_obs, _ = env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })
    before = env.inv_q_state(inv_obs)
    env.accuse(1)
    env.soc_reac()
    after = env.inv_q_state(env.get_inv_obs(inv_obs["market"]))
    assert before[9:11] == (-1, -1)
    assert after[9:11] == (StatType.ACCUSE.value, 0)

def test_trd_q_state_is_compact():
    env = make_env()
    env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })
    assert len(env.trd_state(0)) == 15
    assert len(env.trd_q_state(0)) == 6

def test_trd_q_state_is_role_blind():
    env = make_env()

    env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.CONCEAL,
        2: TraderAction.SIGNAL,
    })

    for trader_id in TRADER_IDS:
        state = env.trd_q_state(trader_id)

        assert isinstance(state, tuple)
        assert len(state) == 6

def test_trd_q_state_is_role_blind():
    env_a = make_env()
    env_b = make_env()

    actions = {
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    }

    env_a.step(actions)
    env_b.step(actions)

    # Same observable history, different hidden-role assignment.
    env_b.roles = {
        0: Role.FRAUDSTER,
        1: Role.TRICKSTER,
        2: Role.INNOCENT,
    }

    for trader_id in TRADER_IDS:
        state_a = env_a.trd_q_state(trader_id)
        state_b = env_b.trd_q_state(trader_id)

        assert isinstance(state_a, tuple)
        assert isinstance(state_b, tuple)
        assert len(state_a) == 6
        assert len(state_b) == 6
        assert state_a == state_b

def test_focus_consumes_charge_not_budget():
    env = make_env()

    env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })

    budget_before = env.rem_budget
    charges_before = env.focus_charges

    env.inv_act((InvAction.FOCUS, 1))

    assert env.rem_budget == budget_before
    assert env.focus_charges == charges_before - 1
    assert env.focus_target == 1

def test_focus_is_cleared_after_next_market_tick():
    env = make_env()

    env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })

    env.inv_act((InvAction.FOCUS, 1))

    env.step({
        0: TraderAction.NORMAL,
        1: TraderAction.NORMAL,
        2: TraderAction.NORMAL,
    })

    assert env.focus_target is None