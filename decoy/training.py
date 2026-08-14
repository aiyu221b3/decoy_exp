from collections import Counter

import numpy as np

from decoy.environment import DecoyEnv, InvAction, TRADER_IDS, Role, TraderAction
from decoy.q_learning import QLearning
from decoy.runner import run_investigator_episode
from decoy.monte_carlo import MonteCarlo
from decoy.agent import QAgent
from decoy.multi_runner import run_multi
from decoy.agent import DualAgent, TrickAgent


def inv_action_space():
    return [
        (kind, target)
        for kind in (
            InvAction.OBSERVE,
            InvAction.FOCUS,
            InvAction.ACCUSE,
            InvAction.VOTE,
        )
        for target in TRADER_IDS
    ]


def train_q_investigator(
        n_episodes,
        seed=None,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        max_steps=20,
        budget=10,
):
    if n_episodes <= 0:
        raise ValueError("need at least one episode, bestie")

    seed_seq = np.random.SeedSequence(seed)
    env_seed, learner_seed = seed_seq.spawn(2)

    env = DecoyEnv(
        rng=np.random.default_rng(env_seed),
        max_steps=max_steps,
        budget=budget,
    )
    learner = QLearning(
        actions=inv_action_space(),
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        rng=np.random.default_rng(learner_seed),
    )

    history = []

    def q_policy(state, valid_actions):
        return learner.act(
            state,
            available_actions=valid_actions,
        )

    for episode_no in range(n_episodes):
        episode, outcome = run_investigator_episode(
            env,
            q_policy,
            state_fn=env.inv_q_state,
        )

        for transition in episode.transitions:
            learner.update(
                state=transition.observation,
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_observation,
                done=transition.done,
                next_actions=transition.next_actions,
            )

        actions = [
            transition.action[0]
            for transition in episode.transitions
        ]

        history.append({
            "episode": episode_no,
            "reward": sum(
                transition.reward
                for transition in episode.transitions
            ),
            "outcome": outcome,
            "observations": actions.count(InvAction.OBSERVE),
            "accusations": actions.count(InvAction.ACCUSE),
            "votes": actions.count(InvAction.VOTE),
            "length": len(episode),
            "remaining_budget": env.rem_budget,
        })

    outcomes = Counter(row["outcome"] for row in history)

    summary = {
        "mean_reward": float(np.mean([
            row["reward"]
            for row in history
        ])),
        "fraudster_catch_rate": (
            outcomes["fraudster_caught"] / n_episodes
        ),
        "innocent_catch_rate": (
            outcomes["innocent_caught"] / n_episodes
        ),
        "trickster_catch_rate": (
            outcomes["trickster_caught"] / n_episodes
        ),
        "timeout_rate": outcomes[None] / n_episodes,
        "mean_observations": float(np.mean([
            row["observations"]
            for row in history
        ])),
        "mean_accusations": float(np.mean([
            row["accusations"]
            for row in history
        ])),
        "mean_votes": float(np.mean([
            row["votes"]
            for row in history
        ])),
        "mean_episode_length": float(np.mean([
            row["length"]
            for row in history
        ])),
        "mean_remaining_budget": float(np.mean([
            row["remaining_budget"]
            for row in history
        ])),
        "q_states": len(learner.q),
    }

    return learner, history, summary

def train_mc_investigator(
        n_episodes,
        seed=None,
        gamma=0.99,
        epsilon=0.1,
        max_steps=20,
        budget=10,
):
    if n_episodes <= 0:
        raise ValueError("need at least one episode, bestie")
    seed_seq = np.random.SeedSequence(seed)
    env_seed, learner_seed = seed_seq.spawn(2)
    env = DecoyEnv(
        rng=np.random.default_rng(env_seed),
        max_steps=max_steps,
        budget=budget,
    )
    learner = MonteCarlo(
        actions=inv_action_space(),
        gamma=gamma,
        rng=np.random.default_rng(learner_seed),
    )
    history = []

    def mc_policy(state, valid_actions):
        return learner.act(
            state,
            epsilon=epsilon,
            available_actions=valid_actions,
        )

    for episode_no in range(n_episodes):
        episode, outcome = run_investigator_episode(
            env,
            mc_policy,
            state_fn=env.inv_q_state,
        )
        learner.update(episode)
        actions = [
            transition.action[0]
            for transition in episode.transitions
        ]
        history.append({
            "episode": episode_no,
            "reward": sum(
                transition.reward
                for transition in episode.transitions
            ),
            "outcome": outcome,
            "observations": actions.count(InvAction.OBSERVE),
            "accusations": actions.count(InvAction.ACCUSE),
            "votes": actions.count(InvAction.VOTE),
            "length": len(episode),
            "remaining_budget": env.rem_budget,
        })

    outcomes = Counter(row["outcome"] for row in history)
    summary = {
        "mean_reward": float(np.mean([
            row["reward"]
            for row in history
        ])),
        "fraudster_catch_rate": (
            outcomes["fraudster_caught"] / n_episodes
        ),
        "innocent_catch_rate": (
            outcomes["innocent_caught"] / n_episodes
        ),
        "trickster_catch_rate": (
            outcomes["trickster_caught"] / n_episodes
        ),
        "timeout_rate": outcomes[None] / n_episodes,
        "mean_observations": float(np.mean([
            row["observations"]
            for row in history
        ])),
        "mean_accusations": float(np.mean([
            row["accusations"]
            for row in history
        ])),
        "mean_votes": float(np.mean([
            row["votes"]
            for row in history
        ])),
        "mean_episode_length": float(np.mean([
            row["length"]
            for row in history
        ])),
        "mean_remaining_budget": float(np.mean([
            row["remaining_budget"]
            for row in history
        ])),
        "q_states": len(learner.q),
    }
    return learner, history, summary

def eval_inv(
        agent,
        n_eps,
        seed=None,
        max_steps=20,
        budget=10,
        state_fn=None,
):
    if n_eps <= 0:
        raise ValueError("need at least one eval episode, bestie")
    env = DecoyEnv(
        rng=np.random.default_rng(seed),
        max_steps=max_steps,
        budget=budget,
    )
    state_fn = env.inv_q_state if state_fn is None else state_fn
    hist = []
    def pol(state, valid_actions):
        return agent.act(
            state,
            epsilon=0.0,
            available_actions=valid_actions,
        )
    for ep_no in range(n_eps):
        episode, outcome = run_investigator_episode(
            env,
            pol,
            state_fn=state_fn,
        )
        acts = [transition.action[0] for transition in episode.transitions]
        hist.append({
            "episode": ep_no,
            "reward": sum(
                transition.reward
                for transition in episode.transitions
            ),
            "outcome": outcome,
            "observations": acts.count(InvAction.OBSERVE),
            "accusations": acts.count(InvAction.ACCUSE),
            "votes": acts.count(InvAction.VOTE),
            "length": len(episode),
            "remaining_budget": env.rem_budget,
        })
    out = Counter(row["outcome"] for row in hist)
    summ = {
        "mean_reward": float(np.mean([row["reward"] for row in hist])),
        "fraudster_catch_rate": out["fraudster_caught"] / n_eps,
        "innocent_catch_rate": out["innocent_caught"] / n_eps,
        "trickster_catch_rate": out["trickster_caught"] / n_eps,
        "timeout_rate": out[None] / n_eps,
        "mean_observations": float(np.mean([
            row["observations"]
            for row in hist
        ])),
        "mean_accusations": float(np.mean([
            row["accusations"]
            for row in hist
        ])),
        "mean_votes": float(np.mean([
            row["votes"]
            for row in hist
        ])),
        "mean_episode_length": float(np.mean([
            row["length"]
            for row in hist
        ])),
        "mean_remaining_budget": float(np.mean([
            row["remaining_budget"]
            for row in hist
        ])),
        "q_states": len(agent.q),
    }
    return hist, summ

def trd_space():
    return list(TraderAction)

def mk_role(env, agents):
    if set(agents) != set(Role):
        raise ValueError("need one agent for each role")
    return {
        i: agents[env.roles[i]]
        for i in TRADER_IDS
    }

def mk_q_trd(learners, eps=None):
    if set(learners) != set(Role):
        raise ValueError("need one learner for each role")
    agents = {
        role: QAgent(learner, eps=eps)
        for role, learner in learners.items()
    }
    return lambda env: mk_role(env, agents)

def mk_inv(agent):
    def pol(state, acts):
        return agent.act(
            state,
            epsilon=0.0,
            available_actions=acts,
        )
    return pol

def tr_trd(
        n_eps,
        inv,
        seed=None,
        alpha=0.1,
        gamma=0.99,
        eps=0.1,
        max_steps=20,
        budget=10,
):
    if n_eps <= 0:
        raise ValueError("need at least one episode, Sensei")
    seq = np.random.SeedSequence(seed)
    seeds = seq.spawn(len(Role) + 1)
    env = DecoyEnv(
        rng=np.random.default_rng(seeds[0]),
        max_steps=max_steps,
        budget=budget,
    )
    learners = {
        role: QLearning(
            actions=trd_space(),
            alpha=alpha,
            gamma=gamma,
            epsilon=eps,
            rng=np.random.default_rng(role_seed),
        )
        for role, role_seed in zip(Role, seeds[1:])
    }
    hist = []
    pol = mk_inv(inv)
    for ep_no in range(n_eps):
        _, trd_eps, outcome = run_multi(
            env,
            pol,
            mk_q_trd(learners, eps=eps),
            state_fn=env.inv_q_state,
            trd_state_fn=env.trd_q_state,
        )
        role_eps = {
            env.roles[i]: trd_eps[i]
            for i in TRADER_IDS
        }
        for role, ep in role_eps.items():
            for trans in ep.transitions:
                learners[role].update(
                    state=trans.observation,
                    action=trans.action,
                    reward=trans.reward,
                    next_state=trans.next_observation,
                    done=trans.done,
                    next_actions=trans.next_actions,
                )
        hist.append({
            "episode": ep_no,
            "outcome": outcome,
            "inn_rew": sum(
                trans.reward
                for trans in role_eps[Role.INNOCENT].transitions
            ),
            "fraud_rew": sum(
                trans.reward
                for trans in role_eps[Role.FRAUDSTER].transitions
            ),
            "trick_rew": sum(
                trans.reward
                for trans in role_eps[Role.TRICKSTER].transitions
            ),
        })
    out = Counter(row["outcome"] for row in hist)
    summ = {
        "inn_rew": float(np.mean([row["inn_rew"] for row in hist])),
        "fraud_rew": float(np.mean([row["fraud_rew"] for row in hist])),
        "trick_rew": float(np.mean([row["trick_rew"] for row in hist])),
        "fraudster_catch_rate": out["fraudster_caught"] / n_eps,
        "innocent_catch_rate": out["innocent_caught"] / n_eps,
        "trickster_catch_rate": out["trickster_caught"] / n_eps,
        "timeout_rate": out[None] / n_eps,
    }
    return learners, hist, summ

def mk_dual(
    env,
    mrk,
    soc,
    eps=None,
    trick_advisor=None,
    episode_no=0,
    total_episodes=0,
):
    agents = {}
    innocent_id = next(
        i for i in TRADER_IDS
        if env.roles[i] == Role.INNOCENT
    )
    fraudster_id = next(
        i for i in TRADER_IDS
        if env.roles[i] == Role.FRAUDSTER
    )
    faction_targets = {
        "investigator": (
            innocent_id,
            fraudster_id,
        ),
        "criminal": (
            fraudster_id,
            innocent_id,
        ),
    }
    for role in Role:
        if role == Role.TRICKSTER:
            agents[role] = TrickAgent(
                mrk[role],
                soc[role],
                eps=eps,
                advisor=trick_advisor,
                episode_no=episode_no,
                total_episodes=total_episodes,
                faction_targets=faction_targets,
            )
        else:
            agents[role] = DualAgent(
                mrk[role],
                soc[role],
                eps,
            )
    return {
        i: agents[env.roles[i]]
        for i in TRADER_IDS
    }

def mk_dual_q(
    mrk,
    soc,
    eps=None,
    trick_advisor=None,
    episode_no=0,
    total_episodes=0,
):
    return lambda env: mk_dual(
        env,
        mrk,
        soc,
        eps=eps,
        trick_advisor=trick_advisor,
        episode_no=episode_no,
        total_episodes=total_episodes,
    )