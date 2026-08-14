from collections import Counter
import numpy as np
from decoy.environment import DecoyEnv, InvAction, Role, TRADER_IDS, StatType, TraderAction
from decoy.multi_runner import run_multi
from decoy.q_learning import QLearning
from decoy.training import inv_action_space, mk_inv, mk_q_trd, trd_space, mk_dual_q
from decoy.win_register import WinRegister

def scheduled_epsilon(
    start,
    end,
    episode,
    total_episodes,
):
    if total_episodes <= 1:
        return end

    progress = min(
        episode / (0.8 * total_episodes),
        1.0,
    )

    return start + progress * (end - start)

def get_eps(env, trd_eps):
    return {
        env.roles[i]: trd_eps[i]
        for i in TRADER_IDS
    }

def mk_row(
                ep_no,
                env,
                inv_ep,
                trd_episodes,
                outcome,
            ):
    role_eps = get_eps(env, trd_episodes)
    acts = [
        trans.action[0]
        for trans in inv_ep.transitions
    ]
    return {
        "episode": ep_no,
        "outcome": outcome,
        "inv_rew": sum(
            trans.reward
            for trans in inv_ep.transitions
        ),
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
        "observations": acts.count(InvAction.OBSERVE),
        "accusations": acts.count(InvAction.ACCUSE),
        "votes": acts.count(InvAction.VOTE),
        "length": len(inv_ep),
        "remaining_budget": env.rem_budget,
        "focuses": acts.count(InvAction.FOCUS),
    }

def mk_summ(hist):
    out = Counter(row["outcome"] for row in hist)
    n_eps = len(hist)
    return {
        "inv_rew": float(np.mean([row["inv_rew"] for row in hist])),
        "inn_rew": float(np.mean([row["inn_rew"] for row in hist])),
        "fraud_rew": float(np.mean([row["fraud_rew"] for row in hist])),
        "trick_rew": float(np.mean([row["trick_rew"] for row in hist])),
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
        "mean_focuses": float(np.mean([
            row["focuses"]
            for row in hist
        ])),
    }

def ev_multi(
        inv,
        mrk,
        soc=None,
        n_eps=1,
        seed=None,
        max_steps=20,
        budget=10,
        win_register=None,
):
    if n_eps <= 0:
        raise ValueError("need at least one episode, Sensei")
    env = DecoyEnv(
        rng=np.random.default_rng(seed),
        max_steps=max_steps,
        budget=budget,
    )
    hist = []
    for ep_no in range(n_eps):
        if soc is None:
            trd_mk = mk_q_trd(
                mrk,
                eps=0.0,
            )
        else:
            if win_register is None:
                trick_episode = ep_no
                total_episodes = n_eps
                advisor = None
            else:
                trick_episode = (
                    win_register.episodes_seen
                    + ep_no
                )
                total_episodes = (
                    win_register.episodes_seen
                    + n_eps
                )
                advisor = win_register
            trd_mk = mk_dual_q(
                mrk,
                soc,
                eps=0.0,
                trick_advisor=advisor,
                episode_no=trick_episode,
                total_episodes=total_episodes,
            )
        inv_ep, trd_episodes, outcome = run_multi(
            env,
            mk_inv(inv),
            trd_mk,
            state_fn=env.inv_q_state,
            trd_state_fn=env.trd_q_state,
        )
        hist.append(mk_row(
            ep_no,
            env,
            inv_ep,
            trd_episodes,
            outcome,
        ))
    return hist, mk_summ(hist)

def tr_multi(
        n_eps,
        seed=None,
        max_steps=20,
        budget=10,
        alpha=0.1,
        alpha_inv=None,
        alpha_trd=None,
        gamma=0.99,
        inv_eps=0.1,
        trd_eps=0.1,
        epsilon_decay=False,
        epsilon_start=0.2,
        epsilon_end=0.02,
        adaptive_inv_alpha=False,
        alpha_inv_min=0.03,
        alpha_inv_max=0.12,
        alpha_inv_step=0.005,
        performance_window=100,
        performance_tolerance=0.1,
        warmup_episodes=0,
):
    if n_eps <= 0:
        raise ValueError("need at least one episode -v-")
    if warmup_episodes < 0:
        raise ValueError("warmup_episodes must be non-negative")
    seq = np.random.SeedSequence(seed)
    seeds = seq.spawn(8)
    env = DecoyEnv(
        rng=np.random.default_rng(seeds[0]),
        max_steps=max_steps,
        budget=budget,
    )
    inv = QLearning(
        actions=inv_action_space(),
        alpha=alpha if alpha_inv is None else alpha_inv,
        gamma=gamma,
        epsilon=inv_eps,
        rng=np.random.default_rng(seeds[1]),
    )
    mrk = {
        role: QLearning(
            actions=trd_space(),
            alpha=alpha if alpha_trd is None else alpha_trd,
            gamma=gamma,
            epsilon=trd_eps,
            rng=np.random.default_rng(role_seed),
        )
        for role, role_seed in zip(Role, seeds[2:5])
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
            alpha=alpha if alpha_trd is None else alpha_trd,
            gamma=gamma,
            epsilon=trd_eps,
            rng=np.random.default_rng(role_seed),
        )
        for role, role_seed in zip(Role, seeds[5:8])
    }
    register = WinRegister()
    hist = []
    recent_inv_rewards = []

    def pol(state, acts):
        return inv.act(
            state,
            available_actions=acts,
        )

    for _ in range(warmup_episodes):
        warmup_trd_mk = mk_dual_q(
            mrk,
            soc,
            eps=0.0,
            trick_advisor=None,
        )

        inv_ep, _, _ = run_multi(
            env,
            pol,
            warmup_trd_mk,
            state_fn=env.inv_q_state,
            trd_state_fn=env.trd_q_state,
        )

        for trans in inv_ep.transitions:
            inv.update(
                state=trans.observation,
                action=trans.action,
                reward=trans.reward,
                next_state=trans.next_observation,
                done=trans.done,
                next_actions=trans.next_actions,
            )
    
    for ep_no in range(n_eps):

        if epsilon_decay:
            current_inv_eps = scheduled_epsilon(
                inv_eps,
                epsilon_end,
                ep_no,
                n_eps,
            )
            current_trd_eps = scheduled_epsilon(
                epsilon_start,
                epsilon_end,
                ep_no,
                n_eps,
            )
        else:
            current_inv_eps = inv_eps
            current_trd_eps = trd_eps

        inv.epsilon = current_inv_eps

        predicted_side = register.should_commit(
            ep_no,
            n_eps,
        )
        trd_mk = mk_dual_q(
            mrk,
            soc,
            eps=current_trd_eps,
            trick_advisor=register,
            episode_no=ep_no,
            total_episodes=n_eps,
        )
        if (
            adaptive_inv_alpha
            and len(recent_inv_rewards) >= 2 * performance_window
        ):
            previous_window = recent_inv_rewards[
                :performance_window
            ]

            recent_window = recent_inv_rewards[
                performance_window:
            ]

            previous_mean = np.mean(previous_window)
            recent_mean = np.mean(recent_window)

            delta = recent_mean - previous_mean

            if delta < -performance_tolerance:
                inv.alpha = min(
                    inv.alpha + alpha_inv_step,
                    alpha_inv_max,
                )
            elif delta > performance_tolerance:
                inv.alpha = max(
                    inv.alpha - alpha_inv_step,
                    alpha_inv_min,
                )
        inv_ep, trd_episodes, outcome = run_multi(
            env,
            pol,
            trd_mk,
            state_fn=env.inv_q_state,
            trd_state_fn=env.trd_q_state,
        )
        register.record(outcome)
        episode_reward = sum(
            trans.reward
            for trans in inv_ep.transitions
        )
        if adaptive_inv_alpha:
            recent_inv_rewards.append(episode_reward)

            if len(recent_inv_rewards) > 2 * performance_window:
                recent_inv_rewards.pop(0)
        
        role_eps = get_eps(env, trd_episodes)
        trick_id = next(
            i
            for i in TRADER_IDS
            if env.roles[i] == Role.TRICKSTER
        )
        trick_ep = role_eps[Role.TRICKSTER]
        had_social_action = any(
            isinstance(trans.action, tuple)
            and isinstance(trans.action[0], StatType)
            for trans in trick_ep.transitions
        )
        committed = (
            predicted_side is not None
            and had_social_action
        )
        trick_reward = 0.0
        if not committed:
            if outcome in (
                "fraudster_caught",
                "innocent_caught",
            ):
                trick_reward += 1.0
            if trick_id in env.acc_hist:
                trick_reward -= 1.0
            trick_reward += register.confusion_bonus()
        else:
            if predicted_side == "investigator":
                if outcome == "fraudster_caught":
                    trick_reward += 5.0
                elif outcome in (
                    "innocent_caught",
                    "trickster_caught",
                ):
                    trick_reward -= 5.0
            elif predicted_side == "criminal":
                if outcome == "innocent_caught":
                    trick_reward += 5.0
                elif outcome in (
                    "fraudster_caught",
                    "trickster_caught",
                ):
                    trick_reward -= 5.0
            if trick_id in env.acc_hist:
                trick_reward -= 1.0
        trick_ep.transitions[-1] = trick_ep.transitions[-1].__class__(
            observation=trick_ep.transitions[-1].observation,
            action=trick_ep.transitions[-1].action,
            reward=trick_reward,
            next_observation=trick_ep.transitions[-1].next_observation,
            done=trick_ep.transitions[-1].done,
            next_actions=trick_ep.transitions[-1].next_actions,
        )
        for trans in inv_ep.transitions:
            inv.update(
                state=trans.observation,
                action=trans.action,
                reward=trans.reward,
                next_state=trans.next_observation,
                done=trans.done,
                next_actions=trans.next_actions,
            )
        for role, ep in role_eps.items():
            for trans in ep.transitions:
                if isinstance(trans.action, TraderAction):
                    mrk[role].update(
                        state=trans.observation,
                        action=trans.action,
                        reward=trans.reward,
                        next_state=trans.next_observation,
                        done=trans.done,
                        next_actions=trans.next_actions,
                    )
                elif (
                    isinstance(trans.action, tuple)
                    and isinstance(trans.action[0], StatType)
                ):
                    soc[role].update(
                        state=trans.observation,
                        action=trans.action,
                        reward=trans.reward,
                        next_state=trans.next_observation,
                        done=trans.done,
                        next_actions=trans.next_actions,
                    )
        hist.append(mk_row(
            ep_no,
            env,            
            inv_ep,
            trd_episodes,

            outcome,
        ))
    return (
        inv,
        mrk,
        soc,
        hist,
        mk_summ(hist),
        register,
    )