from decoy.agent import FixAgent
from decoy.environment import InvAction, Role, TraderAction, TRADER_IDS
from decoy.episode import Episode, Transition

def mk_fix(env):
    acts = {
        Role.INNOCENT: TraderAction.NORMAL,
        Role.FRAUDSTER: TraderAction.CONCEAL,
        Role.TRICKSTER: TraderAction.SIGNAL,
    }
    return {
        i: FixAgent(acts[env.roles[i]])
        for i in TRADER_IDS
    }

def trd_acts(env, agents, states=None):
    if set(agents) != set(TRADER_IDS):
        raise ValueError("need one trader agent per trader, Gojo")
    if states is None:
        states = {
            i: env.trd_state(i)
            for i in TRADER_IDS
        }
    acts = {}
    for i in TRADER_IDS:
        act = agents[i].act(states[i])
        if not isinstance(act, TraderAction):
            raise ValueError("trader picked a market move from another universe")
        acts[i] = act
    return acts

def default_trader_actions(env):
    return trd_acts(env, mk_fix(env))

def available_inv_actions(env):
    if env.done:
        return []
    acts = []
    if env.rem_budget > 0:
        for target in TRADER_IDS:
            acts.append((InvAction.OBSERVE, target))
            acts.append((InvAction.ACCUSE, target))
        if env.accus is not None:
            acts.append((InvAction.VOTE, env.accus))
    if env.focus_charges > 0:
        for target in TRADER_IDS:
            acts.append((InvAction.FOCUS, target))
    return acts

def run_investigator_episode(
        env,
        investigator_policy,
        trader_policy=default_trader_actions,
        state_fn=None,
):
    env.reset()
    state_fn = env.inv_state if state_fn is None else state_fn
    episode = Episode(transitions=[])
    outcome = None
    inv_obs, done = env.step(trader_policy(env))
    if done:
        return episode, outcome
    while not env.done:
        state = state_fn(inv_obs)
        valid_actions = available_inv_actions(env)
        if not valid_actions:
            env.done = True
            break
        action = investigator_policy(state, valid_actions)
        if action not in valid_actions:
            raise ValueError("investigator policy picked an action that ain't on the menu")
        kind, _ = action
        result = env.inv_act(action)
        if kind == InvAction.VOTE:
            outcome = result
            episode.add(Transition(
                observation=state,
                action=action,
                reward=env.inv_reward(outcome),
                next_observation=None,
                done=True,
                next_actions=(),
            ))
            break
        if kind == InvAction.ACCUSE:
            env.soc_reac()
        if env.rem_budget == 0:
            env.done = True
            episode.add(Transition(
                observation=state,
                action=action,
                reward=env.inv_reward(None),
                next_observation=None,
                done=True,
                next_actions=(),
            ))
            break
        next_inv_obs, done = env.step(trader_policy(env))
        next_state = None if done else state_fn(next_inv_obs)
        next_actions = () if done else tuple(available_inv_actions(env))
        episode.add(Transition(
            observation=state,
            action=action,
            reward=env.inv_reward(None) if done else 0,
            next_observation=next_state,
            done=done,
            next_actions=next_actions,
        ))
        if done:
            break
        inv_obs = next_inv_obs
    return episode, outcome