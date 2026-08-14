from decoy.environment import InvAction, TraderAction, TRADER_IDS
from decoy.episode import Episode, Transition
from decoy.runner import available_inv_actions, mk_fix, trd_acts
from decoy.agent import DualAgent, TrickAgent

TRD_ACTIONS = tuple(TraderAction)

def get_trd(env, agents, state_fn = None):
    state_fn = env.trd_state if state_fn is None else state_fn
    states = {
        i: state_fn(i)
        for i in TRADER_IDS
    }
    acts = trd_acts(env, agents, states)
    return states, acts

def get_soc(env, agents):
    acts = {}
    for i in TRADER_IDS:
        agent = agents[i]
        if not isinstance(agent, DualAgent):
            continue
        state = env.soc_state(i)
        space = env.soc_space(i)
        action = agent.soc_act(state, space)
        if action not in space:
            raise ValueError("social action escaped containment")
        acts[i] = (state, action)
    return acts

def do_soc(env, agents):
    trans = {}
    for i, (state, action) in get_soc(env, agents).items():
        statement = env.do_soc(i, action)
        trans[i] = (state, action, env.soc_rew(i, statement))
    return trans

def add_trd(eps, states, acts, next_states, rews, done):
    for i in TRADER_IDS:
        eps[i].add(Transition(
            observation=states[i],
            action=acts[i],
            reward=rews[i],
            next_observation=next_states[i],
            done=done,
            next_actions=() if done else TRD_ACTIONS,
        ))

def run_multi(
        env,
        inv_pol,
        trd_agents=None,
        state_fn=None,
        trd_state_fn=None,
):
    env.reset()
    state_fn = env.inv_state if state_fn is None else state_fn
    if trd_agents is None:
        trd_agents = mk_fix(env)
    elif callable(trd_agents):
        trd_agents = trd_agents(env)
    if set(trd_agents) != set(TRADER_IDS):
        raise ValueError("need one trader agent per trader, gang")
    for agent in trd_agents.values():
        agent.reset()
    inv_ep = Episode(transitions=[])
    trd_eps = {
        i: Episode(transitions=[])
        for i in TRADER_IDS
    }
    outcome = None
    trd_states, trd_actions = get_trd(env, trd_agents, trd_state_fn)
    inv_obs, done = env.step(trd_actions)
    if done:
        rews = {
            i: env.trd_rew(i, None)
            for i in TRADER_IDS
        }
        add_trd(
            trd_eps,
            trd_states,
            trd_actions,
            {i: None for i in TRADER_IDS},
            rews,
            True,
        )
        return inv_ep, trd_eps, outcome
    while not env.done:
        inv_state = state_fn(inv_obs)
        inv_actions = available_inv_actions(env)
        if not inv_actions:
            env.done = True
            outcome = None
            break
        inv_action = inv_pol(inv_state, inv_actions)
        if inv_action not in inv_actions:
            raise ValueError("investigator policy picked an action that ain't on the menu")
        kind, _ = inv_action
        result = env.inv_act(inv_action)
        if kind == InvAction.VOTE:
            outcome = result
            inv_ep.add(Transition(
                observation=inv_state,
                action=inv_action,
                reward=env.inv_reward(outcome),
                next_observation=None,
                done=True,
                next_actions=(),
            ))
            rews = {
                i: env.trd_rew(i, outcome)
                for i in TRADER_IDS
            }
            add_trd(
                trd_eps,
                trd_states,
                trd_actions,
                {i: None for i in TRADER_IDS},
                rews,
                True,
            )
            break
        if kind == InvAction.ACCUSE:
            soc_trans = do_soc(env, trd_agents)
        else:
            soc_trans = {}
        if env.rem_budget == 0:
            env.done = True
            inv_ep.add(Transition(
                observation=inv_state,
                action=inv_action,
                reward=env.inv_reward(None),
                next_observation=None,
                done=True,
                next_actions=(),
            ))
            rews = {
                i: env.trd_rew(i, None)
                for i in TRADER_IDS
            }
            add_trd(
                trd_eps,
                trd_states,
                trd_actions,
                {i: None for i in TRADER_IDS},
                rews,
                True,
            )
            break
        next_trd_states, next_trd_actions = get_trd(env, trd_agents, trd_state_fn)
        add_trd(
            trd_eps,
            trd_states,
            trd_actions,
            next_trd_states,
            {i: 0 for i in TRADER_IDS},
            False,
        )
        for i, (state, action, reward) in soc_trans.items():
            trd_eps[i].add(Transition(
                observation=state,
                action=action,
                reward=reward,
                next_observation=None,
                done=True,
                next_actions=(),
            ))
        next_inv_obs, done = env.step(next_trd_actions)
        next_inv_state = None if done else state_fn(next_inv_obs)
        next_inv_actions = () if done else tuple(available_inv_actions(env))
        inv_ep.add(Transition(
            observation=inv_state,
            action=inv_action,
            reward=env.inv_reward(None) if done else 0,
            next_observation=next_inv_state,
            done=done,
            next_actions=next_inv_actions,
        ))
        if done:
            outcome = None
            rews = {
                i: env.trd_rew(i, None)
                for i in TRADER_IDS
            }
            add_trd(
                trd_eps,
                next_trd_states,
                next_trd_actions,
                {i: None for i in TRADER_IDS},
                rews,
                True,
            )
            break
        trd_states = next_trd_states
        trd_actions = next_trd_actions
        inv_obs = next_inv_obs
    return inv_ep, trd_eps, outcome