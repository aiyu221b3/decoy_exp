from enum import Enum
import numpy as np
from decoy.ou_process import OUProcess
from dataclasses import dataclass

class Role(Enum):
    INNOCENT = 0
    FRAUDSTER = 1
    TRICKSTER = 2

class TraderAction(Enum):
    NORMAL = 0
    CONCEAL = 1
    SIGNAL = 2

class StatType(Enum):
    NEUTRAL = 0
    DEFEND = 1
    ACCUSE = 2
    SUPPORT = 3

class InvAction(Enum):
    OBSERVE = 0
    FOCUS = 1
    STATEMENT = 2
    ACCUSE = 3
    VOTE = 4

@dataclass
class Statement:
    speaker: int
    kind: StatType
    target: int | None = None

TRADER_IDS = (0, 1, 2)
INVESTIGATOR = 3

class DecoyEnv:
    def __init__(
            self,
            rng = None,
            n_trd = 3,
            max_steps = 20,
            budget = 10,
    ):
        self.rng = (
            rng
            if rng is not None 
            else np.random.default_rng()
        )
        self.n_trd = n_trd
        self.max_steps = max_steps
        self.budget = budget
        self.roles = None
        self.step_count = 0
        self.rem_budget = None
        self.done = False
        self.ou = None
        self.x = None
        self.conc_sc = 0.5
        self.sig_sc = 1.5
        self.obs_sig = 0.5
        self.history = []
        self.stats = None
        self.obs_hist = None
        self.accus = None
        self.acc_hist = []
        self.inspections = None
        self.focus_charges_max = 2
        self.focus_noise_scale = 0.5
        self.focus_charges = self.focus_charges_max
        self.focus_target = None

    def reset(self):
        self.step_count = 0
        self.rem_budget = self.budget
        self.focus_charges = self.focus_charges_max
        self.focus_target = None
        self.history = []
        self.done = False
        self.accus = None
        self.acc_hist = []
        self.stats = {
            i:{
            "statements": 0,
            "accusations": 0,
            "defenses": 0,
            "supports": 0,
            "target_sw": 0,
            "l_target": None,
            "manip_ev": 0,
            }
            for i in TRADER_IDS
        }
        self.obs_hist = {
            i: []
            for i in TRADER_IDS
        }
        self.pub_obs = {
            i: 0.0
            for i in TRADER_IDS
        }
        self.inspections = {
            i: {
                "seen": False,
                "mean": 0.0,
                "trend": 0.0,
            }
            for i in TRADER_IDS
        }
        roles = [Role.INNOCENT, Role.FRAUDSTER, Role.TRICKSTER]
        self.rng.shuffle(roles)
        self.roles = {
            i: roles[i]
            for i in range(self.n_trd)
        }
        self.ou = {
            i: OUProcess(
                theta = 0.3,
                mu = 0.0,
                sigma = 1.0,
                dt = 0.01,
                rng = self.rng,
            )
            for i in TRADER_IDS
        }
        self.x = {
            i: 0.0
            for i in TRADER_IDS
        }
        return self.roles

    def step(self, actions):
        if self.done:
            raise RuntimeError("episode is donee")
        focused_target = self.focus_target
        for i in TRADER_IDS:
            self.x[i] = self.ou[i].step(self.x[i])
        observ = {
            i: self.observe(
                i,
                actions[i],
                focused=(i == focused_target),
            )
            for i in TRADER_IDS
        }
        self.focus_target = None
        self.pub_obs = observ.copy()
        for i in TRADER_IDS:
            self.obs_hist[i].append(observ[i])
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        inv_obs = self.get_inv_obs(observ)
        return inv_obs, self.done

    def observe(self, trader_id, action, focused=False):
        x = self.x[trader_id]

        if action == TraderAction.NORMAL:
            scale = 1.0
        elif action == TraderAction.CONCEAL:
            scale = self.conc_sc
        elif action == TraderAction.SIGNAL:
            scale = self.sig_sc
        else:
            raise ValueError("invalid trader act")

        noise_sig = (
                self.obs_sig * self.focus_noise_scale
                if focused
                else self.obs_sig
            )

        noise = self.rng.normal(0.0, noise_sig)

        return scale*x + noise

    def mk_stat(self, speaker, kind, target = None):
        if kind != StatType.NEUTRAL and target is None:
            raise ValueError("you need target, smh")
        if target is not None and target not in TRADER_IDS:
            raise ValueError("whatcha doing' dude.")
        statement = Statement(
            speaker = speaker,
            kind = kind,
            target = target,
        )
        
        self.stats[speaker]["statements"] += 1
        if kind == StatType.ACCUSE:
            self.stats[speaker]["accusations"] += 1
        if kind == StatType.DEFEND:
            self.stats[speaker]["defenses"] += 1
        if kind == StatType.SUPPORT:
            self.stats[speaker]["supports"] += 1
        # target switching
        if target is not None:
            l = self.stats[speaker]["l_target"]
            if l is not None and l != target:
                self.stats[speaker]["target_sw"] += 1
            self.stats[speaker]["l_target"] = target
        self.history.append(statement)
        return statement

    def get_prof(self, trader_id):
        s = self.stats[trader_id]
        n = s["statements"]

        if n == 0:
            return{
                "acc_rate": 0.0,
                "defen": 0.0,
                "supp_rt": 0.0,
                "target_sw_rt": 0.0,
                "manip_rate": 0.0,
            }
        return {
            "acc_rate": s["accusations"] / n,
            "defen": s["defenses"] / n,
            "supp_rt": s["supports"] / n,
            "target_sw_rt": s["target_sw"] / max(n-1, 1),
            "manip_rate": s["manip_ev"] / n,
        }

    def get_inv_obs(self, observ):
        return {
            "market": observ,
            "profiles": {
                i: self.get_prof(i)
                for i in TRADER_IDS
            },
            "inspections": {
                i: self.inspections[i].copy()
                for i in TRADER_IDS
            },
            "history": self.history[-5:],
            "budget": self.rem_budget,
            "step": self.step_count,
            "accusation": self.accus,
            "focus_charges": self.focus_charges,
            "focus_target": self.focus_target,
        }

    def disc_mrk(self, x):
        if x < -1.0:
            return -2
        if x < -0.25:
            return -1
        if x <= 0.25:
            return 0
        if x <= 1.0:
            return 1
        return 2

    def disc_prof(self, x):
        if x < 0.25:
            return 0
        if x < 0.6:
            return 1
        return 2

    def inv_state(self, inv_obs):
        state = []
        for i in TRADER_IDS:
            state.append(
                self.disc_mrk(inv_obs["market"][i])
            )
            prof = inv_obs["profiles"][i]
            state.extend([
                self.disc_prof(prof["acc_rate"]),
                self.disc_prof(prof["defen"]),
                self.disc_prof(prof["supp_rt"]),
                self.disc_prof(prof["target_sw_rt"]),
                self.disc_prof(prof["manip_rate"]),
            ])
            inspection = inv_obs["inspections"][i]
            state.extend([
                int(inspection["seen"]),
                self.disc_mrk(inspection["mean"]),
                self.disc_mrk(inspection["trend"]),
            ])
        state.append(
            min(inv_obs["budget"], 3)
        )
        state.append(
            -1
            if inv_obs["accusation"] is None
            else inv_obs["accusation"]
        )
        state.append(
            min(inv_obs["step"] // 5, 3)
        )
        state.append(inv_obs["focus_charges"])
        state.append(
            -1
            if inv_obs["focus_target"] is None
            else inv_obs["focus_target"]
        )
        return tuple(state)

    def inv_q_state(self, inv_obs):
        state = []
        for i in TRADER_IDS:
            inspection = inv_obs["inspections"][i]
            state.extend([
                self.disc_mrk(inv_obs["market"][i]),
                int(inspection["seen"]),
                self.disc_mrk(inspection["mean"]),
            ])
        if inv_obs["history"]:
            last = inv_obs["history"][-1]
            state.extend([
                last.kind.value,
                -1 if last.target is None else last.target,
            ])
        else:
            state.extend([-1, -1])
        state.append(
            -1
            if inv_obs["accusation"] is None
            else inv_obs["accusation"]
        )
        state.append(min(inv_obs["budget"], 3))
        state.append(min(inv_obs["step"] // 5, 3))
        state.append(inv_obs["focus_charges"])
        state.append(
            -1
            if inv_obs["focus_target"] is None
            else inv_obs["focus_target"]
        )
        return tuple(state)
    
    def inv_act(self, action):
        kind, target = action
        if not isinstance(kind, InvAction):
            raise ValueError("invalid investigator action")
        if target not in TRADER_IDS:
            raise ValueError("invalid target")
        if kind == InvAction.OBSERVE:
            return self.inv_observe(target)
        if kind == InvAction.FOCUS:
            return self.focus(target)
        if kind == InvAction.ACCUSE:
            return self.accuse(target)
        if kind == InvAction.VOTE:
            return self.vote(target)
        if kind == InvAction.STATEMENT:
            raise NotImplementedError(
                "investigator statements are not implemented yet"
            )
        raise ValueError("invalid investigator action")

    def focus(self, target):
        if target not in TRADER_IDS:
            raise ValueError("invalid target")
        if self.focus_charges <= 0:
            raise RuntimeError("no focus charges left")
        self.focus_charges -= 1
        self.focus_target = target
        return target

    def inv_observe(self, target):
        if target not in TRADER_IDS:
            raise ValueError("invalid target")
        if self.rem_budget <= 0:
            raise RuntimeError("no budget left")

        history = self.obs_hist[target][-5:]
        if not history:
            raise RuntimeError(
                "nothing to inspect yet, gimme one market tick first"
            )
        self.rem_budget -= 1
        self.inspections[target] = {
            "seen": True,
            "mean": float(np.mean(history)),
            "trend": float(history[-1] - history[0]),
        }

        return history

    def accuse(self, target):
        if target not in TRADER_IDS:
            raise ValueError("invalid target")
        if self.rem_budget <= 0:
            raise RuntimeError("naurr budget left")
        self.rem_budget -= 1
        self.accus = target
        self.acc_hist.append(target)
        return target

    def vote(self, target):
        if target not in TRADER_IDS:
            raise ValueError("invalid target")
        if self.accus is None:
            raise RuntimeError("naurr accus")
        if target != self.accus:
            raise ValueError("vote target needs to match")
        if self.rem_budget <= 0:
            raise RuntimeError("brokeee T-T")

        self.rem_budget -= 1
        result = self.outcome(target)
        self.done = True
        return result

    def outcome(self, target):
        role = self.roles[target]
        if role == Role.FRAUDSTER:
            return "fraudster_caught"
        if role == Role.INNOCENT:
            return "innocent_caught"
        return "trickster_caught"

    def inv_reward(self, outcome):
        rewards = {
            "fraudster_caught": 10,
            "innocent_caught": -10,
            "trickster_caught": -3,
            None: -1,
        }

        if outcome not in rewards:
            raise ValueError("invalid outcome")

        return rewards[outcome]

    def react(self, trader_id):
        role = self.roles[trader_id]
        if role == Role.INNOCENT:
            return TraderAction.NORMAL
        if role == Role.FRAUDSTER:
            return TraderAction.CONCEAL
        return TraderAction.SIGNAL

    def react_acc(self):
        if self.accus is None:
            raise RuntimeError("no accus")
        target = self.accus
        actions = {}
        for i in TRADER_IDS:
            if i == target:
                actions[i] = self.react(i)
            else:
                actions[i] = TraderAction.NORMAL
        return actions

    def soc_reac(self):
        if self.accus is None:
            raise RuntimeError("no accus")
        target = self.accus
        role = self.roles[target]
        if role == Role.INNOCENT:
            return self.mk_stat(
                target,
                StatType.DEFEND,
                target,
            )
        if role == Role.FRAUDSTER:
            other = next(i for i in TRADER_IDS if i != target)
            return self.mk_stat(
                target, StatType.ACCUSE,
                other,
            )
        other = next(i for i in TRADER_IDS if i != target)
        return self.mk_stat(
            target, 
            StatType.SUPPORT,
            other,
        )

    def trd_obs(self, trd_id):
        if trd_id not in TRADER_IDS:
            raise ValueError("who even is that trader")
        return {
            "own_mrk": self.pub_obs[trd_id],
            "market": self.pub_obs.copy(),
            "profiles": {
                i: self.get_prof(i)
                for i in TRADER_IDS
            },
            "history": self.history[-5:],
            "accus": self.accus,
            "step": self.step_count,
            "budget": self.rem_budget,
        }

    def trd_state(self, trd_id):
        obs = self.trd_obs(trd_id)
        state = [self.disc_mrk(obs["own_mrk"])]
        state.extend(
            self.disc_mrk(obs["market"][i])
            for i in TRADER_IDS
        )
        prof = obs["profiles"][trd_id]
        state.extend([
            self.disc_prof(prof["acc_rate"]),
            self.disc_prof(prof["defen"]),
            self.disc_prof(prof["supp_rt"]),
            self.disc_prof(prof["target_sw_rt"]),
            self.disc_prof(prof["manip_rate"]),
        ])
        if obs["history"]:
            last = obs["history"][-1]
            state.extend([
                last.speaker,
                last.kind.value,
                -1 if last.target is None else last.target,
            ])
        else:
            state.extend([-1, -1, -1])
        state.append(-1 if obs["accus"] is None else obs["accus"])
        state.append(min(obs["budget"], 3))
        state.append(min(obs["step"] // 5, 3))
        return tuple(state)

    def trd_q_state(self, trd_id):
        if trd_id not in TRADER_IDS:
            raise ValueError("who even is that trader")
        obs = self.trd_obs(trd_id)
        state = [
            self.disc_mrk(obs["own_mrk"]),
            -1 if obs["accus"] is None else obs["accus"],
            min(obs["budget"], 3),
            min(obs["step"] // 5, 3),
        ]
        if obs["history"]:
            last = obs["history"][-1]
            state.extend([
                last.kind.value,
                -1 if last.target is None else last.target,
            ])
        else:
            state.extend([-1, -1])
        return tuple(state)

    def soc_targets(self, trd_id):
        if trd_id not in TRADER_IDS:
            raise ValueError("who even is that trader")
        return tuple(i for i in TRADER_IDS if i != trd_id)

    def soc_space(self, trd_id):
        acts = [(StatType.NEUTRAL, None)]
        for target in self.soc_targets(trd_id):
            acts.extend([
                (StatType.DEFEND, target),
                (StatType.ACCUSE, target),
                (StatType.SUPPORT, target),
            ])
        return acts

    def soc_state(self, trd_id):
        obs = self.trd_obs(trd_id)
        state = [
            self.disc_mrk(obs["own_mrk"]),
            int(obs["accus"] == trd_id),
            -1 if obs["accus"] is None else obs["accus"],
            min(obs["step"] // 5, 3),
        ]
        if obs["history"]:
            last = obs["history"][-1]
            state.extend([
                last.kind.value,
                -1 if last.target is None else last.target,
            ])
        else:
            state.extend([-1, -1])
        return tuple(state)

    def do_soc(self, speaker, action):
        kind, target = action
        if kind == StatType.NEUTRAL:
            return self.mk_stat(speaker, kind)
        if target not in self.soc_targets(speaker):
            raise ValueError("you cannot monologue at yourself, Sukuna")
        return self.mk_stat(speaker, kind, target)

    def trd_rew(self, trd_id, outcome):
        if trd_id not in TRADER_IDS:
            raise ValueError("who even is that trader")
        rews = {
            Role.INNOCENT: {
                "fraudster_caught": 5,
                "innocent_caught": -10,
                "trickster_caught": 1,
                None: 0,
            },
            Role.FRAUDSTER: {
                "fraudster_caught": -10,
                "innocent_caught": 10,
                "trickster_caught": 2,
                None: 1,
            },
            Role.TRICKSTER: {
                "fraudster_caught": -3,
                "innocent_caught": 5,
                "trickster_caught": -10,
                None: 1,
            },
        }
        if outcome not in rews[Role.INNOCENT]:
            raise ValueError("that outcome is giving nonexistent")
        return rews[self.roles[trd_id]][outcome]

    def soc_rew(self, trd_id, statement):
        role = self.roles[trd_id]
        if statement.kind == StatType.NEUTRAL:
            return 0
        if role == Role.INNOCENT:
            if statement.kind == StatType.DEFEND and statement.target == trd_id:
                return 1
            if statement.kind == StatType.ACCUSE:
                return 0
            return -1
        if role == Role.FRAUDSTER:
            if statement.kind == StatType.ACCUSE:
                return 1
            if statement.kind == StatType.DEFEND and statement.target == trd_id:
                return 1
            return 0
        if statement.kind in (StatType.ACCUSE, StatType.SUPPORT):
            return 1
        return 0



