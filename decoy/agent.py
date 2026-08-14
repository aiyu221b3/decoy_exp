from abc import ABC, abstractmethod
from decoy.environment import StatType

class Agent(ABC):
    @abstractmethod
    def act(self, obs):
        pass
    def observe(self, trans):
        pass
    def reset(self):
        pass

class FixAgent(Agent):
    def __init__(self, act):
        self.action = act
    def act(self, obs):
        return self.action

class QAgent(Agent):
    def __init__(self, learner, eps=None):
        self.learner = learner
        self.eps = eps
    def act(self, obs):
        return self.learner.act(obs, epsilon=self.eps)
    def observe(self, trans):
        self.learner.update(
            state=trans.observation,
            action=trans.action,
            reward=trans.reward,
            next_state=trans.next_observation,
            done=trans.done,
            next_actions=trans.next_actions,
        )

class DualAgent(Agent):
    def __init__(self, mrk, soc, eps=None):
        self.mrk = mrk
        self.soc = soc
        self.eps = eps
    def act(self, obs):
        return self.mrk.act(obs, epsilon=self.eps)
    def soc_act(self, obs, acts):
        return self.soc.act(
            obs,
            epsilon=self.eps,
            available_actions=acts,
        )

class TrickAgent(DualAgent):
    def __init__(
        self,
        mrk,
        soc,
        eps=None,
        advisor=None,
        episode_no=0,
        total_episodes=0,
        faction_targets=None,
    ):
        super().__init__(mrk, soc, eps)

        self.advisor = advisor
        self.episode_no = episode_no
        self.total_episodes = total_episodes
        self.faction_targets = faction_targets or {}

        self.side = None

    def reset(self):
        self.side = None

    def commit(self, side):
        if side not in self.faction_targets:
            raise ValueError("invalid Trickster faction")

        self.side = side

    def _committed_actions(self, acts):
        if self.side is None:
            return acts

        ally, opponent = self.faction_targets[self.side]

        allowed = []

        for action in acts:
            kind, target = action

            if kind in (
                StatType.DEFEND,
                StatType.SUPPORT,
            ) and target == ally:
                allowed.append(action)

            elif (
                kind == StatType.ACCUSE
                and target == opponent
            ):
                allowed.append(action)

        return allowed

    def soc_act(self, obs, acts):
        if self.side is None and self.advisor is not None:
            side = self.advisor.should_commit(
                self.episode_no,
                self.total_episodes,
            )

            if side is not None:
                self.commit(side)

        available = self._committed_actions(acts)

        if not available:
            available = acts

        return self.soc.act(
            obs,
            epsilon=self.eps,
            available_actions=available,
        )