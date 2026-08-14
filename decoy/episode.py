from dataclasses import dataclass


@dataclass
class Transition:
    observation: object
    action: object
    reward: object
    next_observation: object
    done: bool
    next_actions: tuple = ()


@dataclass
class Episode:
    transitions: list[Transition]

    def add(self, transition):
        self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)