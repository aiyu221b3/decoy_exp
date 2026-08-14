from collections import deque
import math


INVESTIGATOR = "investigator"
CRIMINAL = "criminal"


class WinRegister:
    def __init__(
        self,
        recent_size=500,
        min_episodes=2000,
        recent_weight=0.65,
        min_probability=0.60,
        min_expected_margin=10.0,
        confusion_interval=100,
        confusion_scale=2.0,
    ):
        if recent_size <= 0:
            raise ValueError("should be pos")
        if min_episodes < 0:
            raise ValueError("be non-negative")
        if not 0.0 <= recent_weight <= 1.0:
            raise ValueError("should be in [0, 1]")
        if not 0.5 <= min_probability <= 1.0:
            raise ValueError("range is [0.5, 1]")
        if min_expected_margin < 0:
            raise ValueError("non-negative req :<")
        if confusion_interval <= 0:
            raise ValueError("must be positive")

        self.recent_size = recent_size
        self.min_episodes = min_episodes
        self.recent_weight = recent_weight
        self.min_probability = min_probability
        self.min_expected_margin = min_expected_margin
        self.confusion_interval = confusion_interval
        self.confusion_scale = confusion_scale

        self.episodes_seen = 0
        self.all_history = []
        self.recent_history = deque(maxlen=recent_size)
        self.entropy_history = deque(maxlen=recent_size)

    @staticmethod
    def _winner(outcome):
        if outcome == "fraudster_caught":
            return INVESTIGATOR

        if outcome == "innocent_caught":
            return CRIMINAL

        # trickster caught and timeout are not wins for either 
        return None

    @staticmethod
    def _entropy(p):
        if p <= 0.0 or p >= 1.0:
            return 0.0

        return (
            -p * math.log2(p)
            - (1.0 - p) * math.log2(1.0 - p)
        )

    def record(self, outcome):
        self.episodes_seen += 1

        winner = self._winner(outcome)

        if winner is None:
            return

        self.all_history.append(winner)
        self.recent_history.append(winner)

        inv_rate = self.all_history.count(INVESTIGATOR) / len(
            self.all_history
        )
        self.entropy_history.append(self._entropy(inv_rate))

    @property
    def decisive_episodes(self):
        return len(self.all_history)

    def rates(self):
        if not self.all_history:
            return {
                INVESTIGATOR: {
                    "all": 0.5,
                    "recent": 0.5,
                },
                CRIMINAL: {
                    "all": 0.5,
                    "recent": 0.5,
                },
            }

        all_inv = self.all_history.count(INVESTIGATOR)
        all_n = len(self.all_history)

        recent_inv = self.recent_history.count(INVESTIGATOR)
        recent_n = len(self.recent_history)

        all_inv_rate = all_inv / all_n
        recent_inv_rate = (
            recent_inv / recent_n
            if recent_n
            else all_inv_rate
        )

        return {
            INVESTIGATOR: {
                "all": all_inv_rate,
                "recent": recent_inv_rate,
            },
            CRIMINAL: {
                "all": 1.0 - all_inv_rate,
                "recent": 1.0 - recent_inv_rate,
            },
        }

    def predict(self):
        rates = self.rates()

        predictions = {
            side: (
                (1.0 - self.recent_weight) * values["all"]
                + self.recent_weight * values["recent"]
            )
            for side, values in rates.items()
        }

        winner = max(predictions, key=predictions.get)
        probability = predictions[winner]

        if not self.all_history:
            return None, 0.5

        return winner, probability

    def should_commit(self, episode_no, total_episodes):
        if self.episodes_seen < self.min_episodes:
            return None

        winner, probability = self.predict()

        if winner is None:
            return None

        remaining = max(
            total_episodes - episode_no - 1,
            0,
        )

        expected_margin = (
            abs(probability - 0.5) * remaining
        )

        if probability < self.min_probability:
            return None

        if expected_margin < self.min_expected_margin:
            return None

        return winner

    def confusion_bonus(self):
        if self.episodes_seen == 0:
            return 0.0

        if self.episodes_seen % self.confusion_interval != 0:
            return 0.0

        if not self.entropy_history:
            return 0.0

        window = list(
            self.entropy_history
        )[-self.confusion_interval:]

        mean_entropy = sum(window) / len(window)

        return self.confusion_scale * mean_entropy