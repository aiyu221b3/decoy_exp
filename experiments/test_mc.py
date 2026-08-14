from decoy.episode import Episode, Transition
from decoy.monte_carlo import MonteCarlo


def main():
    episode = Episode([])

    episode.add(
        Transition(
            observation=0,
            action=0,
            reward=0,
            next_observation=1,
            done=False,
        )
    )

    episode.add(
        Transition(
            observation=1,
            action=1,
            reward=1,
            next_observation=None,
            done=True,
        )
    )

    agent = MonteCarlo(
        actions=[0, 1],
        gamma=0.99,
    )

    agent.update(episode)

    print("State 0:", agent.q[0])
    print("State 1:", agent.q[1])


if __name__ == "__main__":
    main()