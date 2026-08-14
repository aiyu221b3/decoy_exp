from decoy.training import train_q_investigator

def print_summary(n_episodes, summary):
    print(f"\n{n_episodes} episodes in the Decoy blender")
    for key, value in summary.items():
        print(f"{key}: {value:.3f}")

def main():
    for n_episodes in (10, 100, 1000, 10_000):
        _, _, summary = train_q_investigator(
            n_episodes=n_episodes,
            seed=42,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1,
            max_steps=20,
            budget=10,
        )
        print_summary(n_episodes, summary)

if __name__ == "__main__":
    main()