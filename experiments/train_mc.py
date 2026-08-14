from decoy.training import train_mc_investigator

def show(n, summ):
    print(f"\n{n} MC episodes in the Decoy blender")
    for key, val in summ.items():
        print(f"{key}: {val:.3f}")

def main():
    for n in (10, 100, 1000, 10_000):
        _, _, summ = train_mc_investigator(
            n_episodes=n,
            seed=42,
            gamma=0.99,
            epsilon=0.1,
            max_steps=20,
            budget=10,
        )
        show(n, summ)

if __name__ == "__main__":
    main()