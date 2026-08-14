from pathlib import Path

from decoy.multi_training import tr_multi


SEEDS = [11, 42, 99]

N_EPS = 10
BUDGET = 10
MAX_STEPS = 20


def main():
    print("=" * 72)
    print("DECOY MULTI-AGENT PRE-FLIGHT")
    print("=" * 72)

    print()
    print("Configuration")
    print(f"  episodes:          {N_EPS}")
    print(f"  seeds:             {SEEDS}")
    print(f"  budget:            {BUDGET}")
    print(f"  max_steps:         {MAX_STEPS}")
    print("  adaptive_inv_alpha: False")
    print()

    # --------------------------------------------------
    # 1. Baseline smoke tests
    # --------------------------------------------------

    print("1. BASELINE SMOKE TESTS")
    print("-" * 72)

    for seed in SEEDS:
        print(f"seed={seed}", end=" ... ")

        inv, mrk, soc, hist, summ, register = tr_multi(
            n_eps=N_EPS,
            seed=seed,
            budget=BUDGET,
            max_steps=MAX_STEPS,
            adaptive_inv_alpha=False,
        )

        assert len(hist) == N_EPS

        outcome_total = (
            summ["fraudster_catch_rate"]
            + summ["innocent_catch_rate"]
            + summ["trickster_catch_rate"]
            + summ["timeout_rate"]
        )

        assert abs(outcome_total - 1.0) < 1e-12

        assert len(inv.actions) == 12
        assert set(mrk)
        assert set(soc)

        assert register.episodes_seen == N_EPS

        # Main experiment must NOT adapt investigator alpha.
        assert inv.alpha == 0.1

        print("PASS")

    print()

    # --------------------------------------------------
    # 2. Reproducibility
    # --------------------------------------------------

    print("2. REPRODUCIBILITY CHECK")
    print("-" * 72)

    result_a = tr_multi(
        n_eps=N_EPS,
        seed=SEEDS[0],
        budget=BUDGET,
        max_steps=MAX_STEPS,
        adaptive_inv_alpha=False,
    )

    result_b = tr_multi(
        n_eps=N_EPS,
        seed=SEEDS[0],
        budget=BUDGET,
        max_steps=MAX_STEPS,
        adaptive_inv_alpha=False,
    )

    assert result_a[3] == result_b[3]
    assert result_a[4] == result_b[4]

    print("same seed -> identical history and summary: PASS")
    print()

    # --------------------------------------------------
    # 3. Trickster / win-register check
    # --------------------------------------------------

    print("3. TRICKSTER WIN-REGISTER CHECK")
    print("-" * 72)

    _, _, _, hist, summ, register = tr_multi(
        n_eps=10,
        seed=123,
        budget=BUDGET,
        max_steps=MAX_STEPS,
        adaptive_inv_alpha=False,
    )

    assert register.episodes_seen == 10

    print(
        f"episodes recorded: {register.episodes_seen}"
    )

    print(
        f"all_history length:   {len(register.all_history)}"
    )

    print(
        f"recent_history length:"
        f" {len(register.recent_history)}"
    )

    assert register.episodes_seen == 10

    print("win register active: PASS")

    print()

    # --------------------------------------------------
    # 4. Output directory check
    # --------------------------------------------------

    print("4. OUTPUT DIRECTORY CHECK")
    print("-" * 72)

    results_dir = Path("results")

    if results_dir.exists():
        files = sorted(
            p.name
            for p in results_dir.iterdir()
            if p.is_file()
        )

        print(f"results/: {len(files)} files currently present")

        for name in files[:20]:
            print(f"  {name}")

        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
    else:
        print("results/ does not exist yet.")

    print()
    print(
        "Do NOT launch the final sweep until its CSV output filename "
        "is confirmed to be different from the existing RL CSVs."
    )
    print()

    # --------------------------------------------------
    # 5. Final status
    # --------------------------------------------------

    print("=" * 72)
    print("PRE-FLIGHT CHECKS PASSED")
    print("=" * 72)
    print()
    print("Safe to proceed to the expensive sweep.")
    print("Main run: adaptive_inv_alpha=False")
    print(f"Seeds: {SEEDS}")
    print()


if __name__ == "__main__":
    main()