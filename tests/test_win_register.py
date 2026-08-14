from decoy.win_register import (
    WinRegister,
    INVESTIGATOR,
    CRIMINAL,
)


def test_win_register_records_only_faction_wins():
    reg = WinRegister(
        min_episodes=0,
    )

    reg.record("fraudster_caught")
    reg.record("innocent_caught")
    reg.record("trickster_caught")
    reg.record(None)

    assert reg.episodes_seen == 4
    assert reg.decisive_episodes == 2


def test_win_register_predicts_from_history():
    reg = WinRegister(
        min_episodes=0,
        recent_weight=0.65,
    )

    for _ in range(7):
        reg.record("fraudster_caught")

    for _ in range(3):
        reg.record("innocent_caught")

    winner, probability = reg.predict()

    assert winner == INVESTIGATOR
    assert probability > 0.5


def test_win_register_waits_until_minimum_episode_count():
    reg = WinRegister(
        min_episodes=5,
    )

    for _ in range(4):
        reg.record("fraudster_caught")

    assert reg.should_commit(4, 100) is None


def test_win_register_uses_remaining_horizon():
    reg = WinRegister(
        min_episodes=0,
        min_probability=0.60,
        min_expected_margin=10,
    )

    for _ in range(8):
        reg.record("fraudster_caught")

    for _ in range(2):
        reg.record("innocent_caught")

    assert reg.should_commit(20, 100) == INVESTIGATOR