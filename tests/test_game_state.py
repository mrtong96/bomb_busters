from collections import Counter

import pytest

from src.constraint import (
    RankIndicatorConstraint,
    WireAskConstraint,
    YellowWireAskConstraint,
)
from src.decision import (
    AskeeResponseDecision,
    AskerResponseDecision,
    DualCutDecision,
    SingleCutDecision,
)
from src.game_state import GameState
from src.wire import BLUE, RED, YELLOW, Wire


def b(rank):
    return Wire(rank=rank, color=BLUE)


def y(rank):
    return Wire(rank=rank, color=YELLOW)


def r(rank):
    return Wire(rank=rank, color=RED)


def make_game_state(hands, revealed=None, total_health=5):
    """Build a GameState with deterministic hands by overriding the random deal.

    Hands must already be sorted by wire.raw_int (rank * 10 + color).
    """
    assert len(hands) == 5, "GameState hard-codes 5 players"
    for i, hand in enumerate(hands):
        ordered = sorted(hand, key=lambda w: w.raw_int)
        assert [w.raw_int for w in hand] == [w.raw_int for w in ordered], (
            f"hand {i} not sorted by raw_int: {hand}"
        )

    gs = GameState(num_players=5, total_health=total_health)
    gs.player_wires = [list(h) for h in hands]
    gs.revealed_wires = (
        [list(flags) for flags in revealed] if revealed is not None
        else [[False] * len(h) for h in hands]
    )
    counts = Counter(w for h in hands for w in h)
    gs.wire_ranks = sorted(counts.keys())
    gs.wire_counts = [counts[w] for w in gs.wire_ranks]
    gs.wire_to_index_mapping = {w: i for i, w in enumerate(gs.wire_ranks)}
    gs.wire_revealed_counts = [0] * len(gs.wire_ranks)
    for hand, flags in zip(gs.player_wires, gs.revealed_wires):
        for wire, flag in zip(hand, flags):
            if flag:
                gs.wire_revealed_counts[gs.wire_to_index_mapping[wire]] += 1
    gs.turns = []
    gs.public_constraints = []
    gs.total_health = total_health
    gs._has_won = False
    gs._has_lost = False
    return gs


# ========== _update_state_from_decision ==========

def test_state_single_cut_blue_reveals_every_matching_unrevealed_position():
    hands = [
        [b(3), b(3), b(3), b(3), b(7)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_state_from_decision(SingleCutDecision(wire=b(3), player_index=0))

    for pos in range(4):
        assert gs.revealed_wires[0][pos], pos
    assert not gs.revealed_wires[0][4]
    assert gs.wire_revealed_counts[gs.wire_to_index_mapping[b(3)]] == 4
    # other players untouched
    for p in range(1, 5):
        assert not any(gs.revealed_wires[p])


def test_state_single_cut_yellow_reveals_every_yellow_position_only():
    hands = [
        [b(1), b(2), y(3), b(5), y(7)],
        [b(3), b(4), b(6), b(8), b(9)],
        [b(1), b(2), b(10), b(11), b(12)],
        [b(3), b(4), b(5), b(6), b(7)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_state_from_decision(
        SingleCutDecision(wire=Wire(rank=0, color=YELLOW), player_index=0),
    )
    assert gs.revealed_wires[0][2]  # y(3)
    assert gs.revealed_wires[0][4]  # y(7)
    for pos in (0, 1, 3):
        assert not gs.revealed_wires[0][pos], pos


def test_state_single_cut_skips_already_revealed_positions():
    hands = [
        [b(3), b(3), b(3), b(3), b(7)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    revealed = [[False] * 5 for _ in range(5)]
    revealed[0][1] = True  # one b(3) already revealed beforehand
    gs = make_game_state(hands, revealed=revealed)
    # pre-bump the revealed count to reflect the pre-revealed position
    gs.wire_revealed_counts[gs.wire_to_index_mapping[b(3)]] = 1

    gs._update_state_from_decision(SingleCutDecision(wire=b(3), player_index=0))

    # all b(3) positions now revealed, but the counter bumps only by 3 (not 4)
    assert gs.wire_revealed_counts[gs.wire_to_index_mapping[b(3)]] == 4
    for pos in range(4):
        assert gs.revealed_wires[0][pos]


def test_state_dual_cut_is_a_noop():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands, total_health=5)
    revealed_before = [list(flags) for flags in gs.revealed_wires]
    counts_before   = list(gs.wire_revealed_counts)

    gs._update_state_from_decision(DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    ))

    assert gs.revealed_wires == revealed_before
    assert gs.wire_revealed_counts == counts_before
    assert gs.total_health == 5
    assert not gs._has_won and not gs._has_lost


def test_state_askee_response_success_reveals_askee_position():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands, total_health=5)
    gs._update_state_from_decision(AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=True, indicator_wire=b(6), indicator_wire_position=0,
    ))

    assert gs.revealed_wires[1][0]
    assert gs.wire_revealed_counts[gs.wire_to_index_mapping[b(6)]] == 1
    assert gs.total_health == 5


def test_state_askee_response_failure_decrements_health_and_does_not_reveal():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands, total_health=3)
    gs._update_state_from_decision(AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=False, indicator_wire=b(6), indicator_wire_position=0,
    ))

    assert not gs.revealed_wires[1][0]
    assert gs.wire_revealed_counts[gs.wire_to_index_mapping[b(6)]] == 0
    assert gs.total_health == 2
    assert not gs._has_lost


def test_state_askee_response_failure_sets_has_lost_when_health_hits_zero():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands, total_health=1)
    gs._update_state_from_decision(AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=False, indicator_wire=b(6), indicator_wire_position=0,
    ))
    assert gs.total_health == 0
    assert gs._has_lost
    assert not gs._has_won


def test_state_asker_response_reveals_asker_position():
    hands = [
        [b(1), b(3), b(4), b(5), b(6)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_state_from_decision(AskerResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1, hand_position=4,
    ))
    assert gs.revealed_wires[0][4]
    assert gs.wire_revealed_counts[gs.wire_to_index_mapping[b(6)]] == 1


def test_state_sets_has_won_when_all_wires_including_yellow_and_red_are_cut():
    # Tiny board: one wire per player, mixing colors. Reveal the last one and check win.
    hands = [[b(1)], [b(2)], [y(3)], [r(4)], [b(5)]]
    revealed = [[True], [True], [True], [True], [False]]
    gs = make_game_state(hands, revealed=revealed)
    gs._update_state_from_decision(SingleCutDecision(wire=b(5), player_index=4))
    assert gs._has_won
    assert not gs._has_lost


def test_state_win_takes_priority_over_loss_when_both_would_trigger():
    # Reveal the last wire AND bring health to zero in the same tick by pre-setting health=0
    # from prior turns. Win condition (all cut) must take priority.
    hands = [[b(1)], [b(2)], [b(3)], [b(4)], [b(5)]]
    revealed = [[True], [True], [True], [True], [False]]
    gs = make_game_state(hands, revealed=revealed, total_health=0)
    gs._update_state_from_decision(SingleCutDecision(wire=b(5), player_index=4))
    assert gs._has_won
    assert not gs._has_lost


# ========== _update_constraints_from_decision ==========

def test_constraints_single_cut_blue_emits_rank_indicator_per_matching_position():
    hands = [
        [b(3), b(3), b(3), b(3), b(7)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_constraints_from_decision(SingleCutDecision(wire=b(3), player_index=0))

    emitted = [c for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)]
    assert len(emitted) == 4
    b3_idx = gs.wire_to_index_mapping[b(3)]
    assert all(c.player_index == 0 for c in emitted)
    assert all(c.wire_rank_index == b3_idx for c in emitted)
    assert sorted(c.indicator_location_index for c in emitted) == [0, 1, 2, 3]


def test_constraints_single_cut_yellow_emits_one_per_yellow_wire_with_actual_rank():
    hands = [
        [b(1), b(2), y(3), b(5), y(7)],
        [b(3), b(4), b(6), b(8), b(9)],
        [b(1), b(2), b(10), b(11), b(12)],
        [b(3), b(4), b(5), b(6), b(7)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_constraints_from_decision(
        SingleCutDecision(wire=Wire(rank=0, color=YELLOW), player_index=0),
    )
    emitted = [c for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)]
    assert len(emitted) == 2
    by_pos = {c.indicator_location_index: c.wire_rank_index for c in emitted}
    assert by_pos[2] == gs.wire_to_index_mapping[y(3)]
    assert by_pos[4] == gs.wire_to_index_mapping[y(7)]


def test_constraints_single_cut_skips_already_revealed_positions():
    hands = [
        [b(3), b(3), b(3), b(3), b(7)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    revealed = [[False] * 5 for _ in range(5)]
    revealed[0][2] = True  # middle b(3) already revealed
    gs = make_game_state(hands, revealed=revealed)
    gs._update_constraints_from_decision(SingleCutDecision(wire=b(3), player_index=0))
    emitted = [c for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)]
    assert sorted(c.indicator_location_index for c in emitted) == [0, 1, 3]


def test_constraints_dual_cut_specific_rank_emits_wire_ask():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_constraints_from_decision(DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    ))
    wire_asks = [
        c for c in gs.public_constraints
        if isinstance(c, WireAskConstraint) and not isinstance(c, YellowWireAskConstraint)
    ]
    assert len(wire_asks) == 1
    c = wire_asks[0]
    assert c.player_index == 0
    assert c.wire_rank_index == gs.wire_to_index_mapping[b(6)]


def test_constraints_dual_cut_yellow_emits_yellow_wire_ask_over_every_yellow_rank():
    hands = [
        [b(1), b(2), y(3), b(5), b(6)],
        [b(7), y(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [y(5), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_constraints_from_decision(DualCutDecision(
        wire=Wire(rank=0, color=YELLOW), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    ))
    yellow_asks = [c for c in gs.public_constraints if isinstance(c, YellowWireAskConstraint)]
    assert len(yellow_asks) == 1
    c = yellow_asks[0]
    assert c.player_index == 0
    expected_indexes = sorted(
        i for i, w in enumerate(gs.wire_ranks) if w.color == YELLOW
    )
    assert sorted(c.yellow_rank_indexes) == expected_indexes


def test_constraints_askee_response_success_emits_rank_indicator_at_asked_position():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_constraints_from_decision(AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=True, indicator_wire=b(6), indicator_wire_position=0,
    ))
    emitted = [c for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)]
    assert len(emitted) == 1
    c = emitted[0]
    assert c.player_index == 1
    assert c.wire_rank_index == gs.wire_to_index_mapping[b(6)]
    assert c.indicator_location_index == 0


def test_constraints_askee_response_failure_still_emits_rank_indicator():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    # failed askee: indicator_wire is the actual wire at the asked position (b(6) at askee pos 0)
    gs._update_constraints_from_decision(AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=False, indicator_wire=b(6), indicator_wire_position=0,
    ))
    emitted = [c for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)]
    assert len(emitted) == 1


def test_constraints_asker_response_emits_rank_indicator_at_asker_hand_position():
    hands = [
        [b(1), b(3), b(4), b(5), b(6)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    gs._update_constraints_from_decision(AskerResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1, hand_position=4,
    ))
    emitted = [c for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)]
    assert len(emitted) == 1
    c = emitted[0]
    assert c.player_index == 0
    assert c.wire_rank_index == gs.wire_to_index_mapping[b(6)]
    assert c.indicator_location_index == 4
