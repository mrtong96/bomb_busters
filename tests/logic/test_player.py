from collections import Counter

import pytest

from src.logic.decision import (
    AskeeResponseDecision,
    AskerResponseDecision,
    DualCutDecision,
    PassDecision,
    SingleCutDecision,
)
from src.logic.game_state import GameState
from src.logic.player import Player
from src.logic.wire import BLUE, RED, YELLOW, Wire


def b(rank):
    return Wire(rank=rank, color=BLUE)


def y(rank):
    return Wire(rank=rank, color=YELLOW)


def r(rank):
    return Wire(rank=rank, color=RED)


def make_game_state(hands, revealed=None, turns=None, past_first_round=True):
    """Build a GameState with deterministic hands by overriding the random deal.

    Hands must already be sorted by wire.raw_int (rank * 10 + color).

    By default the state is fast-forwarded past the first round (N dummy PassDecision
    turns prepended) so tests that care about cut / response dispatch don't have to
    stage first-round reveals. Set past_first_round=False to exercise the first-round
    code paths directly.
    """
    assert len(hands) == 5, "GameState hard-codes 5 players"
    for i, hand in enumerate(hands):
        ordered = sorted(hand, key=lambda w: w.raw_int)
        assert [w.raw_int for w in hand] == [w.raw_int for w in ordered], (
            f"hand {i} not sorted by raw_int: {hand}"
        )

    gs = GameState(num_players=5)
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
    gs.turns = list(turns) if turns is not None else []
    if past_first_round:
        gs.turns = [[PassDecision()] for _ in range(len(hands))] + gs.turns
    return gs


# --- single cut ---

def test_single_cut_blue_all_remaining_in_hand():
    # Player 0 holds all 4 blue-3s; no other player has a blue-3.
    hands = [
        [b(3), b(3), b(3), b(3), b(7)],
        [b(2), b(4), b(5), b(6), b(7)],  # player 1 still has blue-7
        [b(8), b(9), b(10), b(11), b(12)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    assert sum(1 for d in decisions if d.wire == b(3)) == 1
    assert not any(d.wire == b(7) for d in decisions)


def test_single_cut_blue_blocked_by_other_player():
    hands = [
        [b(3), b(3), b(3), b(7), b(7)],
        [b(1), b(2), b(3), b(4), b(5)],  # player 1 still holds a blue-3
        [b(6), b(8), b(9), b(10), b(11)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(12), b(12), b(12), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    assert not any(d.wire == b(3) for d in decisions)


def test_single_cut_blue_unblocked_after_reveal():
    hands = [
        [b(3), b(3), b(3), b(7), b(7)],
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(8), b(9), b(10), b(11)],
        [b(1), b(2), b(4), b(5), b(6)],
        [b(8), b(12), b(12), b(12), b(12)],
    ]
    revealed = [[False] * 5 for _ in range(5)]
    revealed[1][2] = True  # b(3) at sorted position 2 in player 1's hand
    gs = make_game_state(hands, revealed=revealed)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    assert sum(1 for d in decisions if d.wire == b(3)) == 1


def test_single_cut_yellow_all_in_own_hand():
    hands = [
        [b(1), b(2), y(3), b(5), y(7)],
        [b(3), b(4), b(6), b(8), b(9)],
        [b(1), b(2), b(10), b(11), b(12)],
        [b(3), b(4), b(5), b(6), b(7)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    yellow_decisions = [d for d in decisions if d.wire.color == YELLOW]
    assert len(yellow_decisions) == 1  # single cut-action covers every matching yellow position
    assert yellow_decisions[0].wire.rank == 0  # rank=0 signals "all yellows"


def test_single_cut_yellow_blocked_by_other_player():
    hands = [
        [b(1), b(2), y(3), b(5), b(6)],
        [b(3), b(4), y(7), b(8), b(9)],
        [b(1), b(2), b(10), b(11), b(12)],
        [b(3), b(4), b(5), b(6), b(7)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    assert not any(d.wire.color == YELLOW for d in decisions)


def test_single_cut_red_only_when_all_unrevealed_are_red():
    hands = [
        [b(1), b(2), r(3), r(4), r(5)],
        [b(3), b(4), b(6), b(8), b(9)],
        [b(1), b(2), b(10), b(11), b(12)],
        [b(3), b(4), b(5), b(6), b(7)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    revealed = [[False] * 5 for _ in range(5)]
    revealed[0][0] = revealed[0][1] = True  # blues revealed, only reds left unrevealed
    gs = make_game_state(hands, revealed=revealed)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    red_decisions = [d for d in decisions if d.wire.color == RED]
    assert len(red_decisions) == 1  # single cut-action covers every matching red position
    assert red_decisions[0].wire.rank == 0  # rank=0 signals "all reds"


def test_single_cut_red_blocked_when_mixed_hand():
    hands = [
        [b(1), b(2), r(3), r(4), r(5)],
        [b(3), b(4), b(6), b(8), b(9)],
        [b(1), b(2), b(10), b(11), b(12)],
        [b(3), b(4), b(5), b(6), b(7)],
        [b(8), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_single_cut_decisions(gs)
    assert not any(d.wire.color == RED for d in decisions)


# --- dual cut ---

def test_dual_cut_enumeration_counts():
    # Own blue ranks: {1, 2}. Others' hand sizes: all 5, unrevealed. No own yellow.
    hands = [
        [b(1), b(1), b(1), b(2), b(2)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(3), b(9), b(10), b(11), b(12)],
        [b(2), b(3), b(4), b(5), b(6)],
        [b(7), b(8), b(9), b(10), b(11)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_dual_cut_decisions(gs)
    # 4 other players × 5 unrevealed positions × 2 own-blue-ranks
    assert len(decisions) == 4 * 5 * 2
    assert all(d.asker_player_index == 0 for d in decisions)
    assert all(d.askee_player_index != 0 for d in decisions)


def test_dual_cut_skips_revealed_askee_positions():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    revealed = [[False] * 5 for _ in range(5)]
    revealed[1][0] = revealed[1][1] = True  # two positions in player 1's hand revealed
    gs = make_game_state(hands, revealed=revealed)
    decisions = Player(player_index=0)._get_all_dual_cut_decisions(gs)
    p1_targets = [d for d in decisions if d.askee_player_index == 1]
    assert all(d.askee_hand_position not in (0, 1) for d in p1_targets)


def test_dual_cut_yellow_unspecified_rank_only_when_owning_yellow():
    with_yellow = [
        [b(1), b(2), y(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(with_yellow)
    decisions = Player(player_index=0)._get_all_dual_cut_decisions(gs)
    yellow_duals = [d for d in decisions if d.wire.color == YELLOW]
    assert len(yellow_duals) == 4 * 5
    assert all(d.wire.rank == 0 for d in yellow_duals)

    without_yellow = [[b(1), b(2), b(3), b(4), b(5)]] + [list(h) for h in with_yellow[1:]]
    gs2 = make_game_state(without_yellow)
    decisions2 = Player(player_index=0)._get_all_dual_cut_decisions(gs2)
    assert not any(d.wire.color == YELLOW for d in decisions2)


def test_dual_cut_never_red():
    hands = [
        [b(1), b(2), r(3), b(4), b(5)],
        [r(4), b(6), b(7), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0)._get_all_dual_cut_decisions(gs)
    assert not any(d.wire.color == RED for d in decisions)


# --- askee response ---

def test_askee_response_blue_success():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    dual = DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    )
    gs = make_game_state(hands, turns=[[dual]])
    response = Player(player_index=1)._get_askee_response_decision(gs)
    assert isinstance(response, AskeeResponseDecision)
    assert response.is_successful_dual_cut
    assert response.indicator_wire == b(6)
    assert response.indicator_wire_position == 0


def test_askee_response_blue_failure():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    dual = DualCutDecision(
        wire=b(1), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    )
    gs = make_game_state(hands, turns=[[dual]])
    response = Player(player_index=1)._get_askee_response_decision(gs)
    assert not response.is_successful_dual_cut
    assert response.indicator_wire == b(6)


def test_askee_response_yellow_unspecified_rank_matches_any_yellow():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(7), y(7), b(8), b(9), b(10)],  # y(7)=71 comes right after b(7)=70
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    dual = DualCutDecision(
        wire=Wire(rank=0, color=YELLOW),
        asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=1,  # y(7) is at position 1 once sorted
    )
    gs = make_game_state(hands, turns=[[dual]])
    response = Player(player_index=1)._get_askee_response_decision(gs)
    assert response.is_successful_dual_cut
    assert response.indicator_wire == y(7)


def test_askee_response_raises_without_dual_cut():
    hands = [[b(i + 1) for i in range(5)] for _ in range(5)]
    gs = make_game_state(hands)  # empty turns
    with pytest.raises(ValueError):
        Player(player_index=0)._get_askee_response_decision(gs)


# --- asker response ---

def test_asker_response_picks_matching_wire():
    hands = [
        [b(1), b(3), b(4), b(5), b(6)],  # asker has b(6) at sorted position 4
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    dual = DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    )
    askee = AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=True,
        indicator_wire=b(6), indicator_wire_position=0,
    )
    gs = make_game_state(hands, turns=[[dual, askee]])
    responses = Player(player_index=0)._get_all_asker_response_decisions(gs)
    assert len(responses) == 1  # only b(6) at position 4 matches in player 0's hand
    response = responses[0]
    assert isinstance(response, AskerResponseDecision)
    assert response.wire == b(6)
    assert response.hand_position == 4


def test_asker_response_raises_when_askee_response_failed():
    hands = [[b(i + 1) for i in range(5)] for _ in range(5)]
    dual = DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    )
    failed_askee = AskeeResponseDecision(
        wire=b(1), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=False,
        indicator_wire=b(1), indicator_wire_position=0,
    )
    gs = make_game_state(hands, turns=[[dual, failed_askee]])
    with pytest.raises(ValueError):
        Player(player_index=0)._get_all_asker_response_decisions(gs)


# --- get_all_legal_decisions dispatch ---

def test_legal_decisions_start_of_turn_returns_singles_and_duals():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    gs = make_game_state(hands)
    decisions = Player(player_index=0).get_all_legal_decisions(gs)
    assert all(isinstance(d, (SingleCutDecision, DualCutDecision)) for d in decisions)


def test_legal_decisions_after_dual_cut_returns_askee_response():
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    dual = DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    )
    gs = make_game_state(hands, turns=[[dual]])
    decisions = Player(player_index=1).get_all_legal_decisions(gs)
    assert len(decisions) == 1
    assert isinstance(decisions[0], AskeeResponseDecision)


def test_legal_decisions_after_successful_askee_returns_asker_response():
    hands = [
        [b(1), b(3), b(4), b(5), b(6)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    dual = DualCutDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        askee_player_position=1, askee_hand_position=0,
    )
    askee = AskeeResponseDecision(
        wire=b(6), asker_player_index=0, askee_player_index=1,
        is_successful_dual_cut=True,
        indicator_wire=b(6), indicator_wire_position=0,
    )
    gs = make_game_state(hands, turns=[[dual, askee]])
    decisions = Player(player_index=0).get_all_legal_decisions(gs)
    assert len(decisions) == 1
    assert isinstance(decisions[0], AskerResponseDecision)


# --- make_best_entropy_decision (performance + smoke) ---

def _add_rank_indicator_constraints_for_revealed_positions(gs):
    """For each revealed (player, position), add the RankIndicatorConstraint that a real game
    flow would have produced when that wire was revealed, so the density matrix properly
    collapses to 0/1 at those cells."""
    from src.logic.constraint import RankIndicatorConstraint
    for player_index, (hand, revealed_flags) in enumerate(zip(gs.player_wires, gs.revealed_wires)):
        for position, (wire, revealed) in enumerate(zip(hand, revealed_flags)):
            if not revealed:
                continue
            gs.public_constraints.append(RankIndicatorConstraint(
                player_index=player_index,
                wire_rank_index=gs.wire_to_index_mapping[wire],
                indicator_location_index=position,
            ))


def test_make_best_entropy_decision_on_heavily_revealed_state():
    """Performance/smoke: exact Level C minimax on a board where most wires are already
    revealed. A nearly-solved state should complete in a handful of kernel calls because
    few ranks have non-trivial density at each unrevealed position and each player holds
    only a couple of unrevealed wires.
    """
    # 5 players × 5-wire hands. Sorted by raw_int.
    hands = [
        [b(1), b(2), b(3), b(4), b(5)],
        [b(6), b(7), b(8), b(9), b(10)],
        [b(1), b(2), b(3), b(11), b(12)],
        [b(4), b(5), b(6), b(7), b(8)],
        [b(1), b(9), b(10), b(11), b(12)],
    ]
    # Reveal every position except the last one per player. Each player has 1 unrevealed wire.
    revealed = [[True, True, True, True, False] for _ in range(5)]
    gs = make_game_state(hands, revealed=revealed)
    gs.public_constraints = []  # ensure clean slate in case make_game_state didn't reset it
    _add_rank_indicator_constraints_for_revealed_positions(gs)

    decision = Player(player_index=0).make_decision(gs)
    assert decision is not None
    # the chosen decision must have been among the legal ones — verify by type
    assert isinstance(decision, (SingleCutDecision, DualCutDecision))
