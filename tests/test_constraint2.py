import numpy as np
import pytest

from src.constraint2 import (
    CountIndicatorConstraint,
    RankEqualConstraint,
    RankIndicatorConstraint,
    RankNotEqualConstraint,
    SubsetConstraint,
    WireAskConstraint,
    WireLimitConstraint,
    get_constraint_matrix,
)

NUM_PLAYERS = 3
NUM_RANKS = 4
MAX_PREFILLED = 6
MAX_NUM_WIRES = 5


def make_matrix(
        num_players=NUM_PLAYERS,
        num_ranks=NUM_RANKS,
        max_prefilled=MAX_PREFILLED,
        max_num_wires=MAX_NUM_WIRES,
):
    return np.ones((num_players, num_ranks, max_prefilled, max_num_wires), dtype=np.bool_)


# --- RankIndicatorConstraint ---

def test_rank_indicator_rank_must_cover_location():
    matrix = make_matrix()
    RankIndicatorConstraint(
        player_index=0, wire_rank_index=2, indicator_location_index=3,
    ).update_constraint_matrix(matrix)

    for prefilled in range(MAX_PREFILLED):
        for num_wires in range(MAX_NUM_WIRES):
            covers = (prefilled <= 3) and (3 < prefilled + num_wires)
            assert matrix[0, 2, prefilled, num_wires] == covers, (prefilled, num_wires)


def test_rank_indicator_other_ranks_must_not_cover_location():
    matrix = make_matrix()
    RankIndicatorConstraint(
        player_index=0, wire_rank_index=2, indicator_location_index=3,
    ).update_constraint_matrix(matrix)

    for other_rank in (r for r in range(NUM_RANKS) if r != 2):
        for prefilled in range(MAX_PREFILLED):
            for num_wires in range(MAX_NUM_WIRES):
                covers = (prefilled <= 3) and (3 < prefilled + num_wires)
                assert matrix[0, other_rank, prefilled, num_wires] == (not covers), (
                    other_rank, prefilled, num_wires,
                )


def test_rank_indicator_does_not_touch_other_players():
    matrix = make_matrix()
    RankIndicatorConstraint(
        player_index=0, wire_rank_index=2, indicator_location_index=3,
    ).update_constraint_matrix(matrix)
    for player in range(1, NUM_PLAYERS):
        assert matrix[player].all()


# --- CountIndicatorConstraint ---

def test_count_indicator_forces_exact_count_on_rank_covering_location():
    matrix = make_matrix()
    CountIndicatorConstraint(
        player_index=1, indicator_location_index=2, wire_count=2,
    ).update_constraint_matrix(matrix)

    for rank in range(NUM_RANKS):
        for prefilled in range(MAX_PREFILLED):
            for num_wires in range(MAX_NUM_WIRES):
                covers = (prefilled <= 2) and (2 < prefilled + num_wires)
                expected = not (covers and num_wires != 2)
                assert matrix[1, rank, prefilled, num_wires] == expected, (
                    rank, prefilled, num_wires,
                )


def test_count_indicator_leaves_other_players_untouched():
    matrix = make_matrix()
    CountIndicatorConstraint(
        player_index=0, indicator_location_index=1, wire_count=1,
    ).update_constraint_matrix(matrix)
    for player in range(1, NUM_PLAYERS):
        assert matrix[player].all()


# --- WireAskConstraint ---

def test_wire_ask_forbids_zero_count_at_targeted_player_and_rank():
    matrix = make_matrix()
    WireAskConstraint(player_index=2, wire_rank_index=1).update_constraint_matrix(matrix)
    assert not matrix[2, 1, :, 0].any()
    assert matrix[2, 1, :, 1:].all()


def test_wire_ask_leaves_other_ranks_and_players_alone():
    matrix = make_matrix()
    WireAskConstraint(player_index=2, wire_rank_index=1).update_constraint_matrix(matrix)
    for rank in (r for r in range(NUM_RANKS) if r != 1):
        assert matrix[2, rank, :, 0].all()
    for player in (p for p in range(NUM_PLAYERS) if p != 2):
        assert matrix[player, 1, :, 0].all()


# --- SubsetConstraint ---

def test_subset_caps_counts_at_one_per_player():
    matrix = make_matrix()
    subset_ranks = [1, 3]
    other_ranks = [r for r in range(NUM_RANKS) if r not in subset_ranks]
    SubsetConstraint(wire_rank_indexes=subset_ranks, subset_count=1).update_constraint_matrix(matrix)
    assert not matrix[:, subset_ranks, :, 2:].any()
    assert matrix[:, subset_ranks, :, :2].all()
    assert matrix[:, other_ranks].all()


# --- RankEqualConstraint ---

def test_rank_equal_forbids_blocks_that_straddle_the_boundary():
    matrix = make_matrix()
    left = 2
    RankEqualConstraint(
        player_index=0, left_indicator_location_index=left,
    ).update_constraint_matrix(matrix)

    for rank in range(NUM_RANKS):
        for prefilled in range(MAX_PREFILLED):
            for num_wires in range(MAX_NUM_WIRES):
                covers_left = (prefilled <= left) and (left < prefilled + num_wires)
                covers_right = (prefilled <= left + 1) and (left + 1 < prefilled + num_wires)
                expected = not (covers_left ^ covers_right)
                assert matrix[0, rank, prefilled, num_wires] == expected, (
                    rank, prefilled, num_wires,
                )


# --- RankNotEqualConstraint ---

def test_rank_not_equal_forbids_blocks_that_cover_both_positions():
    matrix = make_matrix()
    left = 2
    RankNotEqualConstraint(
        player_index=0, left_indicator_location_index=left,
    ).update_constraint_matrix(matrix)

    for rank in range(NUM_RANKS):
        for prefilled in range(MAX_PREFILLED):
            for num_wires in range(MAX_NUM_WIRES):
                covers_left = (prefilled <= left) and (left < prefilled + num_wires)
                covers_right = (prefilled <= left + 1) and (left + 1 < prefilled + num_wires)
                expected = not (covers_left and covers_right)
                assert matrix[0, rank, prefilled, num_wires] == expected, (
                    rank, prefilled, num_wires,
                )


# --- WireLimitConstraint ---

def test_wire_limit_forbids_block_extending_past_hand_size():
    matrix = make_matrix()
    wire_limit = 4
    WireLimitConstraint(player_index=1, wire_limit=wire_limit).update_constraint_matrix(matrix)

    for rank in range(NUM_RANKS):
        for prefilled in range(MAX_PREFILLED):
            for num_wires in range(MAX_NUM_WIRES):
                expected = (prefilled + num_wires) <= wire_limit
                assert matrix[1, rank, prefilled, num_wires] == expected, (
                    rank, prefilled, num_wires,
                )


def test_wire_limit_does_not_touch_other_players():
    matrix = make_matrix()
    WireLimitConstraint(player_index=1, wire_limit=4).update_constraint_matrix(matrix)
    for player in (p for p in range(NUM_PLAYERS) if p != 1):
        assert matrix[player].all()


# --- get_constraint_matrix ---

def test_get_constraint_matrix_shape_and_auto_wire_limits():
    wire_limits_per_player = [3, 5]
    wire_ranks = [0, 1, 2]
    matrix = get_constraint_matrix(
        wire_limits_per_player=wire_limits_per_player,
        wire_ranks=wire_ranks,
        constraints=[],
    )
    expected_shape = (
        len(wire_limits_per_player),
        len(wire_ranks),
        max(wire_limits_per_player) + 1,
        MAX_NUM_WIRES,
    )
    assert matrix.shape == expected_shape
    for player, limit in enumerate(wire_limits_per_player):
        for prefilled in range(expected_shape[2]):
            for num_wires in range(MAX_NUM_WIRES):
                assert matrix[player, 0, prefilled, num_wires] == (
                    (prefilled + num_wires) <= limit
                )


def test_get_constraint_matrix_composes_multiple_constraints():
    wire_limits_per_player = [4, 4]
    wire_ranks = [0, 1, 2]
    matrix = get_constraint_matrix(
        wire_limits_per_player=wire_limits_per_player,
        wire_ranks=wire_ranks,
        constraints=[
            WireAskConstraint(player_index=0, wire_rank_index=1),
            SubsetConstraint(wire_rank_indexes=[2], subset_count=1),
        ],
    )
    # WireAsk: player 0 rank 1 nw=0 forbidden
    assert not matrix[0, 1, :, 0].any()
    # Subset: rank 2 counts >= 2 forbidden for everyone
    assert not matrix[:, 2, :, 2:].any()
    # WireLimit still in effect
    max_prefilled_here = max(wire_limits_per_player) + 1
    prefilled, num_wires = np.indices((max_prefilled_here, MAX_NUM_WIRES))
    exceeds = (prefilled + num_wires) > max(wire_limits_per_player)
    assert not matrix[:, :, exceeds].any()


def test_abstract_constraint_cannot_be_instantiated():
    from src.constraint2 import Constraint
    with pytest.raises(TypeError):
        Constraint()
