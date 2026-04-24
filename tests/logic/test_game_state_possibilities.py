import numpy as np
import pytest

from src.logic.constraint import RankIndicatorConstraint, SubsetConstraint
from src.logic.probability_utils import compute_probability_matrices


def rank_indicators_from_hand(player_index, hand, empty_sentinel=-1):
    """Emit one RankIndicatorConstraint per filled slot in a per-player hand array."""
    return [
        RankIndicatorConstraint(
            player_index=player_index,
            wire_rank_index=int(rank),
            indicator_location_index=position,
        )
        for position, rank in enumerate(hand)
        if rank != empty_sentinel
    ]


@pytest.mark.parametrize(
    "wire_limits_per_player,wire_limits",
    [
        # max 2 wires of each type, 3 total spaces, valid combos are 1,1,2/1,2,2
        (np.array([3, 3]), {i + 1: (3, 3) for i in range(2)}),
        # Can only select one wire at a time, uniform distribution
        (np.array([1, 1, 1, 1]), {i + 1: (1, 1) for i in range(4)}),
        # Scale test
        (np.array([2, 2, 2, 2]), {i + 1: (2, 2) for i in range(4)}),
        # Bigger scale test
        (np.array([10, 10, 10, 9, 9]), {i + 1: (4, 4) for i in range(12)}),
    ],
)
def test_sanity_combination_checks(wire_limits_per_player, wire_limits):
    results = compute_probability_matrices(wire_limits_per_player, wire_limits)
    density_matrix = results[0]
    combinations_matrix = results[1]

    # all the max limits per wire are equal, there should be symmetry among players with equal hand sizes
    if len(set([el[0] for el in wire_limits.values()])) == 1:
        for i in range(1, len(wire_limits_per_player)):
            if wire_limits_per_player[i] != wire_limits_per_player[0]:
                continue
            assert np.all(np.isclose(density_matrix[0], density_matrix[i])), (
                'failed symmetry', density_matrix,
            )
            assert np.all(np.isclose(combinations_matrix[0], combinations_matrix[i])), (
                'failed symmetry', combinations_matrix,
            )

    shannon_entropy = 0
    for row in density_matrix[0, :, :]:
        for el in row:
            if el == 0:
                continue
            shannon_entropy += - el * np.log2(el)

    # magic number, for a 10-card hand with all blue wires
    if np.isclose(np.sum(density_matrix), 48):
        assert np.isclose(23.076475108967493, shannon_entropy), shannon_entropy


def test_constraints():
    wires = 4

    # compute the wire limits per player and the individual wire limits for the remaining cards
    wire_limits_per_player = np.array([((wires * 4) + 4 - i) // 5 for i in range(5)])
    wire_limits = {i: (4, 4) for i in range(wires)}

    per_player_constraints = [np.full(limit, -1) for limit in wire_limits_per_player]
    # have one of each wire for the first player
    per_player_constraints[0] = np.array(list(range(len(per_player_constraints[0]))))
    indicator_constraints = [
        c for player_index, hand in enumerate(per_player_constraints)
        for c in rank_indicators_from_hand(player_index, hand)
    ]

    # profile things
    results = compute_probability_matrices(
        wire_limits_per_player=wire_limits_per_player,
        wire_limits=wire_limits,
        constraints=indicator_constraints,
    )

    assert np.all(np.isclose(results[0][0], np.identity(wires))), (
        "If the first player has one of each wire, then the wire density matrix should be the identity matrix",
        results[0][0],
    )


def test_constraints2():
    wires = 12

    # compute the wire limits per player and the individual wire limits for the remaining cards
    wire_limits_per_player = np.array([((wires * 4) + 4 - i) // 5 for i in range(5)])
    wire_limits = {i: (4, 4) for i in range(wires)}

    for i in range(10):
        per_player_constraints = [np.full(limit, -1) for limit in wire_limits_per_player]
        # have one of each wire for the first player

        deck = np.array([i for i in range(wires) for _ in range(4)])
        hand = np.random.choice(deck, size=((wires * 4) + 4) // 5, replace=False, p=None)
        per_player_constraints[0] = np.array(list(sorted(hand)))
        indicator_constraints = [
            c for player_index, player_hand in enumerate(per_player_constraints)
            for c in rank_indicators_from_hand(player_index, player_hand)
        ]
        # profile things
        _ = compute_probability_matrices(
            wire_limits_per_player=wire_limits_per_player,
            wire_limits=wire_limits,
            constraints=indicator_constraints,
        )


def test_subset_constraints():
    # 4 blue wires, 2/3 yellow, 1/2 red
    blue_wires = [(i + 1) * 10 for i in range(4)]
    yellow_wires = [21, 51, 71]
    red_wires = [55, 85]

    total_wires = len(blue_wires) * 4 + 3
    wire_limits_per_player = np.array([(total_wires + 4 - i) // 5 for i in range(5)], dtype=np.int32)
    wire_ranks = sorted(blue_wires + yellow_wires + red_wires)
    wire_map = {wire: i for i, wire in enumerate(wire_ranks)}
    yellow_wires_mapped = [wire_map[el] for el in sorted(yellow_wires)]
    red_wires_mapped = [wire_map[el] for el in sorted(red_wires)]
    yellow_subset = SubsetConstraint(wire_rank_indexes=yellow_wires_mapped, subset_count=2)
    red_subset = SubsetConstraint(wire_rank_indexes=red_wires_mapped, subset_count=1)

    wire_limits = {i: (4, 4) if wire in blue_wires else (0, 1) for i, wire in enumerate(wire_ranks)}

    results = compute_probability_matrices(
        wire_limits_per_player=wire_limits_per_player,
        wire_limits=wire_limits,
        constraints=[yellow_subset, red_subset],
    )

    for i in range(results[0].shape[2]):
        wire_rank = wire_ranks[i]
        wire_sum = np.sum(results[0][:, :, i])

        # blue wires have 4 count
        if wire_rank in blue_wires:
            assert np.isclose(wire_sum, 4)
        # yellow wires should be 2/3 chance each
        elif wire_rank in yellow_wires:
            assert np.isclose(wire_sum, 2.0 / 3.0), wire_sum
        # red wires should be 1/2 chance each
        elif wire_rank in red_wires:
            assert np.isclose(wire_sum, 1.0 / 2.0), wire_sum
        else:
            pytest.fail(f'unexpected wire rank {wire_rank}')
