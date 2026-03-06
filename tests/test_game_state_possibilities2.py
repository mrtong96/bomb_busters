import time

import numpy as np

from src.constraints2 import Constraint, IndicatorConstraint
from src.game_state_possibilities2 import compute_probability_matrices


def test_sanity_combination_checks():
    test_cases = [
        # max 2 wires of each type, 3 total spaces, valid combos are 1,1,2/1,2,2
        (np.array([3, 3]), {i + 1: (3, 3) for i in range(2)}),
        # Can only select one wire at a time, uniform distribution
        (np.array([1, 1, 1, 1]), {i + 1: (1, 1) for i in range(4)}),
        # Scale test
        (np.array([2, 2, 2, 2]), {i + 1: (2, 2) for i in range(4)}),
        # Bigger scale test
        (np.array([10, 10, 10, 9, 9]), {i + 1: (4, 4) for i in range(12)}),
    ]

    for wire_limits_per_player, wire_limits in test_cases:
        results = compute_probability_matrices(wire_limits_per_player, wire_limits)
        density_matrix = results[0]
        combinations_matrix = results[1]

        # all the max limits per wire are equal, there should be symmetry among players with equal hand sizes
        if len(set([el[0] for el in wire_limits.values()])) == 1:
            for i in range(1, len(wire_limits_per_player)):
                if wire_limits_per_player[i] != wire_limits_per_player[0]:
                    continue
                if not np.all(np.isclose(density_matrix[0], density_matrix[i])):
                    assert False, ('failed symmetry', density_matrix)
                if not np.all(np.isclose(combinations_matrix[0], combinations_matrix[i])):
                    assert False, ('failed symmetry', combinations_matrix)

        shannon_entropy = 0
        for row in density_matrix[0, :, :]:
            for el in row:
                if el == 0:
                    continue
                shannon_entropy += - el * np.log2(el)

        # magic number, for a 10-card hand with all blue wires
        if np.isclose(np.sum(density_matrix), 48):
            assert np.isclose(23.076475108967493, shannon_entropy), shannon_entropy

def test_constraint_tests():
    wires = 4

    t0 = time.time()

    # compute the wire limits per player and the individual wire limits for the remaining cards
    wire_limits_per_player = np.array([((wires * 4) + 4 - i) // 5 for i in range(5)])
    wire_limits = {i: (4, 4) for i in range(wires)}

    constraints = [np.ones(limit) * Constraint.EMPTY for limit in wire_limits_per_player]
    # have one of each wire for the first player
    constraints[0] = np.array(list(range(len(constraints[0]))))
    wire_ranks = sorted(wire_limits.keys())
    indicator_constraint = IndicatorConstraint(wire_limits_per_player, wire_ranks, constraints)

    # profile things
    results = compute_probability_matrices(
        wire_limits_per_player=wire_limits_per_player,
        wire_limits=wire_limits,
        constraints=[indicator_constraint],
    )

    if not np.all(np.isclose(results[0][0], np.identity(wires))):
        print(results[0][0])
        assert False, "If the first player has one of each wire, then the wire density matrix should be the identity matrix"

    # print(time.time() - t0)

def test_constraint_tests2():
    wires = 12

    t0 = time.time()

    # compute the wire limits per player and the individual wire limits for the remaining cards
    wire_limits_per_player = np.array([((wires * 4) + 4 - i) // 5 for i in range(5)])
    wire_limits = {i: (4, 4) for i in range(wires)}

    for i in range(10):
        constraints = [np.ones(limit) * Constraint.EMPTY for limit in wire_limits_per_player]
        # have one of each wire for the first player

        deck = np.array([i for i in range(wires) for _ in range(4)])
        hand = np.random.choice(deck, size=((wires * 4) + 4) // 5, replace=False, p=None)
        constraints[0] = np.array(list(sorted(hand)))
        wire_ranks = sorted(wire_limits.keys())
        indicator_constraint = IndicatorConstraint(wire_limits_per_player, wire_ranks, constraints)
        # profile things
        _ = compute_probability_matrices(
            wire_limits_per_player=wire_limits_per_player,
            wire_limits=wire_limits,
            constraints=[indicator_constraint],
        )


if __name__ == "__main__":
    test_sanity_combination_checks()
    test_constraint_tests()
    test_constraint_tests2()
