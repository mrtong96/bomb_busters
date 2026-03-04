import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import functools
import itertools
from collections import Counter

from src.constraints import Constraint, IndicatorConstraint, SubsetConstraint


@dataclass
class Player:
    # [0, n) player index
    player_index: int
    # wires in the player's hand
    player_wires: np.array

    # (num_players - 1 length) list of [max_wires_in_hand, num_possible_wires]
    # probability density matrices of each other players' wire
    wire_density_matrices: list[np.array]
    # (num_players - 1 length) list of [num_possible_wires, 4]
    # probability density matrices of  each player having single/double/etc of wires
    wire_count_matrices: np.array

@dataclass
class GameState:
    # total number of wires for n players
    num_wires: int

    # total number of players
    num_players: int

    players: list[Player]

    # limits for each wire type
    wire_limits: dict[int, int]

@functools.cache
def get_single_wire_rank_distributions(num_wires: int, num_players: int) -> Counter:
    """
    Given num_wires limit to distribute among num_players, enumerate through the balls and
    bins problem to return a (ncr(num_wires + num_players - 1, num_wires) x num_players)
    matrix of the counts.

    :param num_wires: number of wires
    :param num_players: number of players
    """
    if num_wires == 0:
        return Counter([tuple([0 for _ in range(num_players)])])

    counts = list()
    for combo in itertools.product(range(num_players), repeat=num_wires):
        count = Counter(list(combo))
        counts.append(tuple([count.get(i, 0) for i in range(num_players)]))

    return Counter(counts)

def compute_probability_matrices(
    wire_limits_per_player: np.array,
    wire_limits: dict[int, tuple[int, int]],
    constraints: Optional[list[Constraint]] = None,
) -> tuple[list[np.array], list[np.array], float, int]:
    """
    way to recursively compute wire limits
    TODO: add constraints

    Args:
        :param player_index: index of the player
        :param wire_limits_per_player: number of wires you can place per player (not self)
        :param wire_limits: dictionary of wire limits [min, max] for the sum of the remaining players
        :param constraints: additional constraints for the hand
    """
    # express this as an array for future jit
    num_players = len(wire_limits_per_player)
    wire_array, wire_array_limits = list(zip(*sorted(wire_limits.items(), key=lambda x: x[0])))
    wire_array = np.array(wire_array, np.int8)
    lower_wire_array_limits = np.array([el[0] for el in wire_array_limits], np.float64)
    upper_wire_array_limits = np.array([el[1] for el in wire_array_limits], np.float64)

    constraints = constraints or []
    fixed_constraints = [el for el in constraints if not el.mutates]
    mutating_constraints = tuple([el for el in constraints if el.mutates])

    @functools.cache
    def compute_probability_matrices_helper(
        remaining_wires: tuple[int, ...],
        wire_array_index: int,
        cur_mutating_constraints: tuple[Constraint],
    ) -> tuple[list[np.array], list[np.array], float, int]:
        """
        Helper function that computes all the cached values.
        Computes things one wire type at a time

        Args:
            :param remaining_wires: number of wires for each remaining player
            :param wire_array_index: int in [0, len(wire_array-1)). Which wire to use
            :param cur_mutating_constraints: additional constraints for the hands

        Returns:
            tuple of:
                wire_density_matrices, list of matrices describing the wire density per position
                wire_count_matrices, list of matrices describing the probability of having x wires per player
                weight, total weight of combinations
                combinations, total number of unique combinations
        """
        wire_density_matrices = [
            np.zeros((player_limit, len(wire_array)), dtype=np.float64)
            for player_limit in remaining_wires
        ]
        wire_count_matrices = [
            np.zeros((len(wire_array), 4), dtype=np.float64)
            for _ in remaining_wires
        ]
        combinations = 0
        weight = 0.0

        # we reached the end of wires
        if wire_array_index == len(wire_array):
            constraint_args = [
                None,
                remaining_wires,
                tuple(0 for _ in range(num_players)),
                True,
            ]
            is_valid = all([constraint.is_valid(*constraint_args) for constraint in cur_mutating_constraints])
            weight = 1.0 if is_valid else 0.0
            combinations = 1 if is_valid else 0
            return wire_density_matrices, wire_count_matrices, weight, combinations

        lower_wires = int(lower_wire_array_limits[wire_array_index])
        upper_wires = int(upper_wire_array_limits[wire_array_index])
        for num_wires_to_distribute in range(lower_wires, upper_wires + 1):
            wire_distributions = get_single_wire_rank_distributions(
                num_wires=int(num_wires_to_distribute),
                num_players=len(remaining_wires)
            )

            for wire_distribution, count in wire_distributions.items():
                # if there are any cases where we have too many wires to distribute towards a player
                if any(r < d for r, d in zip(remaining_wires, wire_distribution)):
                    continue

                # arguments to evaluate every constraint
                constraint_args = [
                    wire_array_index,
                    remaining_wires,
                    wire_distribution,
                    False,
                ]
                # if a constraint is violated, continue
                all_constraints = fixed_constraints + list(cur_mutating_constraints)
                if any(not constraint.is_valid(*constraint_args) for constraint in all_constraints):
                    continue

                # construct the mutating constraints
                recursive_constraints = []
                for constraint in cur_mutating_constraints:
                    recursive_constraints.append(constraint.mutate_constraint(*constraint_args))

                # recursive case, compute sub cases
                sub_results = compute_probability_matrices_helper(
                    remaining_wires = tuple(r - d for r, d in zip(remaining_wires, wire_distribution)),
                    wire_array_index=wire_array_index + 1,
                    cur_mutating_constraints=tuple(recursive_constraints),
                )

                # if the sub results are impossible, continue
                if sub_results[2] == 0:
                    continue

                cur_weight = count * sub_results[2]
                weight += cur_weight
                combinations += sub_results[3]

                for index, wires_for_player in enumerate(wire_distribution):
                    if wire_array_index != len(upper_wire_array_limits) - 1:
                        wire_density_matrices[index][wires_for_player:] += sub_results[0][index] * count
                        wire_count_matrices[index] += sub_results[1][index] * count
                    if wires_for_player > 0:
                        wire_density_matrices[index][:wires_for_player, wire_array_index] += cur_weight
                        wire_count_matrices[index][wire_array_index, wires_for_player - 1] += cur_weight

        return wire_density_matrices, wire_count_matrices, weight, combinations

    helper_results = compute_probability_matrices_helper(
        remaining_wires=tuple(wire_limits_per_player),
        wire_array_index=0,
        cur_mutating_constraints=tuple(mutating_constraints),
    )

    for player_index in range(num_players):
        # normalize the helper matrix weights
        helper_results[0][player_index] /= helper_results[2]
        helper_results[1][player_index] /= helper_results[2]

        # do some validation
        if not np.isclose(np.sum(helper_results[0][player_index]), wire_limits_per_player[player_index]):
            raise RuntimeError(f"probabilities do not seem to sum to expected num of wires, {np.sum(helper_results[0][player_index])}")
        expected_wires = 0.0
        for i in range(4):
            expected_wires += (i + 1) * np.sum(helper_results[1][player_index][:, i])
        if not np.isclose(expected_wires, wire_limits_per_player[player_index]):
            raise RuntimeError("expected wires do not seem to sum to wire limits")

    return (helper_results[0], helper_results[1], helper_results[2], helper_results[3])

@functools.cache
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

@functools.cache
def ncr(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))

def main():
    # small scale
    n = 2
    results = compute_probability_matrices(
        wire_limits_per_player=np.array([n for _ in range(4)]),
        wire_limits={i: (4, 4) for i in range(n)}
    )
    print(results[0][0])
    print(results[1][0])
    print(results[2], results[3])
    assert results[2] == factorial(8) / (2 ** 4)
    # 4 1s, 4*3 211, 6 22,
    assert results[3] == 1 + 12 + 6

    # bigger scale, 40 cards
    t0 = time.time()
    n = 10
    results = compute_probability_matrices(
        wire_limits_per_player=np.array([n for _ in range(4)]),
        wire_limits={i: (4, 4) for i in range(n)}
    )
    print(results[0][0])
    print(results[1][0])
    print(results[2], results[3])
    print(time.time() - t0)

    # sample hands
    t0 = time.time()

    for attempt in range(10):
        # build the deck, select 10 cards as the dealer
        deck = np.array([i for i in range(12) for _ in range(4)])
        hand = np.random.choice(deck, size=10, replace=False, p=None)
        # compute the wire limits per player and the individual wire limits for the remaining cards
        wire_limits_per_player = np.array([(48 + 4 - i) // 5 for i in range(5)])
        wire_limits = {i: 4 for i in range(12)}
        for wire in hand:
            wire_limits[wire] -= 1
        wire_limits = {k: (v, v) for k, v in wire_limits.items()}
        # profile things
        results = compute_probability_matrices(
            wire_limits_per_player=wire_limits_per_player[1:],
            wire_limits=wire_limits
        )
        print(attempt, time.time() - t0, results[2], results[3])
    print(time.time() - t0)

    # biggest blue wire scale, all 5 players
    t0 = time.time()
    _ = compute_probability_matrices(
        wire_limits_per_player=np.array([(48 + 4 - i) // 5 for i in range(5)]),
        wire_limits={i: (4, 4) for i in range(12)},
    )
    print(time.time() - t0)

def constraint_tests():
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

    print(time.time() - t0)

def constraint_tests2():
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
        print(i, time.time() - t0)

    print(time.time() - t0)

if __name__ == '__main__':
    constraint_tests()
    constraint_tests2()