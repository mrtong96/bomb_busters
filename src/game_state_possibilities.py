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
def get_single_wire_rank_distributions(num_wires: int, num_players: int) -> tuple:
    """
    Given num_wires limit to distribute among num_players, enumerate through the balls and
    bins problem to return a (ncr(num_wires + num_players - 1, num_wires) x num_players)
    matrix of the counts.

    :param num_wires: number of wires
    :param num_players: number of players

    Returns: a tuple of the distributions and counts arrays
    """
    if num_wires == 0:
        counter = Counter([tuple([0 for _ in range(num_players)])])

    else:
        counts = list()
        for combo in itertools.product(range(num_players), repeat=num_wires):
            count = Counter(list(combo))
            counts.append(tuple([count.get(i, 0) for i in range(num_players)]))
        counter = Counter(counts)

    distributions, counts = list(zip(*counter.items()))
    return np.array(distributions, dtype=np.int32), np.array(counts, dtype=np.int64)

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

    def compute_probability_matrices_dp():
        n_wire_types = len(wire_array)
        n_players = num_players

        initial_remaining = tuple(int(x) for x in wire_limits_per_player)
        initial_constraints = tuple(mutating_constraints)

        # Forward pass: enumerate all reachable states per level via BFS
        states_per_level = [set() for _ in range(n_wire_types + 1)]
        states_per_level[0].add((initial_remaining, initial_constraints))

        for k in range(n_wire_types):
            lower, upper = int(lower_wire_array_limits[k]), int(upper_wire_array_limits[k])
            for (rem, cstr) in states_per_level[k]:
                rem_arr = np.array(rem, dtype=np.int32)
                for num_wires in range(lower, upper + 1):
                    dist_array, _ = get_single_wire_rank_distributions(num_wires, n_players)
                    valid_mask = ~np.any(dist_array > rem_arr, axis=1)
                    for constraint in fixed_constraints:
                        valid_mask[valid_mask] &= constraint.vectorized_is_valid(k, rem, dist_array[valid_mask], None)
                    for idx in np.where(valid_mask)[0]:
                        dist = dist_array[idx]
                        cargs = [k, rem, dist, False]
                        if cstr and any(not c.is_valid(*cargs) for c in cstr):
                            continue
                        new_cstr = tuple(c.mutate_constraint(*cargs) for c in cstr) if cstr else cstr
                        new_rem = tuple((rem_arr - dist).tolist())
                        states_per_level[k + 1].add((new_rem, new_cstr))

        # Backward pass: compute values bottom-up
        dp = {}

        # Terminal base cases
        for (rem, cstr) in states_per_level[n_wire_types]:
            dm = [np.zeros((rem[p], n_wire_types), dtype=np.float64) for p in range(n_players)]
            cm = [np.zeros((n_wire_types, 4), dtype=np.float64) for p in range(n_players)]
            cargs = [None, rem, tuple(0 for _ in range(n_players)), True]
            ok = all(c.is_valid(*cargs) for c in cstr)
            dp[(rem, cstr)] = (dm, cm, 1.0 if ok else 0.0, 1 if ok else 0)

        for k in range(n_wire_types - 1, -1, -1):
            is_last = (k == n_wire_types - 1)
            lower, upper = int(lower_wire_array_limits[k]), int(upper_wire_array_limits[k])
            for (rem, cstr) in states_per_level[k]:
                dm = [np.zeros((rem[p], n_wire_types), dtype=np.float64) for p in range(n_players)]
                cm = [np.zeros((n_wire_types, 4), dtype=np.float64) for p in range(n_players)]
                weight, combos = 0.0, 0
                rem_arr = np.array(rem, dtype=np.int32)
                for num_wires in range(lower, upper + 1):
                    dist_array, counts_array = get_single_wire_rank_distributions(num_wires, n_players)
                    valid_mask = ~np.any(dist_array > rem_arr, axis=1)
                    for constraint in fixed_constraints:
                        valid_mask[valid_mask] &= constraint.vectorized_is_valid(k, rem, dist_array[valid_mask], None)
                    for idx in np.where(valid_mask)[0]:
                        dist = dist_array[idx]
                        count = int(counts_array[idx])
                        cargs = [k, rem, dist, False]
                        if cstr and any(not c.is_valid(*cargs) for c in cstr):
                            continue
                        new_cstr = tuple(c.mutate_constraint(*cargs) for c in cstr) if cstr else cstr
                        new_rem = tuple((rem_arr - dist).tolist())
                        sub = dp[(new_rem, new_cstr)]
                        if sub[2] == 0:
                            continue
                        cur_weight = count * sub[2]
                        weight += cur_weight
                        combos += sub[3]
                        for p in range(n_players):
                            wpf = int(dist[p])
                            if not is_last:
                                dm[p][wpf:] += sub[0][p] * count
                                cm[p] += sub[1][p] * count
                            if wpf > 0:
                                dm[p][:wpf, k] += cur_weight
                                cm[p][k, wpf - 1] += cur_weight
                dp[(rem, cstr)] = (dm, cm, weight, combos)
            for key in states_per_level[k + 1]:
                del dp[key]

        return dp[(initial_remaining, initial_constraints)]

    helper_results = compute_probability_matrices_dp()

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