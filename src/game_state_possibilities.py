import time
from dataclasses import dataclass
import numpy as np
import functools
import itertools
from collections import Counter

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
    counts = list()
    for combo in itertools.product(range(num_players), repeat=num_wires):
        count = Counter(list(combo))
        counts.append(tuple([count.get(i, 0) for i in range(num_players)]))

    return Counter(counts)

def compute_probability_matrices(
    wire_limits_per_player: np.array,
    wire_limits: dict[int, int],
) -> tuple[list[np.array], list[np.array], float, int]:
    """
    way to recursively compute wire limits
    TODO: add constraints

    Args:
        :param player_index: index of the player
        :param wire_limits_per_player: number of wires you can place per player (not self)
        :param wire_limits: dictionary of wire limits for the sum of the remaining players
    """
    # express this as an array for future jit
    num_players = len(wire_limits_per_player)
    wire_array, wire_array_limits = list(zip(*sorted(wire_limits.items(), key=lambda x: x[0])))
    wire_array = np.array(wire_array, np.int8)
    wire_array_limits = np.array(wire_array_limits, np.float64)

    @functools.cache
    def compute_probability_matrices_helper(
        remaining_wires: tuple[int, ...],
        wire_array_index: int
    ) -> tuple[list[np.array], list[np.array], float, int]:
        """
        Helper function that computes all the cached values.
        Computes things one wire type at a time

        Args:
            :param remaining_wires: number of wires for each remaining player
            :param wire_array_index: int in [0, len(wire_array-1)). Which wire to use

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
            return wire_density_matrices, wire_count_matrices, 1.0, 1

        wire_distributions = get_single_wire_rank_distributions(
            num_wires=int(wire_array_limits[wire_array_index]),
            num_players=len(remaining_wires)
        )

        for wire_distribution, count in wire_distributions.items():
            # if there are any cases where we have too many wires to distribute towards a player
            if np.any(np.array(remaining_wires) < np.array(wire_distribution)):
                continue

            # TODO: add constraints.py checks here once we have them
            # recursive case, compute sub cases
            sub_results = compute_probability_matrices_helper(
                remaining_wires = tuple(np.array(remaining_wires) - np.array(wire_distribution)),
                wire_array_index=wire_array_index + 1
            )

            # if the sub results are impossible, continue
            if sub_results[2] == 0:
                continue

            cur_weight = count * sub_results[2]
            weight += cur_weight
            combinations += sub_results[3]

            for index, wires_for_player in enumerate(wire_distribution):
                if wire_array_index != len(wire_array_limits) - 1:
                    wire_density_matrices[index][wires_for_player:] += sub_results[0][index] * count
                    wire_count_matrices[index] += sub_results[1][index] * count
                if wires_for_player > 0:
                    wire_density_matrices[index][:wires_for_player, wire_array_index] += cur_weight
                    wire_count_matrices[index][wire_array_index, wires_for_player - 1] += cur_weight

        return wire_density_matrices, wire_count_matrices, weight, combinations

    helper_results = compute_probability_matrices_helper(
        remaining_wires=tuple(wire_limits_per_player),
        wire_array_index=0
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
        wire_limits={i: 4 for i in range(n)}
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
        wire_limits={i: 4 for i in range(n)}
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
        hand = np.random.choice(deck, size=10, replace=True, p=None)
        # compute the wire limits per player and the individual wire limits for the remaining cards
        wire_limits_per_player = np.array([(48 + 4 - i) // 5 for i in range(5)])
        wire_limits = {i: 4 for i in range(12)}
        for wire in hand:
            wire_limits[wire] -= 1
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
        wire_limits={i: 4 for i in range(12)},
    )
    print(time.time() - t0)



if __name__ == '__main__':
    main()