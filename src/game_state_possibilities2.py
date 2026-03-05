# now that we wrote the first one write the other one but (hopefully) way faster
import functools
import itertools
import time
from typing import Optional, Iterator
from collections import Counter, defaultdict

import numpy as np

from src.constraints2 import Constraint, SubsetConstraint, IndicatorConstraint


def get_wire_placements(max_wires, wires, players) -> Iterator[tuple[int, ...]]:
    """
    Returns an iterator to go through all possible tuples of distributing wires to players

    :param max_wires: Total wires in play (including the ones not considering). Used to infer hand size
    :param wires: Wires to distribute
    :param players: How many players

    :return: an iterator of tuples to go through each wire placement
    """
    limits = [(max_wires + players - i - 1) // players for i in range(players)]

    def get_wire_placement_helper(helper_wires, helper_players):
        if helper_players == 1:
            if (wires == max_wires and helper_wires == limits[-1]) or helper_wires <= limits[-1]:
                yield(helper_wires,)
            return

        for num_wires in range(min(limits[-helper_players], helper_wires) + 1):
            for remainder in get_wire_placement_helper(helper_wires - num_wires, helper_players - 1):
                yield (num_wires,) + remainder

    return get_wire_placement_helper(wires, players)

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
    return np.array(distributions, dtype=np.int8), np.array(counts, dtype=np.int32)

def get_wire_combinations(
        total_wires: int, wire_limits: dict[int, tuple[int, int]], constraints: list[Constraint]
) -> list[list[tuple[int, int]]]:
    """
    :param total_wires: total wires that we must sum to
    :param wire_limits: wire rank -> (min/max) wire count
    :param constraints: constraints that we get. For now only look at the subset constraints
    :return: list of combos where each combo is a list of (wire, count) tuples
    """
    subset_constraints = [constraint for constraint in constraints if isinstance(constraint, SubsetConstraint)]

    wire_combinations = []
    for wire, (min_wires, max_wires) in wire_limits.items():
        wire_constraint = [(wire, cur_wires) for cur_wires in range(min_wires, max_wires + 1)]
        wire_combinations.append(wire_constraint)

    valid_wire_combinations = []
    for combo in itertools.product(*wire_combinations):
        combo = sorted(combo, key=lambda x: x[0])
        wires, counts = list(zip(*combo))
        # wrong number of wires, continue
        if sum(counts) != total_wires:
            continue

        # check each subset constraint
        passes_subset_constraints = True
        for subset_constraint in subset_constraints:
            if len(set(subset_constraint.wires).intersection(wires)) != subset_constraint.num_wires:
                passes_subset_constraints = False
                break
        if not passes_subset_constraints:
            continue

        valid_wire_combinations.append(combo)

    return valid_wire_combinations

def get_wire_placement_key(wire_subset: tuple[tuple[int, int], ...], constraints: list[Constraint]) -> tuple:
    """
    Aggregate the wire subset information into a key that can be used to cache data

    :param wire_subset: list of (rank, count) tuples
    :param constraints: all the constraints
    :return: some tuple to aggregate similar subsets to each other
    """
    subset_constraints = [constraint for constraint in constraints if isinstance(constraint, SubsetConstraint)]

    wires, counts = list(zip(*wire_subset))
    tuple_args = [len(wire_subset), sum(counts), counts[-1]]
    for constraint in subset_constraints:
        tuple_args.append(tuple(sorted(set(constraint.wires).intersection(wires))))
    return tuple(tuple_args)

def build_wire_key_mapping(wire_subsets: set[tuple[tuple[int, int], ...]], constraints: list[Constraint]) -> dict:
    """
    Build the key mapping to figure out which order to evaluate everything

    :param wire_subsets: set of collections of (rank, count) tuples
    :param constraints: all the constraints
    :return: map of (wire_subset_key: [tuple[num_wires, prev_wire_subset_key], ...])
    """
    wire_key_mapping = defaultdict(list)

    for wire_subset in wire_subsets:
        wire_subset_key = get_wire_placement_key(wire_subset, constraints)
        wires, counts = list(zip(*wire_subset))

        prev_wire_subset_key = None if len(wire_subset) == 1 else get_wire_placement_key(wire_subset[:-1], constraints)
        wire_key_mapping[wire_subset_key].append((prev_wire_subset_key, wires[-1]))

    return wire_key_mapping


def compute_probability_matrices(
    wire_limits_per_player: np.array,
    wire_limits: dict[int, tuple[int, int]],
    constraints: Optional[list[Constraint]] = None,
) -> tuple:
    """
    way to recursively compute wire limits

    Args:
        :param wire_limits_per_player: number of wires you can place per player (not self)
        :param wire_limits: dictionary of wire limits [min, max] for the sum of the remaining players
        :param constraints: additional constraints for the hand
    """
    # init stuff
    num_players = len(wire_limits_per_player)
    total_wires = sum(wire_limits_per_player)
    max_wires_per_player = max(wire_limits_per_player)
    num_possible_wires = len(wire_limits)
    constraints = constraints or []
    indicator_constraints = [constraint for constraint in constraints if isinstance(constraint, IndicatorConstraint)]
    if len(indicator_constraints) > 1:
        raise RuntimeError("Should be at most one indicator constraint")
    indicator_constraint = None if len(indicator_constraints) == 0 else indicator_constraints[0]

    # First compute all possible combinations of wires that can be placed via itertools product
    wire_combinations = get_wire_combinations(total_wires, wire_limits, constraints)
    wire_subsets = set([tuple(combo[:index + 1]) for combo in wire_combinations for index in range(len(combo))])
    wire_key_mapping = build_wire_key_mapping(wire_subsets, constraints)

    # Create a dictionary mapping [partial wire placement] -> vectorized state variable
    wire_placement_dict = dict()

    # state variable should have (wire_density tensor, wire_combinations tensor, weights, combinations)
        # wire_density_tensor maps (each combo, num_players, padded_wires_per_player, num_possible_wires)
        # wire_combinations_tensor maps (each combo, num_players, num_possible_wires, 4)
    for wire_subset_key, wire_mapping_tuples in sorted(wire_key_mapping.items()):
        # wire placement mapping to the output
        len_subset, subset_total_wires, latest_wire_count = wire_subset_key[:3]
        wire_placements = list(get_wire_placements(total_wires, subset_total_wires, num_players))
        distributions, counts = get_single_wire_rank_distributions(latest_wire_count, num_players)

        wire_placement_mapping = {wire_dist: i for i, wire_dist in enumerate(wire_placements)}
        wire_placements_matrix = np.array(wire_placements, dtype=np.int8)
        density_tensor = np.zeros(
            (len(wire_placements), num_players, max_wires_per_player, num_possible_wires),
            dtype=np.float64
        )
        combinations_tensor = np.zeros(
            (len(wire_placements), num_players, num_possible_wires, 4),
            dtype=np.float64
        )
        combinations = np.zeros(len(wire_placements), dtype=np.float64)
        weight = np.zeros(len(wire_placements), dtype=np.float64)

        for wire_mapping_tuple in wire_mapping_tuples:
            prev_wire_subset_key, wire_rank = wire_mapping_tuple

            # base case, just assign counts to the matrices without dot products
            if len_subset == 1:
                # rank of the wire
                wire_array_index = len_subset - 1

                # for each possible distribution of wires
                for cur_distribution, cur_count in zip(distributions, counts):
                    # check the indicator constraints
                    if indicator_constraint:
                        constraint_valid = indicator_constraint.is_valid(
                            wire_array_index,
                            np.zeros(num_players, dtype=np.int8),
                            cur_distribution
                        )
                        if not constraint_valid:
                            continue

                    # assign all the matrix info
                    placement_index = wire_placement_mapping[tuple(cur_distribution)]
                    player_indices = np.repeat(np.arange(num_players), cur_distribution)
                    wire_offsets = np.concatenate([np.arange(el) for el in cur_distribution])

                    density_tensor[
                        placement_index,
                        player_indices,
                        wire_offsets,
                        wire_array_index
                    ] += cur_count

                    non_zero_mask = cur_distribution != 0
                    combinations_tensor[
                        placement_index,
                        np.arange(num_players)[non_zero_mask],
                        wire_array_index,
                        cur_distribution[non_zero_mask] - 1
                    ] += cur_count
                    combinations[placement_index] += 1
                    weight[placement_index] += cur_count
            else:
                wire_array_index = len_subset - 1
                prev_info = wire_placement_dict[prev_wire_subset_key]

                # TODO: write this in more vectorized notation
                for placement_index, placement in enumerate(wire_placements):
                    placement_vector = wire_placements_matrix[placement_index]

                    for cur_distribution, cur_count in zip(distributions, counts):
                        # We are trying to place more wires than we have room for, continue
                        if np.any((placement_vector - cur_distribution) < 0):
                            continue

                        # assign all the matrix info
                        prev_filled = placement_vector - cur_distribution
                        prev_tuple = tuple(prev_filled)
                        prev_index = prev_info[0][prev_tuple]
                        prev_weight = prev_info[5][prev_index]

                        # check the indicator constraints
                        if indicator_constraint:
                            constraint_valid = indicator_constraint.is_valid(
                                wire_array_index,
                                prev_filled,
                                cur_distribution
                            )
                            if not constraint_valid:
                                continue

                        placement_index = wire_placement_mapping[tuple(placement_vector)]
                        player_indices = np.repeat(np.arange(num_players), cur_distribution)
                        prefilled_offsets = np.repeat(prev_filled, cur_distribution)
                        wire_offsets = np.concatenate([np.arange(el) for el in cur_distribution])
                        offsets = prefilled_offsets + wire_offsets

                        density_tensor[
                            placement_index,
                            player_indices,
                            offsets,
                            wire_array_index
                        ] += cur_count * prev_weight
                        non_zero_mask = cur_distribution != 0
                        combinations_tensor[
                            placement_index,
                            np.arange(num_players)[non_zero_mask],
                            wire_array_index,
                            cur_distribution[non_zero_mask] - 1
                        ] += cur_count * prev_weight
                        combinations[placement_index] += prev_info[4][prev_index]
                        weight[placement_index] += cur_count * prev_weight

                        # Multiply the new cases in as a dot product
                        density_tensor[placement_index] += prev_info[2][prev_index] * cur_count
                        combinations_tensor[placement_index] += prev_info[3][prev_index] * cur_count

            wire_placement_dict[wire_subset_key] = (
                wire_placement_mapping,
                wire_placements_matrix,
                density_tensor,
                combinations_tensor,
                combinations,
                weight,
            )

    # Aggregate all the info in one final result
    full_results = [data for tuple_key, data in wire_placement_dict.items() if tuple_key[0] == num_possible_wires]
    density_tensor = functools.reduce(lambda x, y: x + y, [el[2] for el in full_results])
    combinations_tensor = functools.reduce(lambda x, y: x + y, [el[3] for el in full_results])
    combinations = functools.reduce(lambda x, y: x + y, [el[4] for el in full_results])
    weight = functools.reduce(lambda x, y: x + y, [el[5] for el in full_results])

    # validate the shapes are ok
    if density_tensor.shape[0] != 1:
        raise RuntimeError("There should only be one possible way to fill in the wires at the terminal wire_placement_dict")
    elif combinations_tensor.shape[0] != 1:
        raise RuntimeError("There should only be one possible way to fill in the wires at the terminal wire_placement_dict")

    # normalize the data
    density_matrix = density_tensor[0] / weight[0]
    combinations_matrix = combinations_tensor[0] / weight[0]

    # sanity checks on the final results
    for player_index in range(num_players):
        # do some validation
        if not np.isclose(np.sum(density_matrix[player_index]), wire_limits_per_player[player_index]):
            raise RuntimeError(f"probabilities do not seem to sum to expected num of wires, {np.sum(density_matrix[player_index])}")
        expected_wires = 0.0
        for i in range(4):
            expected_wires += (i + 1) * np.sum(combinations_matrix[player_index][:, i])
        if not np.isclose(expected_wires, wire_limits_per_player[player_index]):
            raise RuntimeError("expected wires do not seem to sum to wire limits")

    # done
    return density_matrix, combinations_matrix, combinations[0], weight[0]


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

    for i in range(1):
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

def sanity_combination_checks():
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
        print('case', wire_limits_per_player, wire_limits)

        t0 = time.time()
        results = compute_probability_matrices(wire_limits_per_player, wire_limits)
        density_matrix = results[0]
        combinations_matrix = results[1]

        # all the max limits per wire are equal, there should be symmetry
        if len(set([el[0] for el in wire_limits.values()])) == 1:
            for i in range(1, len(wire_limits_per_player)):
                if not np.all(np.isclose(density_matrix[0], density_matrix[i])):
                    assert False, ('failed symmetry', density_matrix)
                if not np.all(np.isclose(combinations_matrix[0], combinations_matrix[i])):
                    assert False, ('failed symmetry', combinations_matrix)

        shannon_entropy = 0
        for player_data in density_matrix:
            for row in player_data:
                for el in row:
                    if el == 0:
                        continue
                    shannon_entropy += el * np.log2(el)

        print(shannon_entropy)
        print(time.time() - t0)

if __name__ == "__main__":
    # sanity_combination_checks()
    # constraint_tests()
    constraint_tests2()