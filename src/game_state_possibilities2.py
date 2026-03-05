# now that we wrote the first one write the other one but (hopefully) way faster
import functools
import itertools
import time
from typing import Optional, Iterator
from collections import Counter, defaultdict

import numpy as np
from numba import njit

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
        wires = tuple(sorted(constraint.wires))
        wire_intersection = tuple(sorted(set(constraint.wires).intersection(wires)))
        tuple_args.append((wires, wire_intersection))
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

def _accumulate_base_case(
    distributions: np.ndarray,
    counts: np.ndarray,
    wire_array_index: int,
    wire_placement_mapping: dict,
    player_range: np.ndarray,
    indicator_constraint,
    density_tensor: np.ndarray,
    combinations_tensor: np.ndarray,
    weight: np.ndarray,
    combinations: np.ndarray,
) -> None:
    """
    Base case accumulation (first wire rank in a subset).
    Prev-filled slots are all zero, so distributions map directly to placement indices.
    Mutates density_tensor, combinations_tensor, weight, and combinations in-place.
    """
    if indicator_constraint:
        constraint_valid_mask = np.all(
            indicator_constraint.constraint_matrix[
                player_range[None, :],  # (1, num_players) broadcast to (num_distributions, num_players)
                wire_array_index,
                0,                      # prefilled = 0 in base case
                distributions,          # (num_distributions, num_players)
            ],
            axis=1,
        )
        valid_distributions = distributions[constraint_valid_mask]
        valid_counts = counts[constraint_valid_mask]
    else:
        valid_distributions, valid_counts = distributions, counts

    if len(valid_distributions) == 0:
        return

    placement_indices = np.array([wire_placement_mapping[tuple(d)] for d in valid_distributions])
    weight[placement_indices] += valid_counts
    combinations[placement_indices] += 1

    # Scatter density update: build all (placement, player, slot) indices at once
    wires_per_distribution = valid_distributions.sum(axis=1)
    placement_index_repeated = np.repeat(placement_indices, wires_per_distribution)
    player_index_repeated    = np.concatenate([np.repeat(player_range, d) for d in valid_distributions])
    slot_index_repeated      = np.concatenate([np.concatenate([np.arange(el) for el in d]) for d in valid_distributions])
    count_repeated           = np.repeat(valid_counts, wires_per_distribution)
    # indices are unique (each dist maps to a distinct placement, slots within a player are distinct)
    density_tensor[placement_index_repeated, player_index_repeated, slot_index_repeated, wire_array_index] += count_repeated

    # Scatter combinations update
    nonzero_player_lists        = [np.where(d > 0)[0] for d in valid_distributions]
    nonzero_placement_indices   = np.concatenate([np.full(len(nz), placement, dtype=np.intp) for placement, nz in zip(placement_indices, nonzero_player_lists)])
    nonzero_player_indices      = np.concatenate(nonzero_player_lists)
    nonzero_distribution_values = np.concatenate([valid_distributions[i][nz] - 1 for i, nz in enumerate(nonzero_player_lists)])
    nonzero_counts              = np.concatenate([np.full(len(nz), count) for count, nz in zip(valid_counts, nonzero_player_lists)])
    combinations_tensor[nonzero_placement_indices, nonzero_player_indices, wire_array_index, nonzero_distribution_values] += nonzero_counts


@njit(cache=True)
def _accumulate_recursive_case_jit(
    distributions: np.ndarray,         # (num_dists, num_players) int8
    counts: np.ndarray,                 # (num_dists,) int32
    wire_array_index: int,
    wire_placements_matrix: np.ndarray, # (num_placements, num_players) int8
    prev_placement_lookup: np.ndarray,  # (max_encoded_key + 1,) int32
    encoding_powers: np.ndarray,        # (num_players,) int64
    prev_density: np.ndarray,           # (num_prev, num_players, max_wires, num_wire_types) float64
    prev_combinations_tensor: np.ndarray, # (num_prev, num_players, num_wire_types, 4) float64
    prev_combinations: np.ndarray,      # (num_prev,) float64
    prev_weight: np.ndarray,            # (num_prev,) float64
    constraint_matrix: np.ndarray,      # (num_players, num_wire_types, max_wires+1, 5) bool; ignored if not has_constraint
    has_constraint: bool,
    density_tensor: np.ndarray,         # (num_placements, num_players, max_wires, num_wire_types) float64 — mutated
    combinations_tensor: np.ndarray,    # (num_placements, num_players, num_wire_types, 4) float64 — mutated
    weight: np.ndarray,                 # (num_placements,) float64 — mutated
    combinations: np.ndarray,           # (num_placements,) float64 — mutated
) -> None:
    num_placements = wire_placements_matrix.shape[0]
    num_players    = wire_placements_matrix.shape[1]
    num_dists      = distributions.shape[0]
    max_wires      = density_tensor.shape[2]
    num_wire_types = density_tensor.shape[3]

    prev_filled_buf = np.empty(num_players, dtype=np.int64)

    for d_idx in range(num_dists):
        cur_count_f = float(counts[d_idx])

        for p_idx in range(num_placements):
            # Capacity check: compute prev_filled for each player, bail early if any is negative
            valid = True
            prev_key = np.int64(0)
            for player in range(num_players):
                pf = np.int64(wire_placements_matrix[p_idx, player]) - np.int64(distributions[d_idx, player])
                if pf < 0:
                    valid = False
                    break
                prev_filled_buf[player] = pf
                prev_key += pf * encoding_powers[player]
            if not valid:
                continue

            prev_idx = prev_placement_lookup[prev_key]

            # Indicator constraint check
            if has_constraint:
                constraint_valid = True
                for player in range(num_players):
                    nw = int(distributions[d_idx, player])
                    if not constraint_matrix[player, wire_array_index, prev_filled_buf[player], nw]:
                        constraint_valid = False
                        break
                if not constraint_valid:
                    continue

            contribution = cur_count_f * prev_weight[prev_idx]
            weight[p_idx]       += contribution
            combinations[p_idx] += prev_combinations[prev_idx]

            # Dot-product: accumulate scaled previous-level tensors into current level
            for player in range(num_players):
                for slot in range(max_wires):
                    for t in range(num_wire_types):
                        density_tensor[p_idx, player, slot, t] += prev_density[prev_idx, player, slot, t] * cur_count_f
                for t in range(num_wire_types):
                    for k in range(4):
                        combinations_tensor[p_idx, player, t, k] += prev_combinations_tensor[prev_idx, player, t, k] * cur_count_f

            # Scatter: write current wire rank's contribution into the appropriate slots
            for player in range(num_players):
                nw = int(distributions[d_idx, player])
                if nw == 0:
                    continue
                pf = int(prev_filled_buf[player])
                for slot_offset in range(nw):
                    density_tensor[p_idx, player, pf + slot_offset, wire_array_index] += contribution
                combinations_tensor[p_idx, player, wire_array_index, nw - 1] += contribution


def _accumulate_recursive_case(
    distributions: np.ndarray,
    counts: np.ndarray,
    wire_array_index: int,
    wire_placements_matrix: np.ndarray,
    prev_info: tuple,
    wire_limits_per_player: np.ndarray,
    num_players: int,
    player_range: np.ndarray,
    indicator_constraint,
    density_tensor: np.ndarray,
    combinations_tensor: np.ndarray,
    weight: np.ndarray,
    combinations: np.ndarray,
) -> None:
    """
    Recursive case accumulation (subsequent wire ranks in a subset).
    Prepares array inputs and delegates to the JIT-compiled inner function.
    Mutates density_tensor, combinations_tensor, weight, and combinations in-place.
    """
    prev_placements_matrix = prev_info[1]

    # Build O(1) lookup: encode each prev_placement tuple as a base-M integer
    encoding_base   = int(max(wire_limits_per_player)) + 1
    encoding_powers = (encoding_base ** np.arange(num_players)).astype(np.int64)
    prev_placements_encoded = (prev_placements_matrix.astype(np.int64) @ encoding_powers)
    prev_placement_lookup   = np.full(int(prev_placements_encoded.max()) + 1, -1, dtype=np.int32)
    prev_placement_lookup[prev_placements_encoded] = np.arange(len(prev_placements_matrix), dtype=np.int32)

    if indicator_constraint is not None:
        constraint_matrix = indicator_constraint.constraint_matrix.astype(np.bool_)
        has_constraint = True
    else:
        constraint_matrix = np.empty((0, 0, 0, 0), dtype=np.bool_)
        has_constraint = False

    _accumulate_recursive_case_jit(
        distributions, counts, wire_array_index,
        wire_placements_matrix, prev_placement_lookup, encoding_powers,
        prev_info[2], prev_info[3], prev_info[4], prev_info[5],
        constraint_matrix, has_constraint,
        density_tensor, combinations_tensor, weight, combinations,
    )


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

    for wire_subset_key, wire_mapping_tuples in sorted(wire_key_mapping.items()):
        # wire placement mapping to the output
        len_subset, subset_total_wires, latest_wire_count = wire_subset_key[:3]
        wire_placements = list(get_wire_placements(total_wires, subset_total_wires, num_players))
        distributions, counts = get_single_wire_rank_distributions(latest_wire_count, num_players)

        # initialize all the variables used to hold aggregation outputs
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
        player_range = np.arange(num_players)

        for wire_mapping_tuple in wire_mapping_tuples:
            prev_wire_subset_key, wire_rank = wire_mapping_tuple
            wire_array_index = len_subset - 1

            if len_subset == 1:
                _accumulate_base_case(
                    distributions, counts, wire_array_index,
                    wire_placement_mapping, player_range, indicator_constraint,
                    density_tensor, combinations_tensor, weight, combinations,
                )
            else:
                _accumulate_recursive_case(
                    distributions, counts, wire_array_index,
                    wire_placements_matrix, wire_placement_dict[prev_wire_subset_key],
                    wire_limits_per_player, num_players, player_range, indicator_constraint,
                    density_tensor, combinations_tensor, weight, combinations,
                )

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
        for player_data in density_matrix:
            for row in player_data:
                for el in row:
                    if el == 0:
                        continue
                    shannon_entropy += el * np.log2(el)

        print(shannon_entropy)
        print(time.time() - t0)

if __name__ == "__main__":
    sanity_combination_checks()
    constraint_tests()
    constraint_tests2()