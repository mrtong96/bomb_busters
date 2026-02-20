from typing import Optional

import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
import functools

# wires are 10x rank + color, for example 4 yellow = 41
# can represent all of these as int8 arrays under the hood

class WireColor(Enum):
    BLUE = 0
    YELLOW = 1
    RED = 5

UNKNOWN = 0

@dataclass
class HandPossibilities:
    """
    Use this to store a bunch of pre-computed possible hands for the blue wires
    """
    # how many cards are in the hand
    num_wires: int
    # [NUM_WIRES] length array where the index of the element in the array maps to the index
    rank_to_index_mapping: np.array
    # num_cards * len(rank_to_index_mapping) matrix of the wire densities
    wire_density_matrix: np.array
    # num_cards * len(rank_to_index_mapping) * 4 matrix of the probability of having wire counts
    wire_count_matrix: np.array

def factorial(n):
    if n <= 1:
        return 1
    return factorial(n - 1) * n

@functools.cache
def ncr(n, r):
    return factorial(n) // factorial(r) // factorial(n - r)

def compute_hand_possibilities(
        num_wires: int,
        wire_limits: dict[int: int],
        fixed_wire_positions: Optional[dict[int, int]] = None,
) -> HandPossibilities:
    """
    Compute all the information about a hand assuming independence
    :param num_wires: how many wires are in the hand
    :param wire_limits: dictionary mapping rank -> max number of wires of that rank
    :param fixed_wire_positions: dictionary mapping position index -> required rank of the wire
    """
    # express this as an array for future jit
    wire_array, wire_array_limits = list(zip(*sorted(wire_limits.items(), key=lambda x: x[0])))
    wire_array = np.array(wire_array, np.int8)
    wire_array_limits = np.array(wire_array_limits, np.int64)
    total_possible_wires = len(wire_array)
    fixed_wire_positions = fixed_wire_positions or dict()

    @functools.cache
    def helper(remaining_wires: int, min_wire_index: int) -> tuple[np.array, np.array, int]:
        """
        Help compute all the math. Note the matrices are not normalized

        :param remaining_wires: number of wires remaining
        :param min_wire_index: index of the minimum wire rank that we can place

        :return: tuple of the (density_mat, count_mat, sub_weight)
            density_mat: num_cards * len(wire_array) matrix of the wire probability density functions
            count_mat: len(wire_array) * 4 matrix of the probability where count_mat[i,j] has the probability
                of the i-th card having (j+1)-wires
            sub_weight: Number of combinations for the current sub-routine call
        """
        density_mat = np.zeros((remaining_wires, total_possible_wires), dtype=np.int64)
        count_mat = np.zeros((total_possible_wires, 4), dtype=np.int64)
        sub_weight = 0

        # out of wires
        if remaining_wires == 0:
            return density_mat, count_mat, 1
        # at max wires placed. impossible
        elif min_wire_index == total_possible_wires - 1 and remaining_wires > wire_array_limits[-1]:
            return None, None, 0

        for cur_wire_index in range(min_wire_index, total_possible_wires):
            max_wires = min(remaining_wires, wire_array_limits[cur_wire_index])
            min_wires = max_wires if cur_wire_index == total_possible_wires - 1 else 1

            for cur_num_wires in range(min_wires, max_wires + 1):
                # violates fixed wire position constraint
                violates_fixed_position_constraint = False
                start_index = num_wires - remaining_wires
                for wire_num in range(cur_num_wires):
                    fixed_position_constraint = fixed_wire_positions.get(start_index + wire_num)
                    if fixed_position_constraint is not None and fixed_position_constraint != wire_array[cur_wire_index]:
                        violates_fixed_position_constraint = True
                        break
                if violates_fixed_position_constraint:
                    continue

                ncr_value = ncr(wire_array_limits[cur_wire_index], cur_num_wires)

                # we have the rest of the wires
                if cur_num_wires == remaining_wires:
                    density_mat[:, cur_wire_index] += ncr_value
                    count_mat[cur_wire_index, cur_num_wires - 1] += ncr_value
                    sub_weight += ncr_value
                    continue

                # get the sub results
                sub_results = helper(remaining_wires - cur_num_wires, cur_wire_index + 1)
                # if the sub results are impossible, continue
                if sub_results[2] == 0:
                    continue

                # update everything
                cur_weight = ncr_value * sub_results[2]
                density_mat[cur_num_wires:] += sub_results[0] * ncr_value
                density_mat[:cur_num_wires, cur_wire_index] += cur_weight
                count_mat += sub_results[1] * ncr_value
                count_mat[cur_wire_index, cur_num_wires - 1] += cur_weight
                sub_weight += cur_weight

        # if other constraints make this impossible
        if sub_weight == 0:
            return None, None, 0
        return density_mat, count_mat, sub_weight

    wire_density_matrix, wire_count_matrix, weight = helper(num_wires, 0)

    for wire_index in range(num_wires):
        if not(np.isclose(np.sum(wire_density_matrix[wire_index]), weight)):
            raise ValueError(f"wire density sum is off, {weight}, {np.sum(wire_density_matrix[wire_index])}")

    wire_density_matrix = wire_density_matrix.astype(np.float64)
    wire_count_matrix = wire_count_matrix.astype(np.float64)
    wire_density_matrix /= weight
    wire_count_matrix /= weight

    expected_wires = 0
    for i in range(4):
        expected_wires += (i + 1) * np.sum(wire_count_matrix[:, i])

    if not np.isclose(expected_wires, num_wires):
        raise ValueError(f"wire count matrix is off {expected_wires}")

    return HandPossibilities(
        num_wires= num_wires,
        rank_to_index_mapping = np.array(sorted(wire_limits.keys()), dtype=np.int8),
        wire_density_matrix= wire_density_matrix,
        wire_count_matrix=wire_count_matrix,
    )

def main():
    test_cases = [
        # max 2 wires of each type, 3 total spaces, valid combos are 1,1,2/1,2,2
        ({i + 1: 2 for i in range(2)}, 3),
        # Can only select one wire at a time, uniform distribution
        ({i + 1: 1 for i in range(8)}, 1),
        # Scale test
        ({i + 1: 2 for i in range(4)}, 4),
        # Bigger scale test
        ({i + 1: 4 for i in range(12)}, 10),
    ]

    for wires, num_wires in test_cases:
        print('case', wires, num_wires)

        t0 = time.time()
        hand_possibilities = compute_hand_possibilities(num_wires, wires)

        # all the max limits per wire are equal, there should be symmetry
        if len(set(wires.values())) == 1:
            for i in range(1, len(wires)):
                if not np.all(np.isclose(hand_possibilities.wire_count_matrix[0], hand_possibilities.wire_count_matrix[i])):
                    assert False, ('failed symmetry', hand_possibilities.wire_count_matrix)

        shannon_entropy = 0
        for row in hand_possibilities.wire_density_matrix:
            for el in row:
                if el == 0:
                    continue
                shannon_entropy += el * np.log2(el)

        print(hand_possibilities.wire_density_matrix)
        print(shannon_entropy)
        print(time.time() - t0)

if __name__ == "__main__":
    # 23.076475108967493
    main()
