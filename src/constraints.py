# class of constraints
from abc import ABC
import numpy as np

EMPTY = -1

class Constraint(ABC):
    """
    Constraints for the wires

    TODO:
        add wire_limits_per_player as a constructor arg
        optimize the calls of indicator constraint to cache a lot of the outputs instead of going through the for loop
    """
    EMPTY = EMPTY

    def __init__(self, wire_limits_per_player: np.array, wire_ranks: list[int]):
        self.wire_limits_per_player = wire_limits_per_player
        self.wire_ranks = wire_ranks

class IndicatorConstraint(Constraint):
    """
    Meant for wires with a fixed position
    """
    def __init__(self, wire_limits_per_player: np.array, wire_ranks: list[int], constraints: list[np.array]):
        super().__init__(wire_limits_per_player, wire_ranks)

        self.constraints = [c.astype(np.int8) for c in constraints]

        self.constraint_matrix = np.zeros(
            (len(self.wire_limits_per_player), len(wire_ranks), max(self.wire_limits_per_player) + 1, 5),
            dtype=np.bool
        )

        # padded matrix of whether every possible wire constraint is valid for this particular set of indicator constraints or not
        self.constraint_matrices = []
        for player_index, wires_per_player in enumerate(self.wire_limits_per_player):
            for wire_rank_index, wire_rank in enumerate(wire_ranks):
                for prefilled_wires in range(wires_per_player + 1):
                    for num_wires in range(5):
                        satisfies_constraint = True

                        for offset in range(num_wires):
                            index = prefilled_wires + offset
                            # go off the end of the max possible wires, false
                            if index >= wires_per_player:
                                satisfies_constraint = False
                            # spot is empty, continue
                            elif self.constraints[player_index][index] == self.EMPTY:
                                continue
                            # violates constraint
                            elif self.constraints[player_index][index] != self.wire_ranks[wire_rank_index]:
                                satisfies_constraint = False
                        self.constraint_matrix[player_index, wire_rank_index, prefilled_wires, num_wires] = satisfies_constraint
    def is_valid(
            self,
            wire_rank_index: int,
            prefilled_wires: np.array,
            wire_distribution: np.array,
    ) -> bool:
        # TODO: write the vectorized version of this function

        return np.all(self.constraint_matrix[
                          np.arange(len(self.wire_limits_per_player)),
                          wire_rank_index,
                          prefilled_wires,
                          wire_distribution
                      ])

class SubsetConstraint(Constraint):
    """
    Meant to represent yellow/red wires
    """

    def __init__(self, wire_limits_per_player: np.array, wire_ranks: list[int], wires: list[int], num_wires: int):
        super().__init__(wire_limits_per_player, wire_ranks)
        self.wires = wires
        self.num_wires = num_wires