# class of constraints
import functools
from abc import ABC, abstractmethod
from typing import Optional
from numba import jit
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

    @abstractmethod
    def is_valid(
            self,
            wire_rank_index: Optional[int],
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ) -> bool:
        pass

    def mutate_constraint(
            self,
            wire_rank_index: Optional[int],
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ):
        raise RuntimeError("not implemented for this class")

    @property
    def mutates(self) -> bool:
        return False

class IndicatorConstraint(Constraint):
    """
    Meant for wires with a fixed position
    """
    def __init__(self, wire_limits_per_player: np.array, wire_ranks: list[int], constraints: list[np.array]):
        super().__init__(wire_limits_per_player, wire_ranks)

        self.constraints = [c.astype(np.int8) for c in constraints]

        self.constraint_matrices = []
        for player_index, wires_per_player in enumerate(self.wire_limits_per_player):
            constraint_matrix = np.ones((len(wire_ranks), wires_per_player + 1, 5), dtype=np.bool)

            for wire_rank_index, wire_rank in enumerate(wire_ranks):
                for remaining_wires in range(wires_per_player + 1):
                    for num_wires in range(5):
                        satisfies_constraint = True

                        for offset in range(num_wires):
                            index = wires_per_player - remaining_wires + offset
                            if index >= wires_per_player:
                                satisfies_constraint = False
                            elif self.constraints[player_index][index] == self.EMPTY:
                                continue
                            elif self.constraints[player_index][index] != self.wire_ranks[wire_rank_index]:
                                satisfies_constraint = False

                        constraint_matrix[wire_rank_index, remaining_wires, num_wires] = satisfies_constraint
            self.constraint_matrices.append(constraint_matrix)

    def is_valid(
            self,
            wire_rank_index: Optional[int],
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ) -> bool:
        for player_index in range(len(self.wire_limits_per_player)):
            if not self.constraint_matrices[player_index][wire_rank_index, remaining_wires[player_index], wire_distribution[player_index]]:
                return False
        return True

class SubsetConstraint(Constraint):
    """
    Meant to represent yellow/red wires
    """
    def __init__(self, wire_limits_per_player: np.array, wire_ranks: list[int], wires: list[int], num_wires: int):
        super().__init__(wire_limits_per_player, wire_ranks)
        self.wires = wires
        self.num_wires = num_wires

    def is_valid(
        self,
        wire_rank_index: Optional[int],
        remaining_wires: tuple[int, ...],
        wire_distribution: tuple[int, ...],
        is_terminal: bool,
    ) -> bool:
        wire_rank = self.wire_ranks[wire_rank_index]

        # we're at the end, should have zero wires left
        if is_terminal:
            return self.num_wires == 0
        # we don't care, not an important wire
        elif wire_rank not in self.wires:
            return True

        # check the sum
        wire_sum = sum(wire_distribution)
        if wire_sum not in {0, 1}:
            raise RuntimeError("wires sum should be 0/1")

        # no wire to distribute, return whether we have space left
        if wire_sum == 0:
            return len(self.wires) < self.num_wires
        else:
            # do we have space left
            return self.num_wires > 0

    def mutate_constraint(
            self,
            wire_rank_index: Optional[int],
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ):
        wire_rank = self.wire_ranks[wire_rank_index]

        if is_terminal:
            raise RuntimeError("not implemented for this class")
        # we don't care, not an important wire
        elif wire_rank not in self.wires:
            return self

        wire_sum = sum(wire_distribution)
        if wire_sum == 0:
            return self

        remaining_wires = [el for el in self.wires if el != wire_rank]
        return SubsetConstraint(self.wire_limits_per_player, remaining_wires, self.num_wires - 1)

    def __eq__(self, other):
        return (
                sorted(self.wires) == sorted(other.wires)
                and self.num_wires == other.num_wires
        )

    def __hash__(self):
        return hash(('SubsetConstraint', tuple(sorted(self.wires)), self.num_wires))
