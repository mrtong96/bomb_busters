# class of constraints
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

class Constraint(ABC):
    """
    Constraints for the wires
    """
    EMPTY = -1

    def __init__(self):
        pass

    @abstractmethod
    def is_valid(
            self,
            wire_rank: int,
            wire_limits_per_player: np.array,
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ) -> bool:
        pass

    def mutate_constraint(
            self,
            wire_rank: int,
            wire_limits_per_player: np.array,
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
    def __init__(self, constraints: list[np.array]):
        super().__init__()

        self.constraints = constraints

    def is_valid(
            self,
            wire_rank: int,
            wire_limits_per_player: np.array,
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ) -> bool:
        # for each wire we are distributing to the players
        for player_index, player_wires in enumerate(wire_distribution):
            for offset in range(player_wires):
                wire_index = wire_limits_per_player[player_index] - remaining_wires[player_index] + offset
                constraint_rank = self.constraints[player_index][wire_index]

                if constraint_rank != self.EMPTY and constraint_rank != wire_rank:
                    return False
        return True

class SubsetConstraint(Constraint):
    """
    Meant to represent yellow/red wires
    """
    def __init__(self, wires: list[int], num_wires: int):
        super().__init__()
        self.wires = wires
        self.num_wires = num_wires

    def is_valid(
        self,
        wire_rank: int,
        wire_limits_per_player: np.array,
        remaining_wires: tuple[int, ...],
        wire_distribution: tuple[int, ...],
        is_terminal: bool,
    ) -> bool:
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
            wire_rank: int,
            wire_limits_per_player: np.array,
            remaining_wires: tuple[int, ...],
            wire_distribution: tuple[int, ...],
            is_terminal: bool,
    ):
        if is_terminal:
            raise RuntimeError("not implemented for this class")
        # we don't care, not an important wire
        elif wire_rank not in self.wires:
            return self

        wire_sum = sum(wire_distribution)
        if wire_sum == 0:
            return self

        remaining_wires = [el for el in self.wires if el != wire_rank]
        return SubsetConstraint(remaining_wires, self.num_wires - 1)

    def __eq__(self, other):
        return (
                sorted(self.wires) == sorted(other.wires)
                and self.num_wires == other.num_wires
        )

    def __hash__(self):
        return hash(('SubsetConstraint', tuple(sorted(self.wires)), self.num_wires))
