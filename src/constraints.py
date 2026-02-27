# class of constraints
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Constraint(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_positions(self)-> set[tuple[int, int]]:
        """
        Get all (player index, wire_position) tuples
        for which this constraint is relevant for
        """
        pass

    @abstractmethod
    def get_wires(self)-> set[int]:
        """
        Get all wire values for which this constraint is relevant for
        """
        pass

    @abstractmethod
    def is_valid(self, wires: list[np.array], is_partial_list: bool=False) -> bool:
        """
        Check the constraint
        """
        pass

class IndicatorConstraint(Constraint):
    """
    Single indicator tile at a position
    """

    def __init__(self, player_index: int, position_index: int, wire: int):
        super().__init__()
        self.player_index = player_index
        self.position_index = position_index
        self.wire = wire

    def get_positions(self) -> set[tuple[int, int]]:
        return {(self.player_index, self.position_index)}
        pass

    def get_wires(self) -> set[int]:
        return {self.wire}

    def is_valid(self, wires: list[np.array], is_partial_list: bool=False) -> bool:
        return wires[self.player_index][self.position_index] == self.wire

class SubsetConstraint(Constraint):
    """
    Constraint that a subset of wires exists. For example 2/3 yellow wires
    """

    def __init__(self, wires: list[int], num_wires: int):
        """
        Constructor

        :param wires: wires that we care about
        :param num_wires: the maximum number of wires that can exist from the subset
        """
        super().__init__()
        self.wires = set(wires)
        self.num_wires = num_wires

    def get_positions(self) -> set[tuple[int, int]]:
        return set()

    def get_wires(self) -> set[int]:
        return self.wires

    def is_valid(self, wires: list[np.array], is_partial_list: bool=False) -> bool:
        wire_set = set(np.concat(wires))
        if is_partial_list:
            return len(wire_set.intersection(self.wires)) < self.num_wires
        return len(wire_set.intersection(self.wires)) == self.num_wires




