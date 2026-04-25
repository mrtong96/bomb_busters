# class of constraints
from abc import ABC, abstractmethod
import numpy as np
from typing_extensions import deprecated

EMPTY = -1

class Constraint(ABC):

    @abstractmethod
    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        """
        Given a constraint_matrix of size (num_players, wire_ranks, wires_per_hand, 0-4 wires)
        Update the constraint matrix to apply constraints. Each method should only add false values
        """
        pass

class RankIndicatorConstraint(Constraint):
    """
    There exists an indicator tile that shows that there is a tile of either a blue wire (rank 1-12)
    or a yellow wire (rank unspecified) that exists at this particular location
    """
    def __init__(self, player_index, wire_rank_index, indicator_location_index):
        self.player_index = player_index
        self.wire_rank_index = wire_rank_index
        self.indicator_location_index = indicator_location_index

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        _, num_ranks, max_prefilled, max_num_wires = constraint_matrix.shape
        prefilled_offset, num_wires = np.indices((max_prefilled, max_num_wires))
        location = self.indicator_location_index
        covers_location = (prefilled_offset <= location) & (location < prefilled_offset + num_wires)

        # The indicated rank must cover the indicated position; every other rank must not.
        is_indicated_rank = np.arange(num_ranks) == self.wire_rank_index
        per_rank_mask = np.where(
            is_indicated_rank[:, None, None],
            covers_location[None, :, :],
            ~covers_location[None, :, :],
        )
        constraint_matrix[self.player_index] &= per_rank_mask

class CountIndicatorConstraint(Constraint):
    """
    There exists an indicator tile that shows that there are 1/2/3x of a certain tile in the player's hand
    that includes the indicated tile
    """
    def __init__(self, player_index, indicator_location_index, wire_count):
        self.player_index = player_index
        self.indicator_location_index = indicator_location_index
        self.wire_count = wire_count

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        _, _, max_prefilled, max_num_wires = constraint_matrix.shape
        prefilled_offset, num_wires = np.indices((max_prefilled, max_num_wires))
        location = self.indicator_location_index
        covers_location = (prefilled_offset <= location) & (location < prefilled_offset + num_wires)
        # Whichever rank occupies the indicated location must have exactly wire_count copies there.
        wrong_count = num_wires != self.wire_count
        invalid = covers_location & wrong_count
        constraint_matrix[self.player_index] &= ~invalid

class WireAskConstraint(Constraint):
    """
    player at player_index made an ask for a wire of a certain rank.
    This implies that the player has a certain rank
    """
    def __init__(self, player_index, wire_rank_index):
        self.player_index = player_index
        self.wire_rank_index = wire_rank_index

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        # At least one wire of this rank means zero-count is not allowed.
        constraint_matrix[self.player_index, self.wire_rank_index, :, 0] = False

class SubsetConstraint(Constraint):
    """
    Only k/n of these wire ranks exist.
    All the wires that are included in the subset have a max count of 1,
    so each player can have only 0/1 counts
    """
    def __init__(self, wire_rank_indexes, subset_count):
        self.wire_rank_indexes = wire_rank_indexes
        self.subset_count = subset_count

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        # Can only have 0/1 counts for a specific wire rank
        constraint_matrix[:, self.wire_rank_indexes, :, 2:] = False

class RankEqualConstraint(Constraint):
    """
    Constraint that these two wires have the same rank_int
    The constraint is for two neighboring locations
    """
    def __init__(self, player_index, left_indicator_location_index):
        self.player_index = player_index
        self.left_indicator_location_index = left_indicator_location_index

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        _, _, max_prefilled, max_num_wires = constraint_matrix.shape
        prefilled_offset, num_wires = np.indices((max_prefilled, max_num_wires))
        left_location = self.left_indicator_location_index
        right_location = left_location + 1
        covers_left = (prefilled_offset <= left_location) & (left_location < prefilled_offset + num_wires)
        covers_right = (prefilled_offset <= right_location) & (right_location < prefilled_offset + num_wires)
        # Same rank at both positions means no single rank block covers exactly one of them.
        straddles_boundary = covers_left ^ covers_right
        constraint_matrix[self.player_index] &= ~straddles_boundary

class RankNotEqualConstraint(Constraint):
    """
    Constraint that these two wires do not have the same rank_int
    The constraint is for two neighboring locations
    """
    def __init__(self, player_index, left_indicator_location_index):
        self.player_index = player_index
        self.left_indicator_location_index = left_indicator_location_index

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        _, _, max_prefilled, max_num_wires = constraint_matrix.shape
        prefilled_offset, num_wires = np.indices((max_prefilled, max_num_wires))
        left_location = self.left_indicator_location_index
        right_location = left_location + 1
        covers_left = (prefilled_offset <= left_location) & (left_location < prefilled_offset + num_wires)
        covers_right = (prefilled_offset <= right_location) & (right_location < prefilled_offset + num_wires)
        # Different ranks at the two positions means no single rank block covers both.
        covers_both = covers_left & covers_right
        constraint_matrix[self.player_index] &= ~covers_both

class FailedDualCutConstraint(Constraint):
    """
    Constraint that a dual cut failed, as such the askee is guaranteed to not have wires of a certain rank
    at certain locations
    """
    def __init__(self, player_index, wire_rank_indexes, location_indexes):
        self.player_index = player_index
        self.wire_rank_indexes = wire_rank_indexes
        self.location_indexes = location_indexes

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        _, _, max_prefilled, max_num_wires = constraint_matrix.shape
        prefilled_offset, num_wires = np.indices((max_prefilled, max_num_wires))
        # Union the "covers any of the forbidden locations" masks — a block that covers any
        # single forbidden location is invalid for a forbidden rank.
        covers_forbidden_location = np.zeros((max_prefilled, max_num_wires), dtype=bool)
        for location in self.location_indexes:
            covers_forbidden_location |= (
                (prefilled_offset <= location) & (location < prefilled_offset + num_wires)
            )
        for wire_rank_index in self.wire_rank_indexes:
            constraint_matrix[self.player_index, wire_rank_index] &= ~covers_forbidden_location

class WireLimitConstraint(Constraint):
    """
    Constraint that says you can't assign more wires to a hand than what's allowed
    """
    def __init__(self, player_index, wire_limit):
        self.player_index = player_index
        self.wire_limit = wire_limit

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        _, _, max_prefilled, max_num_wires = constraint_matrix.shape
        prefilled_offset, num_wires = np.indices((max_prefilled, max_num_wires))
        # A rank block that extends past the hand size is invalid for every rank in this player's row.
        exceeds_limit = (prefilled_offset + num_wires) > self.wire_limit
        constraint_matrix[self.player_index] &= ~exceeds_limit

class ForbidRankConstraint(Constraint):
    """
    Player holds zero wires of this rank. Used by inclusion-exclusion handling of
    YellowWireAskConstraint — the "no yellow" scenario is expressible per-rank as a
    stack of ForbidRankConstraint instances.
    """
    def __init__(self, player_index, wire_rank_index):
        self.player_index = player_index
        self.wire_rank_index = wire_rank_index

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        # Allow num_wires == 0; forbid every count >= 1.
        constraint_matrix[self.player_index, self.wire_rank_index, :, 1:] = False

class YellowWireAskConstraint(WireAskConstraint):
    """
    Player asked a rank-unspecified yellow dual cut and thus holds at least one wire across
    the given yellow ranks. Semantically a WireAskConstraint over a set of ranks rather
    than a single rank.

    Because this is a JOINT constraint across ranks it cannot be expressed in the
    per-cell constraint matrix; update_constraint_matrix is a no-op and
    compute_probability_matrices applies it via inclusion-exclusion.
    """
    def __init__(self, player_index, yellow_rank_indexes):
        # wire_rank_index has no single value for a joint ask; pass None so callers that
        # look at the parent's attribute will see an explicit sentinel rather than a
        # misleading rank index.
        super().__init__(player_index=player_index, wire_rank_index=None)
        self.yellow_rank_indexes = list(yellow_rank_indexes)

    def update_constraint_matrix(self, constraint_matrix: np.array) -> None:
        # Intentionally empty. See class docstring.
        pass

def get_constraint_matrix(
        wire_limits_per_player,
        wire_ranks,
        constraints: list[Constraint]
):
    """
    Construct a (num_players, num_wires, max_wires_per_player, [0,4] wires count) matrix that represents where you
    can assign a wire to. The dimensions are:
    * the player index [0, num_players)
    * The sorted wire index. Wires are sorted by Wire.rank_int
    * The starting offset of the wire, 0 means the wire starts at the first index
    * How many wires of that rank to assign, counts can be in the range of [0, 4]

    Params:
        wire_limits_per_player: list of length (num_players) that describes how many wires can exist per hand
        wire_ranks: List of all the valid wire ranks
        constraints: List of all constraints that exist
    """
    # To start with, everything is valid
    constraint_matrix = np.ones(
        (len(wire_limits_per_player), len(wire_ranks), max(wire_limits_per_player) + 1, 5),
        dtype=np.bool_
    )

    # We know that there are wire limits per player, add these constraints
    wire_limit_constraints = [
        WireLimitConstraint(player_index, wire_limit)
        for player_index, wire_limit in enumerate(wire_limits_per_player)]

    for constraint in constraints + wire_limit_constraints:
        constraint.update_constraint_matrix(constraint_matrix)

    return constraint_matrix

