# Classes with various game decisions
# No real logic in this class. Mostly storing the hierarchy of decisions with some properties about them
# TODO: figure out how to implement equipment here.
from abc import ABC, abstractmethod

from src.logic.wire import Wire

class Decision(ABC):
    def __init__(self, wire: Wire):
        self.wire = wire

    # For each turn, if you can lead with this decision
    @property
    @abstractmethod
    def is_turn_starting_decision(self):
        pass

    # If this decision is made, does this signal the end of a turn
    @property
    @abstractmethod
    def is_turn_ending_decision(self):
        pass

class RankIndicatorRevealDecision(Decision):
    """First-round reveal: the player publicly shows the rank at one position in their
    hand. The wire is NOT cut — the position stays in `revealed_wires` as False — only
    the rank becomes public knowledge via a RankIndicatorConstraint."""
    def __init__(self, wire: Wire, player_index: int, position: int):
        super().__init__(wire)
        self.player_index = player_index
        self.position = position

    @property
    def is_turn_starting_decision(self):
        return True

    @property
    def is_turn_ending_decision(self):
        return True

class CutDecision(Decision):
    def __init__(self, wire: Wire):
        super().__init__(wire)

    @property
    def is_turn_starting_decision(self):
        return True

# self cutting wires, only can cut if you have all the wires of a certain type
# all 4 blue wires, 2 remaining blue wires, 2 yellow wires, red wires with only red wires remaining in hand
# One SingleCutDecision represents cutting every unrevealed wire in the player's hand that matches the
# decision's wire (by color for rank=0, by color + rank otherwise) — potentially multiple positions at once.
class SingleCutDecision(CutDecision):
    def __init__(self, wire: Wire, player_index: int):
        super().__init__(wire)
        self.player_index = player_index

    @property
    def is_turn_ending_decision(self):
        return True

# Pick a position from another player, see if you can cut that wire
# TODO: keep track of equipment (like dual cut)
class DualCutDecision(CutDecision):
    def __init__(
            self,
            wire: Wire,
            asker_player_index: int,
            askee_player_index: int,
            askee_player_position: int,
            askee_hand_position: int,
    ):
        super().__init__(wire)
        self.asker_player_index = asker_player_index
        self.askee_player_index = askee_player_index
        self.askee_player_position = askee_player_position
        self.askee_hand_position = askee_hand_position

    @property
    def is_turn_ending_decision(self):
        return False

# responding to a dual cut
class ResponseDecision(Decision):
    def __init__(self, wire: Wire, asker_player_index: int, askee_player_index: int):
        super().__init__(wire)
        self.asker_player_index = asker_player_index
        self.askee_player_index = askee_player_index

    @property
    def is_turn_starting_decision(self):
        return False

class AskeeResponseDecision(ResponseDecision):
    """
    If the decision is successful, then the askee indicates that the wire at the indicated position is cut, then
    lets the asker cut their corresponding wire.

    # TODO: this is not compatible with the possibility of using equipment, implement this later

    If the decision is unsuccessful, then the askee indicates what wire was attempted to be cut
    """
    def __init__(
            self,
            wire: Wire,
            asker_player_index: int,
            askee_player_index: int,
            is_successful_dual_cut: bool,
            indicator_wire: Wire,
            indicator_wire_position: int,
    ):
        super().__init__(wire, asker_player_index, askee_player_index)
        self.is_successful_dual_cut = is_successful_dual_cut
        self.indicator_wire = indicator_wire
        self.indicator_wire_position = indicator_wire_position

    @property
    def is_turn_ending_decision(self):
        return not self.is_successful_dual_cut

class AskerResponseDecision(ResponseDecision):
    def __init__(
            self,
            wire: Wire,
            asker_player_index: int,
            askee_player_index: int,
            hand_position: int,
    ):
        super().__init__(wire, asker_player_index, askee_player_index)
        self.hand_position = hand_position

    @property
    def is_turn_ending_decision(self):
        return True

class PassDecision(Decision):
    """
    The acting player has no legal moves (typically because every wire in their hand has
    already been cut) and skips their turn. No constraints emitted, no state updated;
    the turn just ends.
    """
    def __init__(self):
        # No wire is associated with passing; pass None so the parent's `wire` attribute
        # exists as an explicit sentinel rather than being absent.
        super().__init__(wire=None)

    @property
    def is_turn_starting_decision(self):
        return True

    @property
    def is_turn_ending_decision(self):
        return True