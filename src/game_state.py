# keep track of the game state
from collections import Counter
from typing import Optional

import numpy as np

from src.constraint import RankIndicatorConstraint, SubsetConstraint, WireAskConstraint
from src.decision import Decision, SingleCutDecision, DualCutDecision, AskeeResponseDecision, AskerResponseDecision
from src.wire import Wire, BLUE, YELLOW, RED


class GameState:
    '''
    Keep track of a single game state. Glorified dataclass
    Meant to be manipulated by the Game Manager and provide available information to Player
    instances

    TODO list:
        String constructor (for ease of use when playing)
        Figure out how to represent items

    '''

    def __init__(
            self,
            num_players: int=5,
            yellow_wires: Optional[set[int]] = None,
            yellow_wires_in_hands: int = 0,
            red_wires: Optional[set[int]] = None,
            red_wires_in_hands: int = 0,
            # how many wrong moves you can make before things blow up
            total_health: int=5,
            # who's next to move. Default is the first player (0)
            player_to_move: int = 0,
    ):
        self.num_players = num_players
        self.yellow_wires = set()
        if yellow_wires is None:
            self.yellow_wires = set()
        else:
            self.yellow_wires = set([Wire(rank=rank, color=YELLOW) for rank in yellow_wires])
        self.yellow_wires_in_hands = yellow_wires_in_hands

        if red_wires is None:
            self.red_wires = set()
        else:
            self.red_wires = set([Wire(rank=rank, color=RED) for rank in red_wires])
        self.red_wires_in_hands = red_wires_in_hands
        self.total_health = total_health
        self.player_to_move = player_to_move

        # Keep some variables to keep track of a single instance of the game().
        # Helpful for debugging
        self.turns: list[list[Decision]] = []
        self._has_lost = False
        self._has_won = False

        # basic wire info
        self.player_wires = []
        self.revealed_wires = []
        self.wire_ranks = []  # list of all the wire ranks available sorted
        self.wire_counts = []  # list of all the counts per wire in the deck
        self.wire_revealed_counts = []  # list of all the counts per wire that are revealed among the players
        self.wire_limits_per_player = []
        # used for probability_utils calls
        self.wire_to_index_mapping = {}
        self.public_constraints = []

        self._init_board()

    def _init_board(self) -> None:
        """
        Initialize the state of the board. Helpful for debugging
        """
        # get all the wires and shuffle the deck
        blue_wires = [Wire(rank=i+1, color=BLUE) for _ in range(4) for i in range(12)]
        if self.yellow_wires:
            yellow_wires = np.random.choice(self.yellow_wires, self.yellow_wires_in_hands, replace=False)
        else:
            yellow_wires = []
        if self.red_wires:
            red_wires = np.random.choice(self.red_wires, self.red_wires_in_hands, replace=False)
        else:
            red_wires = []
        wires = blue_wires + yellow_wires + red_wires
        np.random.shuffle(wires)

        # get the counts of wires
        wire_counts = Counter(wires)
        for wire, count in sorted(wire_counts.items()):
            self.wire_ranks.append(wire)
            self.wire_counts.append(count)
            self.wire_revealed_counts.append(0)

        # deal them out to each player
        self.wire_limits_per_player = np.array([(len(wires) + 4 - i) // 5 for i in range(5)])
        cum_sum_wire_limits = np.cumsum(self.wire_limits_per_player)
        for player_index, wire_limit in enumerate(self.wire_limits_per_player):
            start_index = 0 if player_index == 0 else cum_sum_wire_limits[player_index - 1]
            cur_player_wires = wires[start_index: start_index + self.wire_limits_per_player[player_index]]
            cur_player_wires = sorted(cur_player_wires)
            self.player_wires.append(cur_player_wires)
            self.revealed_wires.append([False] * len(cur_player_wires))

        # update the maps
        wire_raw_ints = sorted(set([wire.raw_int for wire in wires]))
        self.wire_to_index_mapping = {Wire(raw_int=raw_int): i for i, raw_int in enumerate(wire_raw_ints)}

        # add subset constraints
        zip_data = [(self.yellow_wires, self.yellow_wires_in_hands), (self.red_wires, self.red_wires_in_hands)]
        for subset_wires, wire_count in zip_data:
            if wire_count > 0:
                subset_constraint = SubsetConstraint(
                    wire_rank_indexes=[self.wire_to_index_mapping[wire] for wire in subset_wires],
                    subset_count=wire_count,
                )
                self.public_constraints.append(subset_constraint)

    @property
    def is_start_of_turn(self) -> bool:
        if self.most_recent_turn is None:
            return True
        return self.most_recent_decision.is_turn_ending_decision

    @property
    def most_recent_decision(self) -> Optional[Decision]:
        if self.most_recent_turn is None:
            return None
        return self.most_recent_turn[-1]

    @property
    def most_recent_turn(self) -> Optional[list[Decision]]:
        if len(self.turns) == 0:
            return None
        return self.turns[-1]

    @property
    def has_won(self) -> bool:
        return self._has_won

    @property
    def has_lost(self) -> bool:
        return self._has_lost

    def increment_player_to_move(self):
        self.player_to_move = (self.player_to_move + 1) % self.num_players

    def update_constraints_from_decision(self, decision: Decision) -> None:
        """
        Update the set of constraints known from the fact that a decision occurred
        Args:
            decision: the decision that was most recently made.
        """
        constraints = []
        if isinstance(decision, SingleCutDecision):
            # A single cut reveals every unrevealed wire in the player's hand that matches
            # decision.wire — by color+rank for blue, by color alone for rank=0 (yellow/red).
            # Each matching position becomes a RankIndicatorConstraint for its actual rank.
            player_index = decision.player_index
            cut_wire = decision.wire
            for position, (wire, is_revealed) in enumerate(zip(
                    self.player_wires[player_index], self.revealed_wires[player_index])):
                if is_revealed:
                    continue
                if wire.color != cut_wire.color:
                    continue
                if cut_wire.rank != 0 and wire.rank != cut_wire.rank:
                    continue
                constraints.append(RankIndicatorConstraint(
                    player_index=player_index,
                    wire_rank_index=self.wire_to_index_mapping[wire],
                    indicator_location_index=position,
                ))
        elif isinstance(decision, DualCutDecision):
            # It is now public that the asker player has at least one wire of that rank.
            # TODO: refactor probability_utils.py to take into account WireAskConstraints for yellow
            # (rank-unspecified) asks. The current constraint_matrix only handles independent
            # wire assignments of a single rank, so "has at least one wire across these yellow
            # ranks" is not expressible; skip emitting for now.
            if decision.wire.rank != 0:
                constraints.append(WireAskConstraint(
                    player_index=decision.asker_player_index,
                    wire_rank_index=self.wire_to_index_mapping[decision.wire],
                ))
        elif isinstance(decision, AskeeResponseDecision):
            # The askee reveals the actual wire at the asked position. Success and failure both
            # reveal the same information in the no-equipment setting.
            constraints.append(RankIndicatorConstraint(
                player_index=decision.askee_player_index,
                wire_rank_index=self.wire_to_index_mapping[decision.indicator_wire],
                indicator_location_index=decision.indicator_wire_position,
            ))
        elif isinstance(decision, AskerResponseDecision):
            # The asker reveals the wire they cut from their own hand.
            constraints.append(RankIndicatorConstraint(
                player_index=decision.asker_player_index,
                wire_rank_index=self.wire_to_index_mapping[decision.wire],
                indicator_location_index=decision.hand_position,
            ))
        else:
            raise NotImplementedError(f"can not handle decision of type {type(decision)}")

        self.public_constraints.extend(constraints)