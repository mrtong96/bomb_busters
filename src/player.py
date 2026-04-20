# class to handle player logic
from src.decision import Decision, CutDecision, DualCutDecision, AskeeResponseDecision, SingleCutDecision, \
    AskerResponseDecision
from src.game_state import GameState
from src.wire import Wire, BLUE, YELLOW, RED


class Player:
    def __init__(self, player_index: int):
        self.player_index = player_index

    def make_decision(self, game_state: GameState):
        pass

    def _get_all_single_cut_decisions(self, game_state: GameState) -> list[SingleCutDecision]:
        # Can single cut for blue wires if all the remaining unrevealed blue wires of a rank are in the player's hand
        # Can single cut for the yellow wires if all the remaining unrevealed yellow wires are in the player's hand
        # Can single cut for the red wires if and only if the only unrevealed wires in the hand are red.
        decisions: list[SingleCutDecision] = []
        own_hand = game_state.player_wires[self.player_index]
        own_revealed_flags = game_state.revealed_wires[self.player_index]
        own_unrevealed = [w for w, r in zip(own_hand, own_revealed_flags) if not r]

        # tally how many of each rank this player holds unrevealed, keyed by wire_ranks index
        own_counts_by_rank_index: dict[int, int] = {}
        for wire in own_unrevealed:
            idx = game_state.wire_to_index_mapping[wire]
            own_counts_by_rank_index[idx] = own_counts_by_rank_index.get(idx, 0) + 1

        # Blue: own unrevealed count for rank X equals the global unrevealed count for rank X
        for idx, own_count in own_counts_by_rank_index.items():
            rank_wire = game_state.wire_ranks[idx]
            if rank_wire.color != BLUE:
                continue
            global_unrevealed = game_state.wire_counts[idx] - game_state.wire_revealed_counts[idx]
            if own_count == global_unrevealed:
                decisions.extend(SingleCutDecision(w) for w in own_unrevealed if w == rank_wire)

        # Yellow: aggregate across every yellow rank — all unrevealed yellow globally must be in own hand
        yellow_global_unrevealed = sum(
            game_state.wire_counts[i] - game_state.wire_revealed_counts[i]
            for i, w in enumerate(game_state.wire_ranks)
            if w.color == YELLOW
        )
        own_yellow = [w for w in own_unrevealed if w.color == YELLOW]
        if own_yellow and len(own_yellow) == yellow_global_unrevealed:
            decisions.extend(SingleCutDecision(w) for w in own_yellow)

        # Red: only fires when every unrevealed wire in own hand is red
        if own_unrevealed and all(w.color == RED for w in own_unrevealed):
            decisions.extend(SingleCutDecision(w) for w in own_unrevealed)

        return decisions

    def _get_all_dual_cut_decisions(self, game_state: GameState) -> list[DualCutDecision]:
        # Dual cut, can make a dual cut for any unrevealed blue wires from the current player
        # For yellow cuts, you only ask for a yellow wire of unspecified rank
        # You can not make a dual cut for red cuts
        decisions: list[DualCutDecision] = []
        own_hand = game_state.player_wires[self.player_index]
        own_revealed_flags = game_state.revealed_wires[self.player_index]
        own_blue_ranks = {
            w.rank for w, r in zip(own_hand, own_revealed_flags) if not r and w.color == BLUE
        }
        has_own_yellow = any(
            not r and w.color == YELLOW for w, r in zip(own_hand, own_revealed_flags)
        )

        for askee in range(game_state.num_players):
            if askee == self.player_index:
                continue
            for hand_pos, revealed in enumerate(game_state.revealed_wires[askee]):
                if revealed:
                    continue
                for rank in own_blue_ranks:
                    decisions.append(DualCutDecision(
                        wire=Wire(rank=rank, color=BLUE),
                        asker_player_index=self.player_index,
                        askee_player_index=askee,
                        askee_player_position=askee,
                        askee_hand_position=hand_pos,
                    ))
                if has_own_yellow:
                    # rank=0 signals "unspecified" per Wire's docstring
                    decisions.append(DualCutDecision(
                        wire=Wire(rank=0, color=YELLOW),
                        asker_player_index=self.player_index,
                        askee_player_index=askee,
                        askee_player_position=askee,
                        askee_hand_position=hand_pos,
                    ))

        return decisions

    def _get_askee_response_decision(self, game_state: GameState) -> AskeeResponseDecision:
        if not isinstance(game_state.most_recent_decision, DualCutDecision):
            raise ValueError("Most recent decision is not a DualCutDecision")

        # If the dual cut decision is incorrect, we need to return one decision where is_successful_dual_cut is False
        # If the dual cut is correct, then we can return a single decision where is_successful_dual_cut is True
        dual_cut_decision = game_state.most_recent_decision
        if not isinstance(dual_cut_decision, DualCutDecision):
            raise ValueError(f"decision should be a DualCutDecision, got {type(dual_cut_decision)}")
        actual = game_state.player_wires[dual_cut_decision.askee_player_index][dual_cut_decision.askee_hand_position]
        claim = dual_cut_decision.wire
        is_success = actual.color == claim.color and (claim.rank == 0 or actual.rank == claim.rank)
        return AskeeResponseDecision(
            wire=actual,
            is_successful_dual_cut=is_success,
            indicator_wire=actual,
            indicator_wire_position=dual_cut_decision.askee_hand_position,
        )

    def _get_asker_response_decision(self, game_state: GameState) -> AskerResponseDecision:
        most_recent_decision = game_state.most_recent_decision
        if (not isinstance(most_recent_decision, AskeeResponseDecision)
            or not most_recent_decision.is_successful_dual_cut):
            raise ValueError("Most recent decision can only be a failed dual cut from the Askee")
        # We can pick any of the successful wire ranks that we initially asked to reveal.
        # Turn sequence is always DualCutDecision, AskeeResponseDecision, AskerResponseDecision
        dual_cut_decision = game_state.most_recent_turn[-2]
        if not isinstance(dual_cut_decision, DualCutDecision):
            raise ValueError(f"decision should be a DualCutDecision, got {type(dual_cut_decision)}")
        claim = dual_cut_decision.wire
        asker_hand = game_state.player_wires[dual_cut_decision.asker_player_index]
        asker_revealed_flags = game_state.revealed_wires[dual_cut_decision.asker_player_index]
        for pos, (wire, revealed) in enumerate(zip(asker_hand, asker_revealed_flags)):
            if revealed or wire.color != claim.color:
                continue
            if claim.rank != 0 and wire.rank != claim.rank:
                continue
            return AskerResponseDecision(wire=wire, hand_position=pos)
        raise ValueError("No matching unrevealed wire in asker's hand to reveal")

    def get_all_legal_decisions(self, game_state: GameState) -> list[Decision]:
        legal_decisions = []

        if game_state.is_start_of_turn:
            # check if there are any single cut decisions
            legal_decisions.extend(self._get_all_single_cut_decisions(game_state))

            # enumerate through all possible dual cut decisions
            legal_decisions.extend(self._get_all_dual_cut_decisions(game_state))
        else:
            most_recent_decision = game_state.most_recent_decision
            # If the last decision is a dual cut decision, enumerate through all AskeeResponseDecision possibilities
            if isinstance(most_recent_decision, DualCutDecision):
                legal_decisions.append(self._get_askee_response_decision(game_state))
            # enumerate through all AskerResponseDecision possibilities
            elif (isinstance(most_recent_decision, AskeeResponseDecision)
                and most_recent_decision.is_successful_dual_cut
            ):
                legal_decisions.append(self._get_asker_response_decision(game_state))
            else:
                raise ValueError(f"unrecognized last decision: {game_state.most_recent_decision}")

        return legal_decisions