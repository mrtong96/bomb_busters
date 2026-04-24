# class to handle player logic
import copy

import numpy as np

from src.constraint import (
    RankIndicatorConstraint,
    WireAskConstraint,
    YellowWireAskConstraint,
)
from src.decision import Decision, CutDecision, DualCutDecision, AskeeResponseDecision, SingleCutDecision, \
    AskerResponseDecision
from src.game_state import GameState
from src.probability_utils import compute_probability_matrices, compute_shannon_entropy
from src.wire import Wire, BLUE, YELLOW, RED


class Player:
    def __init__(self, player_index: int):
        self.player_index = player_index

    def make_decision(self, game_state: GameState) -> Decision:
        """
        Function to figure out what each player should do at any given time.
        For now just go with a simple heuristic of trying to reduce the global entropy of the possible wire hands
        given all public information

        At some later point implement some logic to make tradeoffs that move towards maximizing win probability
        Possible improvements could include:
        * Making sure there are likely safe moves after the turn concludes
        * Changing tolerance for failed guesses depending on how many incorrect guesses are remaining
        * Being able to consider hypotheticals and building a game tree

        Args:
            game_state: The state of the game
        """
        legal_decisions = self.get_all_legal_decisions(game_state)
        return self.make_best_entropy_decision(game_state, legal_decisions)

    def make_best_entropy_decision(self, game_state: GameState, legal_decisions: list[Decision]) -> Decision:
        """
        Out of all decisions, choose the one that minimizes the public observer's expected
        Shannon entropy of the density matrix after the decision resolves.

        Args:
            game_state: the current game state before the decision
            legal_decisions: All legal decisions to consider
        """
        best_decision = None
        best_entropy = float("inf")
        for decision in legal_decisions:
            expected = self._expected_entropy_after(game_state, decision)
            if expected < best_entropy:
                best_entropy = expected
                best_decision = decision
        return best_decision

    def _kernel_args(self, game_state: GameState, extra_constraints: list = ()) -> tuple:
        """Derive (wire_limits_per_player, wire_limits, constraints) for compute_probability_matrices."""
        wire_limits_per_player = np.array(
            [len(hand) for hand in game_state.player_wires], dtype=np.int32,
        )
        wire_limits = {
            i: (game_state.wire_counts[i], game_state.wire_counts[i])
            for i in range(len(game_state.wire_ranks))
        }
        constraints = list(game_state.public_constraints) + list(extra_constraints)
        return wire_limits_per_player, wire_limits, constraints

    def _density_matrix(self, game_state: GameState) -> np.ndarray:
        """Compute the density matrix for the current state. Shape (num_players, max_slots, num_ranks)."""
        density_matrix, *_ = compute_probability_matrices(*self._kernel_args(game_state))
        return density_matrix

    def _entropy_of_state(self, game_state: GameState) -> float:
        return compute_shannon_entropy(self._density_matrix(game_state))

    def _entropy_with_extra_constraints(
            self, game_state: GameState, extra_constraints: list,
    ) -> float:
        """Compute the public-observer entropy as if `extra_constraints` were added to the
        state, without mutating game_state. Used for evaluating hypothetical outcomes where
        process_decision's reveal-bookkeeping would otherwise disagree with the hypothetical
        wire (process_decision reads the actual hand, but hypotheticals pin a different rank
        at a position than the actual wire there)."""
        density_matrix, *_ = compute_probability_matrices(
            *self._kernel_args(game_state, extra_constraints)
        )
        return compute_shannon_entropy(density_matrix)

    def _expected_entropy_after(self, game_state: GameState, decision: Decision) -> float:
        """
        Expected Shannon entropy of the density matrix after resolving `decision`.

        - SingleCutDecision: deterministic; process on a copy (actual hand matches the
          revealed ranks, so process_decision bookkeeping is consistent).
        - DualCutDecision: probabilistic. Let r range over every rank with p_r > 0 at the
          askee position. For each r, evaluate the hypothetical as added constraints
          (askee-position → rank r; on success, also asker-position → matching wire,
          minimax'd). Weight entropies by p_r.
        """
        if isinstance(decision, SingleCutDecision):
            cloned = copy.deepcopy(game_state)
            cloned.process_decision(decision)
            return self._entropy_of_state(cloned)

        if isinstance(decision, DualCutDecision):
            return self._expected_entropy_after_dual_cut(game_state, decision)

        raise NotImplementedError(f"cannot evaluate entropy for decision of type {type(decision)}")

    def _dual_cut_base_extras(
            self, game_state: GameState, decision: DualCutDecision,
    ) -> list:
        """Extras from the dual cut ask itself (WireAsk or YellowWireAsk)."""
        if decision.wire.rank != 0:
            return [WireAskConstraint(
                player_index=decision.asker_player_index,
                wire_rank_index=game_state.wire_to_index_mapping[decision.wire],
            )]
        yellow_rank_indexes = [
            i for i, rank_wire in enumerate(game_state.wire_ranks)
            if rank_wire.color == decision.wire.color
        ]
        if not yellow_rank_indexes:
            return []
        return [YellowWireAskConstraint(
            player_index=decision.asker_player_index,
            yellow_rank_indexes=yellow_rank_indexes,
        )]

    def _expected_entropy_after_dual_cut(
            self, game_state: GameState, decision: DualCutDecision,
    ) -> float:
        density_matrix = self._density_matrix(game_state)
        askee = decision.askee_player_index
        askee_pos = decision.askee_hand_position
        claim_color = decision.wire.color
        claim_rank = decision.wire.rank  # 0 if unspecified

        base_extras = self._dual_cut_base_extras(game_state, decision)

        expected_entropy = 0.0
        total_probability = 0.0
        for rank_index in range(len(game_state.wire_ranks)):
            probability = float(density_matrix[askee, askee_pos, rank_index])
            # Skip negligible-probability branches: pinning a near-impossible rank produces
            # infeasible states (weight=0, density=NaN) downstream.
            if probability < 1e-9:
                continue
            rank_wire = game_state.wire_ranks[rank_index]
            is_successful = (
                rank_wire.color == claim_color
                and (claim_rank == 0 or rank_wire.rank == claim_rank)
            )

            branch_extras = base_extras + [RankIndicatorConstraint(
                player_index=askee,
                wire_rank_index=rank_index,
                indicator_location_index=askee_pos,
            )]

            try:
                if is_successful:
                    branch_entropy = self._minimum_entropy_over_asker_responses(
                        game_state, decision, branch_extras,
                    )
                else:
                    branch_entropy = self._entropy_with_extra_constraints(game_state, branch_extras)
            except RuntimeError:
                # Joint infeasibility under the hypothetical outcome — drop this branch.
                continue

            expected_entropy += probability * branch_entropy
            total_probability += probability

        # density-matrix probabilities at the askee position should sum to 1 across ranks;
        # guard against drift by renormalizing if they don't.
        if total_probability > 0.0:
            expected_entropy /= total_probability
        return expected_entropy

    def _minimum_entropy_over_asker_responses(
            self,
            game_state: GameState,
            dual_cut_decision: DualCutDecision,
            branch_extras_so_far: list,
    ) -> float:
        """Minimax: enumerate every legal asker response and return the smallest entropy."""
        asker = dual_cut_decision.asker_player_index
        claim_color = dual_cut_decision.wire.color
        claim_rank = dual_cut_decision.wire.rank

        best_entropy = float("inf")
        for position, wire in enumerate(game_state.player_wires[asker]):
            if game_state.revealed_wires[asker][position]:
                continue
            if wire.color != claim_color:
                continue
            if claim_rank != 0 and wire.rank != claim_rank:
                continue
            extras = branch_extras_so_far + [RankIndicatorConstraint(
                player_index=asker,
                wire_rank_index=game_state.wire_to_index_mapping[wire],
                indicator_location_index=position,
            )]
            # Skip jointly-infeasible asker responses: the single-rank density matrix at the
            # askee position can show p > 0 while the joint constraint with the asker's
            # chosen position triggers a count overflow (sorted-block structure forces extra
            # copies). Those scenarios can't happen in reality, so they shouldn't contribute
            # to the minimax.
            try:
                entropy = self._entropy_with_extra_constraints(game_state, extras)
            except RuntimeError:
                continue
            if entropy < best_entropy:
                best_entropy = entropy

        if best_entropy == float("inf"):
            # Asker has no matching unrevealed wire — shouldn't occur per _get_all_dual_cut_decisions;
            # fall back to post-askee entropy rather than returning inf.
            return self._entropy_with_extra_constraints(game_state, branch_extras_so_far)
        return best_entropy

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

        # Blue: one decision per rank X where the player holds all remaining unrevealed blue-X.
        # A single decision covers every matching wire in the hand simultaneously.
        for idx, own_count in own_counts_by_rank_index.items():
            rank_wire = game_state.wire_ranks[idx]
            if rank_wire.color != BLUE:
                continue
            global_unrevealed = game_state.wire_counts[idx] - game_state.wire_revealed_counts[idx]
            if own_count == global_unrevealed:
                decisions.append(SingleCutDecision(wire=rank_wire, player_index=self.player_index))

        # Yellow: one decision (rank=0 "unspecified") when all unrevealed yellow globally sit in own hand.
        yellow_global_unrevealed = sum(
            game_state.wire_counts[i] - game_state.wire_revealed_counts[i]
            for i, w in enumerate(game_state.wire_ranks)
            if w.color == YELLOW
        )
        own_yellow = [w for w in own_unrevealed if w.color == YELLOW]
        if own_yellow and len(own_yellow) == yellow_global_unrevealed:
            decisions.append(SingleCutDecision(
                wire=Wire(rank=0, color=YELLOW), player_index=self.player_index,
            ))

        # Red: one decision (rank=0 "unspecified") when every unrevealed wire in own hand is red.
        if own_unrevealed and all(w.color == RED for w in own_unrevealed):
            decisions.append(SingleCutDecision(
                wire=Wire(rank=0, color=RED), player_index=self.player_index,
            ))

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
            asker_player_index=dual_cut_decision.asker_player_index,
            askee_player_index=dual_cut_decision.askee_player_index,
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
            return AskerResponseDecision(
                wire=wire,
                asker_player_index=dual_cut_decision.asker_player_index,
                askee_player_index=dual_cut_decision.askee_player_index,
                hand_position=pos,
            )
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