# class to handle player logic
import copy

import numpy as np

from src.constraint import (
    RankIndicatorConstraint,
    WireAskConstraint,
    YellowWireAskConstraint,
)
from src.decision import Decision, DualCutDecision, AskeeResponseDecision, SingleCutDecision, \
    AskerResponseDecision, PassDecision, RankIndicatorRevealDecision
from src.game_state import GameState
from src.probability_utils import compute_probability_matrices, compute_shannon_entropy
from src.wire import Wire, BLUE, YELLOW, RED


class Player:
    def __init__(self, player_index: int, decision_making_process='greedy'):
        self.player_index = player_index
        self.decision_making_process = decision_making_process

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

        if self.decision_making_process == 'greedy':
            return self.make_greedy_decision(game_state, legal_decisions)
        elif self.decision_making_process == 'entropy':
            return self.make_best_entropy_decision(game_state, legal_decisions)
        else:
            raise NotImplementedError(f"unrecognized decision making process: {self.decision_making_process}")

    def make_greedy_decision(self, game_state: GameState, legal_decisions: list[Decision]) -> Decision:
        """
        Out of all decisions, choose the one that maximizes the probability of success.
        Not altruistic but way easier to implement.

        Preference rules:
          - If a response is the only legal option (turn-mid response phase), return it.
          - (TODO implement) else if there are RankIndicatorRevealDecision to make, then pick the one
            that minimizes global entropy
          - Else if any single cut exists, pick the one that minimizes post-state global
            (public-observer) entropy. Single cuts always succeed, so "best" is chosen by
            information content rather than success probability.
          - Else (only dual cuts remain), pick the dual cut whose claim is most likely to
            succeed from the asker's own view: conditioning the density matrix on the
            asker's full hand (every position pinned to its actual rank).
        """
        if len(legal_decisions) == 1:
            return legal_decisions[0]

        # Asker-response phase with multiple matching wires: pick the one minimizing
        # post-state global entropy. (Reaches here only when this branch contains all
        # legal decisions; turn-start phases never mix asker responses with other types.)
        asker_responses = [d for d in legal_decisions if isinstance(d, AskerResponseDecision)]
        if asker_responses:
            return min(
                asker_responses,
                key=lambda d: self._expected_entropy_after(game_state, d),
            )

        # First-round reveal phase: pick the indicator reveal that minimizes post-state
        # global entropy.
        indicator_reveals = [d for d in legal_decisions if isinstance(d, RankIndicatorRevealDecision)]
        if indicator_reveals:
            return min(
                indicator_reveals,
                key=lambda d: self._expected_entropy_after(game_state, d),
            )

        single_cuts = [d for d in legal_decisions if isinstance(d, SingleCutDecision)]
        if single_cuts:
            return min(
                single_cuts,
                key=lambda d: self._expected_entropy_after(game_state, d),
            )

        # Only dual cuts left — pick by asker-view success probability.
        asker_pins = self._own_hand_pins(game_state)
        density_matrix, *_ = compute_probability_matrices(
            *self._kernel_args(game_state, asker_pins)
        )
        best_decision = None
        best_probability = -1.0
        for decision in legal_decisions:
            if not isinstance(decision, DualCutDecision):
                raise RuntimeError(
                    f"expected only DualCutDecision in the fallback branch, got {type(decision)}"
                )
            probability = self._dual_cut_success_probability(
                density_matrix, game_state, decision,
            )
            if probability > best_probability:
                best_probability = probability
                best_decision = decision
        return best_decision

    def _own_hand_pins(self, game_state: GameState) -> list:
        """RankIndicatorConstraint for every position in this player's own hand — the
        asker's private knowledge that they know every rank in their own hand exactly."""
        pins = []
        for position, wire in enumerate(game_state.player_wires[self.player_index]):
            pins.append(RankIndicatorConstraint(
                player_index=self.player_index,
                wire_rank_index=game_state.wire_to_index_mapping[wire],
                indicator_location_index=position,
            ))
        return pins

    def _dual_cut_success_probability(
            self,
            density_matrix: np.ndarray,
            game_state: GameState,
            decision: DualCutDecision,
    ) -> float:
        """P(askee has claimed rank at askee_pos) under the density_matrix given.
        For rank-unspecified yellow claims, sum over every yellow rank at that cell."""
        askee = decision.askee_player_index
        position = decision.askee_hand_position
        if decision.wire.rank != 0:
            return float(density_matrix[
                askee, position, game_state.wire_to_index_mapping[decision.wire]
            ])
        return float(sum(
            density_matrix[askee, position, i]
            for i, rank_wire in enumerate(game_state.wire_ranks)
            if rank_wire.color == YELLOW
        ))

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

    def _entropy_and_weight_with_extra_constraints(
            self, game_state: GameState, extra_constraints: list,
    ) -> tuple[float, float]:
        """Run the kernel under `extra_constraints` and return (entropy, weight) of the
        resulting density matrix without mutating game_state.

        `weight` is the combinatorial count of configurations satisfying every constraint —
        0 when the hypothetical is infeasible. Callers treat weight as the unnormalized
        joint probability of that hypothetical outcome and drop branches with weight == 0."""
        density_matrix, _, _, weight = compute_probability_matrices(
            *self._kernel_args(game_state, extra_constraints)
        )
        return compute_shannon_entropy(density_matrix), float(weight)

    def _expected_entropy_after(self, game_state: GameState, decision: Decision) -> float:
        """
        Expected Shannon entropy of the density matrix after resolving `decision`, weighted
        by the combinatorial count (joint probability) of each hypothetical outcome.

        - SingleCutDecision: deterministic; process on a copy (actual hand matches the
          revealed ranks, so process_decision bookkeeping is consistent).
        - DualCutDecision: enumerate every plausible askee rank r. For each r, the kernel
          returns both the entropy of the post-askee state AND its combinatorial weight
          (joint probability unnormalized). Infeasible outcomes (weight=0) drop out
          automatically. On success branches, Level C minimax picks the asker response
          that minimizes post-state entropy; that minimum becomes the r's contribution.
        """
        if isinstance(decision, (SingleCutDecision, AskerResponseDecision, RankIndicatorRevealDecision)):
            # All three are deterministic information reveals: clone the state, apply, measure.
            cloned = copy.deepcopy(game_state)
            cloned.process_decision(decision)
            return self._entropy_of_state(cloned)
        elif isinstance(decision, DualCutDecision):
            return self._expected_entropy_after_dual_cut(game_state, decision)
        else:
            raise NotImplementedError(f"cannot evaluate entropy for decision of type {type(decision)}")

    def _dual_cut_base_extras(
            self, game_state: GameState, decision: DualCutDecision,
    ) -> list:
        """Extras from the dual cut ask itself (WireAsk or YellowWireAsk).

        Specific-rank asks emit a WireAskConstraint. Rank-unspecified asks must be yellow
        (that's the only colour for which the game allows an unspecified-rank dual cut);
        legal-decision enumeration is responsible for only producing a rank-0 claim when
        at least one yellow rank exists in the deck, so no feasibility check is needed here.
        """
        if decision.wire.rank != 0:
            return [WireAskConstraint(
                player_index=decision.asker_player_index,
                wire_rank_index=game_state.wire_to_index_mapping[decision.wire],
            )]
        assert decision.wire.color == YELLOW, (
            f"rank-0 dual cut claim must be yellow; got color {decision.wire.color}"
        )
        yellow_rank_indexes = [
            i for i, rank_wire in enumerate(game_state.wire_ranks)
            if rank_wire.color == YELLOW
        ]
        return [YellowWireAskConstraint(
            player_index=decision.asker_player_index,
            yellow_rank_indexes=yellow_rank_indexes,
        )]

    def _expected_entropy_after_dual_cut(
            self, game_state: GameState, decision: DualCutDecision,
    ) -> float:
        askee = decision.askee_player_index
        askee_pos = decision.askee_hand_position
        claim_color = decision.wire.color
        claim_rank = decision.wire.rank  # 0 if unspecified

        base_extras = self._dual_cut_base_extras(game_state, decision)

        total_weight = 0.0
        weighted_entropy_sum = 0.0
        for rank_index in range(len(game_state.wire_ranks)):
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

            # Joint weight of "askee has rank r at askee_pos given base constraints".
            # weight == 0 means the hypothetical is infeasible — drop it.
            entropy_r, weight_r = self._entropy_and_weight_with_extra_constraints(
                game_state, branch_extras,
            )
            if weight_r == 0.0:
                continue

            if is_successful:
                entropy_r = self._minimum_entropy_over_asker_responses(
                    game_state, decision, branch_extras, fallback_entropy=entropy_r,
                )

            weighted_entropy_sum += weight_r * entropy_r
            total_weight += weight_r

        if total_weight == 0.0:
            # No feasible outcome at all — treat as zero information gain.
            return 0.0
        return weighted_entropy_sum / total_weight

    def _minimum_entropy_over_asker_responses(
            self,
            game_state: GameState,
            dual_cut_decision: DualCutDecision,
            branch_extras_so_far: list,
            fallback_entropy: float,
    ) -> float:
        """Minimax: enumerate every legal asker response and return the smallest entropy.
        If no asker response is feasible, return `fallback_entropy` (post-askee entropy)."""
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
            entropy, weight = self._entropy_and_weight_with_extra_constraints(
                game_state, extras,
            )
            if weight == 0.0:
                continue
            if entropy < best_entropy:
                best_entropy = entropy

        if best_entropy == float("inf"):
            return fallback_entropy
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

    def _get_all_asker_response_decisions(self, game_state: GameState) -> list[AskerResponseDecision]:
        """Enumerate every legal AskerResponseDecision after a successful dual cut. The
        asker may have more than one matching unrevealed wire in their hand (especially
        for yellow rank-unspecified claims), and each choice is a separate decision."""
        most_recent_decision = game_state.most_recent_decision
        if (not isinstance(most_recent_decision, AskeeResponseDecision)
            or not most_recent_decision.is_successful_dual_cut):
            raise ValueError("Most recent decision can only be a successful AskeeResponseDecision")
        # Turn sequence is always DualCutDecision, AskeeResponseDecision, AskerResponseDecision
        dual_cut_decision = game_state.most_recent_turn[-2]
        if not isinstance(dual_cut_decision, DualCutDecision):
            raise ValueError(f"decision should be a DualCutDecision, got {type(dual_cut_decision)}")
        claim = dual_cut_decision.wire
        asker_hand = game_state.player_wires[dual_cut_decision.asker_player_index]
        asker_revealed_flags = game_state.revealed_wires[dual_cut_decision.asker_player_index]
        decisions: list[AskerResponseDecision] = []
        for pos, (wire, revealed) in enumerate(zip(asker_hand, asker_revealed_flags)):
            if revealed or wire.color != claim.color:
                continue
            if claim.rank != 0 and wire.rank != claim.rank:
                continue
            decisions.append(AskerResponseDecision(
                wire=wire,
                asker_player_index=dual_cut_decision.asker_player_index,
                askee_player_index=dual_cut_decision.askee_player_index,
                hand_position=pos,
            ))
        if not decisions:
            raise ValueError("No matching unrevealed wire in asker's hand to reveal")
        return decisions

    def _get_all_first_round_indicator_decisions(self, game_state: GameState) -> list[RankIndicatorRevealDecision]:
        """Enumerate every legal first-round reveal: one RankIndicatorRevealDecision per
        unrevealed BLUE wire in the acting player's hand. Yellow / red wires can't be
        revealed in the first round. The first-round rule also caps reveals at most 2
        players per rank, so any rank already indicated twice is filtered out."""
        own_hand = game_state.player_wires[self.player_index]
        own_revealed_flags = game_state.revealed_wires[self.player_index]

        # Tally how many times each rank has already been indicated. During the first
        # round only RankIndicatorRevealDecisions are recorded, so iterating the turn
        # history captures every indication so far.
        rank_indication_counts: dict[int, int] = {}
        for turn in game_state.turns:
            for past_decision in turn:
                if isinstance(past_decision, RankIndicatorRevealDecision):
                    rank_index = game_state.wire_to_index_mapping[past_decision.wire]
                    rank_indication_counts[rank_index] = rank_indication_counts.get(rank_index, 0) + 1

        decisions: list[RankIndicatorRevealDecision] = []
        for position, wire in enumerate(own_hand):
            if own_revealed_flags[position]:
                continue
            if wire.color != BLUE:
                continue
            rank_index = game_state.wire_to_index_mapping[wire]
            if rank_indication_counts.get(rank_index, 0) >= 2:
                continue
            decisions.append(RankIndicatorRevealDecision(
                wire=wire,
                player_index=self.player_index,
                position=position,
            ))
        return decisions

    def get_all_legal_decisions(self, game_state: GameState) -> list[Decision]:
        legal_decisions = []

        if game_state.is_first_round:
            indicator_decisions = self._get_all_first_round_indicator_decisions(game_state)
            if indicator_decisions:
                return indicator_decisions
            # No legal indicator (no blue wires, or every blue rank already indicated by 2
            # other players). Skip this player's first-round turn.
            return [PassDecision()]
        elif game_state.is_start_of_turn:
            # If the acting player has no wires left (every position in their own hand is
            # already revealed), they have no legal cut to make and skip their turn.
            if all(game_state.revealed_wires[self.player_index]):
                return [PassDecision()]

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
                legal_decisions.extend(self._get_all_asker_response_decisions(game_state))
            else:
                raise ValueError(f"unrecognized last decision: {game_state.most_recent_decision}")

        return legal_decisions