"""Performance reference: make_best_entropy_decision on a 48-wire, half-revealed board.

Run directly (not auto-collected by pytest since the filename doesn't start with `test_`):

    PYTHONPATH=. python tests/logic/perf_entropy_decision.py

Reveals exactly 2 wires per rank (an even count, reflecting the fact that in real play
every completed cut adds an even number of reveals to a given rank — single cuts remove
every remaining copy and dual cut successes remove two at a time).
"""
import time
from collections import Counter

import numpy as np

from src.logic.constraint import RankIndicatorConstraint
from src.logic.decision import DualCutDecision, SingleCutDecision
from src.logic.game_state import GameState
from src.logic.player import Player
from src.logic.wire import BLUE, Wire


def b(rank):
    return Wire(rank=rank, color=BLUE)


def _collect_rank_positions(hands):
    rank_positions: dict[int, list] = {}
    for player_index, hand in enumerate(hands):
        for position, wire in enumerate(hand):
            rank_positions.setdefault(wire.rank, []).append((player_index, position))
    return rank_positions


def _reveal_first_per_rank(hands, reveals_per_rank: int = 2):
    """Reveal the first `reveals_per_rank` (player, position) occurrences of each rank in
    sorted order. Skews toward lower-index players."""
    revealed = [[False] * len(h) for h in hands]
    for rank, positions in _collect_rank_positions(hands).items():
        for (p, pos) in positions[:reveals_per_rank]:
            revealed[p][pos] = True
    return revealed


def _reveal_balanced(hands, reveals_per_rank: int = 2):
    """Reveal exactly `reveals_per_rank` wires of each rank, preferring players with fewer
    reveals so far so per-player counts stay close to each other.

    Greedy: iterate ranks in ascending order; at each rank, pick the `reveals_per_rank`
    occurrences whose players have the smallest running reveal count (ties broken by
    player index, then position for determinism).
    """
    num_players = len(hands)
    revealed = [[False] * len(h) for h in hands]
    reveals_per_player = [0] * num_players
    for rank in sorted(_collect_rank_positions(hands).keys()):
        positions = _collect_rank_positions(hands)[rank]
        positions.sort(key=lambda pp: (reveals_per_player[pp[0]], pp[0], pp[1]))
        for (p, pos) in positions[:reveals_per_rank]:
            revealed[p][pos] = True
            reveals_per_player[p] += 1
    return revealed


REVEAL_STRATEGIES = {
    "rank_first": _reveal_first_per_rank,
    "balanced":   _reveal_balanced,
}


def build_heavy_revealed_state(seed: int = 42, reveal_strategy: str = "balanced") -> GameState:
    """48 blue wires (12 ranks × 4), dealt to 5 players, with exactly 2 reveals per rank.

    reveal_strategy:
        - "rank_first": first two (player, position) per rank in sorted order.
        - "balanced":   greedy balanced distribution of reveals across players.
    """
    rng = np.random.default_rng(seed)
    wires = [b(r) for r in range(1, 13) for _ in range(4)]
    rng.shuffle(wires)

    hand_sizes = [10, 10, 10, 9, 9]
    hands = []
    offset = 0
    for size in hand_sizes:
        hand = sorted(wires[offset:offset + size], key=lambda w: w.raw_int)
        hands.append(hand)
        offset += size

    if reveal_strategy not in REVEAL_STRATEGIES:
        raise ValueError(f"unknown reveal strategy {reveal_strategy!r}; "
                         f"options: {sorted(REVEAL_STRATEGIES)}")
    revealed = REVEAL_STRATEGIES[reveal_strategy](hands, reveals_per_rank=2)

    gs = GameState(num_players=5)
    gs.player_wires = [list(h) for h in hands]
    gs.revealed_wires = [list(flags) for flags in revealed]
    counts = Counter(w for h in hands for w in h)
    gs.wire_ranks = sorted(counts.keys())
    gs.wire_counts = [counts[w] for w in gs.wire_ranks]
    gs.wire_to_index_mapping = {w: i for i, w in enumerate(gs.wire_ranks)}
    gs.wire_revealed_counts = [0] * len(gs.wire_ranks)
    for hand, flags in zip(gs.player_wires, gs.revealed_wires):
        for wire, flag in zip(hand, flags):
            if flag:
                gs.wire_revealed_counts[gs.wire_to_index_mapping[wire]] += 1
    gs.turns = []
    gs.public_constraints = []

    # Populate RankIndicatorConstraints for every revealed position — mirrors what
    # _update_constraints_from_decision would have accumulated during real play.
    for player_index, (hand, flags) in enumerate(zip(gs.player_wires, gs.revealed_wires)):
        for position, (wire, revealed_flag) in enumerate(zip(hand, flags)):
            if revealed_flag:
                gs.public_constraints.append(RankIndicatorConstraint(
                    player_index=player_index,
                    wire_rank_index=gs.wire_to_index_mapping[wire],
                    indicator_location_index=position,
                ))

    return gs


def describe_state(gs: GameState) -> None:
    print("board layout:")
    for i, hand in enumerate(gs.player_wires):
        revealed = sum(gs.revealed_wires[i])
        print(f"  player {i}: {len(hand)} wires, {revealed} revealed, "
              f"{len(hand) - revealed} unrevealed")
    print(f"per-rank revealed counts: {gs.wire_revealed_counts}")
    print(f"per-rank total counts:    {gs.wire_counts}")
    print(f"public constraints: {len(gs.public_constraints)}")


def measure(gs: GameState, player_index: int = 0, warm: bool = True) -> None:
    player = Player(player_index=player_index)
    legal = player.get_all_legal_decisions(gs)
    num_single = sum(1 for d in legal if isinstance(d, SingleCutDecision))
    num_dual = sum(1 for d in legal if isinstance(d, DualCutDecision))
    print(f"  legal decisions for player {player_index}: {len(legal)} "
          f"(single-cut: {num_single}, dual-cut: {num_dual})")

    if warm:
        t0 = time.perf_counter()
        _ = player.make_decision(gs)
        warm_elapsed = time.perf_counter() - t0
        print(f"  warm-up elapsed: {warm_elapsed:.1f}s")

    t0 = time.perf_counter()
    decision = player.make_decision(gs)
    elapsed = time.perf_counter() - t0
    print(f"  measured elapsed: {elapsed:.1f}s")
    print(f"  chosen decision: {type(decision).__name__}")
    if isinstance(decision, DualCutDecision):
        print(f"    asker={decision.asker_player_index} askee={decision.askee_player_index} "
              f"askee_hand_position={decision.askee_hand_position} wire={decision.wire.raw_int}")
    elif isinstance(decision, SingleCutDecision):
        print(f"    player={decision.player_index} wire={decision.wire.raw_int}")


def main() -> None:
    # First strategy — the "rank_first" layout keeps the numba warm-up cost here and lets
    # the second strategy measure post-warm without re-paying JIT.
    print("=" * 72)
    print("strategy: rank_first (skewed reveals — low-index players revealed first)")
    print("=" * 72)
    gs_rank_first = build_heavy_revealed_state(reveal_strategy="rank_first")
    describe_state(gs_rank_first)
    print()
    measure(gs_rank_first, player_index=0, warm=True)

    print()
    print("=" * 72)
    print("strategy: balanced (roughly equal reveals per player)")
    print("=" * 72)
    gs_balanced = build_heavy_revealed_state(reveal_strategy="balanced")
    describe_state(gs_balanced)
    print()
    measure(gs_balanced, player_index=0, warm=False)


if __name__ == "__main__":
    main()
