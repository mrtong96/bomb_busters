"""Performance + win-rate benchmark: play complete greedy 5-player games on a 48-blue-wire
deck (12 ranks × 4 copies, no yellow/red) to see how long they take and how often the team
wins.

Run directly (not auto-collected by pytest since the filename doesn't start with `test_`):

    PYTHONPATH=. python tests/logic/perf_game_playthrough.py

The GameManager constructs Players with the default decision_making_process='greedy', so
each turn pays at most one compute_probability_matrices call (for dual-cut candidate
scoring) plus one call per single-cut tie-break candidate.
"""
import time

import numpy as np

from src.logic.game_manager import GameManager


MAX_TURNS_PER_GAME = 500  # safety cap; real games should finish well under this.


def play_one_game(seed: int) -> tuple[bool, float, int, int]:
    """Play one greedy game to completion. Returns (won, elapsed_seconds, num_turns, health_remaining)."""
    np.random.seed(seed)
    gm = GameManager(num_players=5)

    t0 = time.perf_counter()
    turns = 0
    while not gm.game_state.has_won and not gm.game_state.has_lost:
        gm.process_turn()
        turns += 1
        if turns > MAX_TURNS_PER_GAME:
            raise RuntimeError(f"game did not terminate in {MAX_TURNS_PER_GAME} turns")
    elapsed = time.perf_counter() - t0
    return gm.game_state.has_won, elapsed, turns, gm.game_state.total_health


def main(num_games: int = 25) -> None:
    print(f"Playing {num_games} greedy games (5 players, 48 blue wires = 12 ranks × 4)...")

    # Warm up numba JIT on the first game; its timing gets reported separately so it
    # doesn't skew the measured statistics.
    print("\nWarming up numba JIT (1 game)...")
    won, elapsed, turns, health = play_one_game(seed=0)
    print(f"  warm-up elapsed: {elapsed:.2f}s  turns={turns}  won={won}  health_remaining={health}")

    print(f"\nMeasured games (seeds 1..{num_games}):")
    results = []
    for i in range(1, num_games + 1):
        won, elapsed, turns, health = play_one_game(seed=i)
        results.append((won, elapsed, turns, health))
        status = "WIN " if won else "loss"
        print(f"  seed={i:2d}  {status}  elapsed={elapsed:6.2f}s  "
              f"turns={turns:3d}  health_remaining={health}")

    wins      = sum(1 for w, _, _, _ in results if w)
    elapseds  = np.array([e for _, e, _, _ in results], dtype=np.float64)
    turnlist  = np.array([t for _, _, t, _ in results], dtype=np.int64)
    health    = np.array([h for _, _, _, h in results], dtype=np.int64)

    print(f"\n=== Summary over {num_games} games ===")
    print(f"  win rate:          {wins}/{num_games}  ({wins / num_games:.1%})")
    print(f"  elapsed (s):       mean={elapseds.mean():6.2f}  std={elapseds.std():6.2f}  "
          f"min={elapseds.min():6.2f}  max={elapseds.max():6.2f}")
    print(f"  turns/game:        mean={turnlist.mean():6.1f}  std={turnlist.std():5.1f}  "
          f"min={turnlist.min():4d}  max={turnlist.max():4d}")
    print(f"  health remaining:  mean={health.mean():5.2f}  min={health.min()}  max={health.max()}")
    print(f"  per-turn cost:     mean={elapseds.sum() / turnlist.sum() * 1000:.1f} ms")


if __name__ == "__main__":
    main()
