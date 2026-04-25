"""Tkinter UI for inspecting a 5-player Bomb Busters game driven by GameManager.

Run from the project root:

    python src/ui/app.py

(The script self-bootstraps the project root onto sys.path so plain `python` works
without needing to set PYTHONPATH manually.)

Controls:
  - New Game        — reset to a fresh deal.
  - ◀ Back          — undo the most recent Step (snapshot-based).
  - Step ▶          — process one turn.
  - Auto-play       — schedule turns at a fixed interval (toggleable).
  - Speed slider    — interval between auto-play steps.
  - View radio      — choose what's visible:
        All (debug)     every wire's rank, regardless of cut/reveal status.
        Public only     only revealed-cut wires plus publicly indicated ranks.
        Cycle           follows the player whose turn is up (auto-shifts each step).
        Player N        the public view PLUS player N's own hand fully visible.
"""
import copy
import math
import sys
from pathlib import Path

# Allow running this file directly (e.g. `python src/ui/app.py`) by putting the project
# root onto sys.path before any `src.logic.*` imports resolve. The project root is the
# directory two levels up from this file (src/ui/app.py → src/ui → src → <root>).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tkinter as tk
from tkinter import ttk

import numpy as np

from src.logic.constraint import RankIndicatorConstraint
from src.logic.decision import (
    AskeeResponseDecision,
    AskerResponseDecision,
    DualCutDecision,
    PassDecision,
    RankIndicatorRevealDecision,
    SingleCutDecision,
)
from src.logic.game_manager import GameManager
from src.logic.probability_utils import compute_probability_matrices, compute_shannon_entropy
from src.logic.wire import BLUE, RED, YELLOW


WIRE_BG = {
    BLUE:   "#3b82f6",
    YELLOW: "#facc15",
    RED:    "#ef4444",
}
WIRE_FG = {
    BLUE:   "#ffffff",
    YELLOW: "#1f2937",
    RED:    "#ffffff",
}
HIDDEN_BG = "#374151"
HIDDEN_FG = "#9ca3af"


class GameUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bomb Busters — Game Viewer")
        self.root.geometry("1200x680")

        self.gm: GameManager | None = None
        self.autoplay_running = False
        self.view_mode = tk.StringVar(value="all")
        self.delay_ms = tk.IntVar(value=500)
        # Snapshot stack for ◀ Back. Each entry is (deep-copied game_state, log-text
        # index pointing at the start of the lines that step() wrote for that turn).
        self.history: list[tuple[object, str]] = []
        # Cache (cache_key → density_matrix) so refresh() doesn't recompute when neither
        # the game state nor the view mode has changed since the last paint.
        self._density_cache_key: tuple | None = None
        self._density_cache: np.ndarray | None = None
        # Separate cache for the global (public-observer) metrics — entropy plus
        # combinatorial count and weight from the same kernel call. Independent of view
        # mode, depends only on game state.
        self._global_metrics_cache_key: tuple | None = None
        self._global_metrics_cache: tuple[float, float, float] | None = None

        self._build_layout()
        self.new_game()

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Button(top, text="New Game", command=self.new_game).pack(side="left")
        self.back_btn = ttk.Button(top, text="◀ Back", command=self.step_back, state="disabled")
        self.back_btn.pack(side="left", padx=4)
        ttk.Button(top, text="Step ▶", command=self.step).pack(side="left", padx=4)
        self.autoplay_btn = ttk.Button(top, text="Auto-play ▶", command=self.toggle_autoplay)
        self.autoplay_btn.pack(side="left", padx=4)

        ttk.Label(top, text="Speed (ms):").pack(side="left", padx=(16, 4))
        ttk.Scale(
            top, from_=50, to=2000, variable=self.delay_ms,
            orient="horizontal", length=140,
        ).pack(side="left")

        ttk.Label(top, text="View:").pack(side="left", padx=(16, 4))
        for value, label in [
            ("all",      "All (debug)"),
            ("public",   "Public only"),
            ("cycle",    "Cycle"),
            ("player_0", "P0"),
            ("player_1", "P1"),
            ("player_2", "P2"),
            ("player_3", "P3"),
            ("player_4", "P4"),
        ]:
            ttk.Radiobutton(
                top, text=label, value=value, variable=self.view_mode,
                command=self.refresh,
            ).pack(side="left")


        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)

        self.players_frame = ttk.LabelFrame(body, text="Players", padding=8)
        self.players_frame.pack(side="left", fill="both", expand=True)

        side = ttk.Frame(body)
        side.pack(side="right", fill="y", padx=(8, 0))

        self.status_label = ttk.Label(side, text="", justify="left", font=("Menlo", 11))
        self.status_label.pack(anchor="w")

        ttk.Label(side, text="Turn log:").pack(anchor="w", pady=(8, 2))
        log_frame = ttk.Frame(side)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_frame, width=44, height=28, state="disabled",
            wrap="none", font=("Menlo", 10),
        )
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        legend = ttk.LabelFrame(side, text="Legend", padding=4)
        legend.pack(fill="x", pady=(8, 0))
        ttk.Label(legend, text="raised border   = unrevealed").pack(anchor="w")
        ttk.Label(legend, text="underlined text = rank indicated (public)").pack(anchor="w")
        ttk.Label(legend, text="✕               = wire cut").pack(anchor="w")
        ttk.Label(legend, text="hidden cells    = best guess + P(rank)").pack(anchor="w")
        ttk.Label(legend, text="green outline   = next-move target").pack(anchor="w")

    # -------------------------------------------------------------- game flow
    def new_game(self) -> None:
        self.stop_autoplay()
        self.gm = GameManager(num_players=5)
        self.history.clear()
        self._clear_log()
        self._log("New game — 5 players, 48 blue wires (12 ranks × 4).")
        self.refresh()

    def step(self) -> None:
        if self.gm is None:
            return
        gs = self.gm.game_state
        if gs.has_won or gs.has_lost:
            return

        # Save undo snapshot before we mutate. The log anchor marks where the lines
        # written by this step begin so step_back() can truncate cleanly.
        snapshot = copy.deepcopy(gs)
        log_anchor = self.log_text.index("end-1c")
        self.history.append((snapshot, log_anchor))

        prev_turn_count = len(gs.turns)
        prev_health = gs.total_health
        actor_index = gs.player_to_move
        self.gm.process_turn()

        # Header for the turn just played, followed by each decision in it.
        new_turns = gs.turns[prev_turn_count:]
        for offset, turn in enumerate(new_turns):
            turn_number = prev_turn_count + offset + 1
            phase = "first-round" if (turn_number <= gs.num_players) else "round " + str(((turn_number - 1) // gs.num_players) + 1)
            self._log(f"— Turn {turn_number} ({phase}) — P{actor_index} —")
            for decision in turn:
                self._log("  " + self._describe_decision(decision))
            # Subsequent turns in the same step (rare — only happens if a player passed
            # and another auto-step is queued externally) move to the next actor.
            actor_index = (actor_index + 1) % gs.num_players
        if gs.total_health < prev_health:
            self._log(f"  → health: {prev_health} → {gs.total_health}")

        self.refresh()

    def step_back(self) -> None:
        """Restore the game state and log to what they looked like just before the most
        recent step()."""
        if self.gm is None or not self.history:
            return
        self.stop_autoplay()
        snapshot, log_anchor = self.history.pop()
        self.gm.game_state = snapshot
        self.log_text.configure(state="normal")
        self.log_text.delete(log_anchor, "end")
        self.log_text.configure(state="disabled")
        # Game state changed under us — invalidate the cached density matrix.
        self._density_cache_key = None
        self._density_cache = None
        self.refresh()

    def toggle_autoplay(self) -> None:
        if self.autoplay_running:
            self.stop_autoplay()
        else:
            self.autoplay_running = True
            self.autoplay_btn.configure(text="Stop ■")
            self._autoplay_tick()

    def stop_autoplay(self) -> None:
        self.autoplay_running = False
        self.autoplay_btn.configure(text="Auto-play ▶")

    def _autoplay_tick(self) -> None:
        if not self.autoplay_running or self.gm is None:
            return
        gs = self.gm.game_state
        if gs.has_won or gs.has_lost:
            self.stop_autoplay()
            return
        self.step()
        if self.autoplay_running:
            self.root.after(self.delay_ms.get(), self._autoplay_tick)

    # ---------------------------------------------------------------- display
    def refresh(self) -> None:
        for child in self.players_frame.winfo_children():
            child.destroy()

        # Keep the Back button's enabled state in sync with the snapshot stack.
        self.back_btn.configure(state="normal" if self.history else "disabled")

        if self.gm is None:
            return
        gs = self.gm.game_state

        # Compute the density matrix once per refresh; reuse for every hidden cell.
        # `all` view doesn't show hidden cells so density isn't needed.
        density_matrix = self._compute_view_density()

        # Predict the next decision once and route it to both the player grid (for
        # cell-level target highlighting) and the status panel (for the text summary).
        next_actor, next_decision = self._predict_next_decision()
        predicted_targets = self._predicted_target_cells(next_decision)

        for player_idx in range(gs.num_players):
            row = ttk.Frame(self.players_frame)
            row.pack(fill="x", pady=3)

            label = f"P{player_idx}"
            if player_idx == gs.player_to_move and not (gs.has_won or gs.has_lost):
                label = "▶ " + label
            else:
                label = "  " + label
            ttk.Label(row, text=label, width=5, font=("Menlo", 11, "bold")).pack(side="left")

            for position in range(len(gs.player_wires[player_idx])):
                self._draw_wire_cell(
                    row, player_idx, position, density_matrix, predicted_targets,
                )

        self._update_status(next_actor, next_decision)

    def _global_metrics(self) -> tuple[float, float, float]:
        """Return (entropy, combinations_count, weight) for the public-observer view of
        the current state. Cached on game state, so view-mode toggles don't recompute."""
        gs = self.gm.game_state
        cache_key = (
            len(gs.turns),
            len(gs.public_constraints),
            tuple(gs.wire_revealed_counts),
        )
        if self._global_metrics_cache_key == cache_key and self._global_metrics_cache is not None:
            return self._global_metrics_cache

        wire_limits_per_player = np.array(
            [len(hand) for hand in gs.player_wires], dtype=np.int32,
        )
        wire_limits = {
            i: (gs.wire_counts[i], gs.wire_counts[i])
            for i in range(len(gs.wire_ranks))
        }
        density_matrix, _, combinations_count, weight = compute_probability_matrices(
            wire_limits_per_player, wire_limits, list(gs.public_constraints),
        )
        entropy = compute_shannon_entropy(density_matrix)
        metrics = (float(entropy), float(combinations_count), float(weight))
        self._global_metrics_cache_key = cache_key
        self._global_metrics_cache = metrics
        return metrics


    def _predicted_target_cells(self, decision) -> set[tuple[int, int]]:
        """Return the set of (player_index, position) cells that `decision` would touch.
        Used to draw a "next move" outline on the affected cells in the player grid."""
        if decision is None or self.gm is None:
            return set()
        targets: set[tuple[int, int]] = set()
        gs = self.gm.game_state
        if isinstance(decision, SingleCutDecision):
            cut_wire = decision.wire
            hand = gs.player_wires[decision.player_index]
            revealed = gs.revealed_wires[decision.player_index]
            for pos, wire in enumerate(hand):
                if revealed[pos]:
                    continue
                if wire.color != cut_wire.color:
                    continue
                if cut_wire.rank != 0 and wire.rank != cut_wire.rank:
                    continue
                targets.add((decision.player_index, pos))
        elif isinstance(decision, DualCutDecision):
            targets.add((decision.askee_player_index, decision.askee_hand_position))
        elif isinstance(decision, AskeeResponseDecision):
            targets.add((decision.askee_player_index, decision.indicator_wire_position))
        elif isinstance(decision, AskerResponseDecision):
            targets.add((decision.asker_player_index, decision.hand_position))
        elif isinstance(decision, RankIndicatorRevealDecision):
            targets.add((decision.player_index, decision.position))
        # PassDecision and any unknown types: no cell-level target.
        return targets

    def _effective_view_mode(self) -> str:
        """Resolve the active view-mode string. The 'cycle' option follows the player
        whose turn is up next, so its effective mode is `player_<player_to_move>`."""
        mode = self.view_mode.get()
        if mode == "cycle" and self.gm is not None:
            return f"player_{self.gm.game_state.player_to_move}"
        return mode

    def _compute_view_density(self) -> np.ndarray | None:
        """Run compute_probability_matrices with constraints appropriate for the current
        view, so hidden cells can show a best-guess rank + probability. Returns None for
        the 'all' view (no cells are hidden, so the work is skipped)."""
        if self.gm is None:
            return None
        mode = self._effective_view_mode()
        if mode == "all":
            return None

        gs = self.gm.game_state
        cache_key = (
            mode,
            len(gs.turns),
            len(gs.public_constraints),
            tuple(gs.wire_revealed_counts),
        )
        if self._density_cache_key == cache_key and self._density_cache is not None:
            return self._density_cache

        wire_limits_per_player = np.array(
            [len(hand) for hand in gs.player_wires], dtype=np.int32,
        )
        wire_limits = {
            i: (gs.wire_counts[i], gs.wire_counts[i])
            for i in range(len(gs.wire_ranks))
        }
        constraints = list(gs.public_constraints)

        if mode.startswith("player_"):
            viewer = int(mode.split("_", 1)[1])
            for position, wire in enumerate(gs.player_wires[viewer]):
                constraints.append(RankIndicatorConstraint(
                    player_index=viewer,
                    wire_rank_index=gs.wire_to_index_mapping[wire],
                    indicator_location_index=position,
                ))

        density_matrix, *_ = compute_probability_matrices(
            wire_limits_per_player, wire_limits, constraints,
        )
        self._density_cache_key = cache_key
        self._density_cache = density_matrix
        return density_matrix

    def _draw_wire_cell(
            self,
            parent: ttk.Frame,
            player_idx: int,
            position: int,
            density_matrix: np.ndarray | None,
            predicted_targets: set[tuple[int, int]],
    ) -> None:
        gs = self.gm.game_state
        wire = gs.player_wires[player_idx][position]
        is_cut = gs.revealed_wires[player_idx][position]
        has_indicator = self._has_public_indicator(player_idx, position)
        is_publicly_known = is_cut or has_indicator
        visible = self._is_visible_to_view(player_idx, position, is_publicly_known)
        relief = "sunken" if is_cut else "raised"

        # Indicated-but-not-cut cells get an underline so they stand out from plain
        # unrevealed cells without needing a border-relief difference.
        is_indicated_only = has_indicator and not is_cut

        if visible and wire is not None:
            # Indicated visible cells use the color-letter+rank label (e.g. "B7") so the
            # color information stays legible alongside the underline; non-indicated
            # visible cells keep the rank-only label since the cell's background colour
            # already encodes the wire's colour.
            if is_indicated_only:
                primary_text = self._wire_str(wire)
            else:
                primary_text = str(wire.rank) if wire.rank > 0 else "*"
            secondary_text = "✕" if is_cut else ""
            bg = WIRE_BG.get(wire.color, "#888")
            fg = WIRE_FG.get(wire.color, "#000")
        else:
            primary_text, secondary_text, bg, fg = self._best_guess_for_cell(
                density_matrix, player_idx, position,
            )

        primary_font = ("Menlo", 12, "bold underline") if is_indicated_only else ("Menlo", 12, "bold")
        secondary_font = ("Menlo", 8, "underline") if is_indicated_only else ("Menlo", 8)

        # If the predicted next move would touch this cell, add a bright outline so the
        # target is obvious in the player grid.
        is_predicted_target = (player_idx, position) in predicted_targets
        if is_predicted_target:
            highlight_color = "#10b981"  # emerald — high contrast against blue/yellow/red
            highlight_thickness = 3
        else:
            highlight_color = bg
            highlight_thickness = 0

        cell = tk.Frame(
            parent, bg=bg, relief=relief, borderwidth=2,
            highlightbackground=highlight_color, highlightcolor=highlight_color,
            highlightthickness=highlight_thickness,
            padx=4, pady=1,
        )
        tk.Label(
            cell, text=primary_text, bg=bg, fg=fg, font=primary_font,
        ).pack()
        tk.Label(
            cell, text=secondary_text or " ", bg=bg, fg=fg, font=secondary_font,
        ).pack()
        cell.pack(side="left", padx=1)

    def _best_guess_for_cell(
            self,
            density_matrix: np.ndarray | None,
            player_idx: int,
            position: int,
    ) -> tuple[str, str, str, str]:
        """Return (primary_text, secondary_text, bg, fg) for a hidden cell, using the
        best-guess rank from `density_matrix` and its probability. Falls back to a plain
        '?' when no density is available (shouldn't happen outside the 'all' view)."""
        if density_matrix is None:
            return "?", "", HIDDEN_BG, HIDDEN_FG

        gs = self.gm.game_state
        densities = density_matrix[player_idx, position]
        total = float(densities.sum())
        if total < 1e-9:
            return "?", "0%", HIDDEN_BG, HIDDEN_FG

        guess_index = int(np.argmax(densities))
        probability = float(densities[guess_index])
        guess_wire = gs.wire_ranks[guess_index]
        primary_text = self._wire_str(guess_wire)
        secondary_text = f"{probability * 100:.0f}%"
        bg = WIRE_BG.get(guess_wire.color, HIDDEN_BG)
        fg = WIRE_FG.get(guess_wire.color, HIDDEN_FG)
        return primary_text, secondary_text, bg, fg

    def _has_public_indicator(self, player_idx: int, position: int) -> bool:
        for constraint in self.gm.game_state.public_constraints:
            if (
                isinstance(constraint, RankIndicatorConstraint)
                and constraint.player_index == player_idx
                and constraint.indicator_location_index == position
            ):
                return True
        return False

    def _is_visible_to_view(self, player_idx: int, position: int, is_publicly_known: bool) -> bool:
        mode = self._effective_view_mode()
        if mode == "all":
            return True
        if mode == "public":
            return is_publicly_known
        if mode.startswith("player_"):
            viewer = int(mode.split("_", 1)[1])
            return viewer == player_idx or is_publicly_known
        return False

    def _update_status(self, next_actor: int, next_decision) -> None:
        gs = self.gm.game_state
        wires_total = sum(gs.wire_counts)
        wires_cut = sum(gs.wire_revealed_counts)
        public_constraint_count = sum(
            1 for c in gs.public_constraints if isinstance(c, RankIndicatorConstraint)
        )
        terminal = (
            "WON" if gs.has_won
            else "LOST" if gs.has_lost
            else "in progress"
        )
        view_setting = self.view_mode.get()
        view_effective = self._effective_view_mode()
        view_display = view_setting if view_setting == view_effective else f"{view_setting} → {view_effective}"
        if gs.has_won or gs.has_lost:
            next_text = "—"
        elif next_decision is None:
            next_text = f"P{next_actor}: (no legal move)"
        else:
            next_text = f"P{next_actor}: {self._describe_decision(next_decision)}"
        entropy, combinations, weight = self._global_metrics()
        log10_c = f"{math.log10(combinations):.2f}" if combinations > 0 else "—"
        log10_w = f"{math.log10(weight):.2f}"        if weight       > 0 else "—"
        self.status_label.configure(text="\n".join([
            f"view:              {view_display}",
            f"turns played:      {len(gs.turns)}",
            f"to move:           P{gs.player_to_move}",
            f"first round:       {gs.is_first_round}",
            f"health:            {gs.total_health}",
            f"wires cut:         {wires_cut}/{wires_total}",
            f"public indicators: {public_constraint_count}",
            f"status:            {terminal}",
            f"next move:         {next_text}",
            f"H (bits):          {entropy:.2f}",
            f"log₁₀(C):          {log10_c}",
            f"log₁₀(W):          {log10_w}",
        ]))

    def _next_actor_index(self) -> int:
        """Mirror GameManager.process_turn's actor-selection logic so we know who's about
        to act. Returns gs.player_to_move except in mid-turn response phases."""
        gs = self.gm.game_state
        if gs.is_start_of_turn:
            return gs.player_to_move
        last = gs.most_recent_decision
        if isinstance(last, DualCutDecision):
            return last.askee_player_index
        # Fallback (asker response phase, or anything else): the asker is whoever
        # player_to_move points to — turns aren't ended until the asker responds.
        return gs.player_to_move

    def _predict_next_decision(self) -> tuple[int, object]:
        """Run the next acting player's `make_decision` without mutating state, returning
        (actor_index, predicted_decision). `predicted_decision` may be None when the
        player has no legal move (extremely rare under PassDecision handling)."""
        actor = self._next_actor_index()
        gs = self.gm.game_state
        if gs.has_won or gs.has_lost:
            return actor, None
        try:
            predicted = self.gm.players[actor].make_decision(gs)
        except Exception:
            return actor, None
        return actor, predicted

    # --------------------------------------------------------------- logging
    def _describe_decision(self, decision) -> str:
        if isinstance(decision, PassDecision):
            return "pass"
        if isinstance(decision, SingleCutDecision):
            target = "all yellow" if decision.wire.color == YELLOW and decision.wire.rank == 0 else \
                     "all red" if decision.wire.color == RED and decision.wire.rank == 0 else \
                     f"{self._wire_str(decision.wire)}"
            return f"P{decision.player_index} single-cut {target}"
        if isinstance(decision, DualCutDecision):
            return (f"P{decision.asker_player_index} dual-cut "
                    f"{self._wire_str(decision.wire)} → P{decision.askee_player_index} pos {decision.askee_hand_position}")
        if isinstance(decision, AskeeResponseDecision):
            verdict = "✓" if decision.is_successful_dual_cut else "✗"
            return (f"P{decision.askee_player_index} askee {verdict} "
                    f"reveals {self._wire_str(decision.indicator_wire)} at pos {decision.indicator_wire_position}")
        if isinstance(decision, AskerResponseDecision):
            return (f"P{decision.asker_player_index} asker cuts "
                    f"{self._wire_str(decision.wire)} at own pos {decision.hand_position}")
        if isinstance(decision, RankIndicatorRevealDecision):
            return (f"P{decision.player_index} indicates "
                    f"{self._wire_str(decision.wire)} at pos {decision.position}")
        return f"unknown decision {type(decision).__name__}"

    @staticmethod
    def _wire_str(wire) -> str:
        if wire is None:
            return "—"
        color_letter = {BLUE: "B", YELLOW: "Y", RED: "R"}.get(wire.color, "?")
        return f"{color_letter}{wire.rank}" if wire.rank > 0 else f"{color_letter}*"

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    GameUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
