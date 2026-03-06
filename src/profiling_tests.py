import numpy as np

from src.constraints2 import SubsetConstraint
from src.game_state_possibilities2 import compute_probability_matrices


def main():
    # 4 blue wires, 2/3 yellow, 1/2 red
    blue_wires = [(i+1) * 10 for i in range(12)]
    yellow_wires = [21, 51, 71]
    red_wires = [55, 85]

    total_wires = len(blue_wires) * 4 + 3
    wire_limits_per_player = np.array([(total_wires + 4 - i) // 5 for i in range(5)], dtype=np.int32)
    wire_ranks = sorted(blue_wires + yellow_wires + red_wires)
    wire_map = {wire: i for i, wire in enumerate(wire_ranks)}
    wire_ranks_mapped = list(range(len(wire_ranks)))
    yellow_wires_mapped = [wire_map[el] for el in sorted(yellow_wires)]
    red_wires_mapped = [wire_map[el] for el in sorted(red_wires)]
    yellow_subset = SubsetConstraint(wire_limits_per_player, wire_ranks_mapped, yellow_wires_mapped, 2)
    red_subset = SubsetConstraint(wire_limits_per_player, wire_ranks_mapped, red_wires_mapped, 1)

    wire_limits = {i: (4, 4) if wire in blue_wires else (0, 1) for i, wire in enumerate(wire_ranks)}

    results = compute_probability_matrices(
        wire_limits_per_player=wire_limits_per_player,
        wire_limits=wire_limits,
        constraints=[yellow_subset, red_subset],
    )

if __name__ == '__main__':
    main()