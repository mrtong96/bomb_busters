import time
import numpy as np
from src.logic.constraint import SubsetConstraint
from src.logic.probability_utils import compute_probability_matrices, compute_shannon_entropy


def make_kwargs():
    blue_wires = [(i+1) * 10 for i in range(12)]
    yellow_wires = [21, 51, 71]
    red_wires = [55, 85]
    total_wires = len(blue_wires) * 4 + 3
    wire_limits_per_player = np.array([(total_wires + 4 - i) // 5 for i in range(5)], dtype=np.int32)
    wire_ranks = sorted(blue_wires + yellow_wires + red_wires)
    wire_map = {wire: i for i, wire in enumerate(wire_ranks)}
    yellow_wires_mapped = [wire_map[el] for el in sorted(yellow_wires)]
    red_wires_mapped = [wire_map[el] for el in sorted(red_wires)]
    yellow_subset = SubsetConstraint(wire_rank_indexes=yellow_wires_mapped, subset_count=2)
    red_subset = SubsetConstraint(wire_rank_indexes=red_wires_mapped, subset_count=1)
    wire_limits = {i: (4, 4) if wire in blue_wires else (0, 1) for i, wire in enumerate(wire_ranks)}
    return {
        'wire_limits_per_player': wire_limits_per_player,
        'wire_limits': wire_limits,
        'constraints': [yellow_subset, red_subset]
    }


if __name__ == '__main__':
    kwargs = make_kwargs()

    # warm up numba + caches
    results = compute_probability_matrices(**kwargs)
    # print some stuff to show-case entropy results
    print(np.log2(results[2]), np.log2(results[3]), compute_shannon_entropy(results[0]))

    N = 20
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        compute_probability_matrices(**kwargs)
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    print(f"n={N}  mean={np.mean(times_ms):.1f}ms  std={np.std(times_ms):.1f}ms  "
          f"min={np.min(times_ms):.1f}ms  max={np.max(times_ms):.1f}ms")
