"""Check every cell of the chapter's Table 2 against the training records shipped in checkpoints/magnet/.

Each Table 2 value is the per-epoch minimum of the validation ADE / FDE stored in the run's
results/loss_record_*.json ("best_ade" / "best_fde"). Prints one line per run: 20 MATCH lines (the 20 cells)
and 2 DIFF lines (two unused high-speed Default runs with 450 and 4500 samples).

Usage: python check_table2.py [checkpoints/magnet]
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "magnet")
paper = {
    ("high_lateral_acceleration", "default"): (2.13, 4.20),
    ("high_lateral_acceleration", "ego_only"): (2.06, 3.87),
    ("high_lateral_acceleration", "nearby_only"): (1.58, 3.31),
    ("high_lateral_acceleration", "no_ego_road"): (2.34, 4.57),
    ("high_magnitude_speed", "default"): (3.09, 6.89),
    ("high_magnitude_speed", "ego_only"): (3.01, 6.38),
    ("high_magnitude_speed", "nearby_only"): (2.53, 5.48),
    ("high_magnitude_speed", "no_ego_road"): (2.92, 5.99),
    ("medium_magnitude_speed", "default"): (2.71, 5.33),
    ("medium_magnitude_speed", "ego_only"): (2.68, 5.31),
    ("medium_magnitude_speed", "nearby_only"): (1.34, 3.51),
    ("medium_magnitude_speed", "no_ego_road"): (1.73, 4.09),
    ("near_multiple_vehicles", "default"): (1.51, 2.77),
    ("near_multiple_vehicles", "ego_only"): (1.38, 2.43),
    ("near_multiple_vehicles", "nearby_only"): (1.80, 3.35),
    ("near_multiple_vehicles", "no_ego_road"): (1.62, 2.94),
    ("traversing_intersection", "default"): (2.45, 5.47),
    ("traversing_intersection", "ego_only"): (2.50, 5.71),
    ("traversing_intersection", "nearby_only"): (3.18, 6.68),
    ("traversing_intersection", "no_ego_road"): (3.06, 6.40),
}
for f in sorted(glob.glob(f"{root}/results_type_*_adjacency/*/results/loss_record_*.json")):
    run = f.split("/")[-3]
    cell = f.split("/")[-4].replace("results_type_", "").replace("_adjacency", "")
    topo = next(t for t in ("default", "ego_only", "nearby_only", "no_ego_road") if cell.endswith(t))
    scen = cell[: -len(topo) - 1]
    d = json.load(open(f))
    ba, bf = d["best_ade"], d["best_fde"]
    pa, pf = paper[(scen, topo)]
    ok = round(ba, 2) == pa and round(bf, 2) == pf
    print(f"{scen:26s} {topo:12s} {run:45s} ep={d['epochs_trained']:3d} n={d['dataset_size']:5d}/{d['validation_size']:4d} "
          f"best {ba:.3f}/{bf:.3f}  table {pa:.2f}/{pf:.2f}  {'MATCH' if ok else 'DIFF'}")
