"""Build the shipped samples and the reference values of this repository from the full reproduction package.

Run once on the original machine (paths are machine-specific). For each scenario whose original data survives
(high_lateral_acceleration, near_multiple_vehicles) and each of its four checkpoints it
  1. evaluates the complete original validation split batch by batch (batch 8, original order) with the original
     train.validate(), and records val_loss / ADE / FDE of every batch plus the ADE / FDE of every sample;
  2. checks that the sample-weighted mean over all batches equals the training record of the checkpoint epoch
     (this is what train.validate() returns on the whole split);
  3. draws one complete batch per scenario with a fixed seed and exports its 8 samples to data/samples/.
Outputs: data/samples/<scenario>_batch<k>.pt and reference/full_validation_reference.json.

Usage: python provenance/build_from_full_package.py --full_package /path/to/MAGNET_original_repro --device cuda:0
"""
import argparse
import contextlib
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ("high_lateral_acceleration", "near_multiple_vehicles")
TOPOLOGIES = ("default", "ego_only", "nearby_only", "no_ego_road")
SEED = 20261110
BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_package", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    full = os.path.abspath(args.full_package)
    sys.path.insert(0, os.path.join(full, "code", "utils"))
    sys.path.insert(0, os.path.join(full, "code"))
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        import train as orig_train
        from transformer_predictor import TransformerPredictor
        from eval_inputs import load_eval_inputs
    from torch_geometric.loader import DataLoader

    device = torch.device(args.device)
    criterion = orig_train.ImprovedTrajectoryLoss(ade_weight=0.5, fde_weight=3.0, smoothness_weight=0.3,
                                                  direction_weight=0.5)
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(REPO, "data", "samples"), exist_ok=True)
    reference = {"seed": SEED, "batch_size": BATCH, "scenarios": {}}

    for sc in SCENARIOS:
        path = os.path.join(full, "data", "eval_inputs", f"val_{sc}.pt")
        payload = torch.load(path, map_location="cpu")
        samples, names = load_eval_inputs(path)
        n = len(samples)
        n_batches = (n + BATCH - 1) // BATCH
        chosen = int(rng.integers(0, n // BATCH))  # complete batches only
        b0 = chosen * BATCH
        out = {"kind": payload["kind"], "source": payload["source"], "keys": payload["keys"],
               "samples": payload["samples"][b0:b0 + BATCH], "names": names[b0:b0 + BATCH],
               "scenario": sc, "batch_index": chosen, "val_indices": list(range(b0, b0 + BATCH))}
        torch.save(out, os.path.join(REPO, "data", "samples", f"{sc}_batch{chosen:03d}.pt"))
        del payload

        sc_ref = {"n_samples": n, "n_batches": n_batches, "shipped_batch": chosen,
                  "shipped_names": names[b0:b0 + BATCH], "runs": {}}
        for topo in TOPOLOGIES:
            ckpt = sorted(glob.glob(os.path.join(full, "checkpoints", "magnet", f"results_type_{sc}_{topo}_adjacency",
                                                 "training_*", "models", f"best_model_{topo}.pth")))
            assert len(ckpt) == 1, ckpt
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                model = TransformerPredictor()
            model.load_state_dict(torch.load(ckpt[0], map_location="cpu"), strict=True)
            model = model.to(device).eval()

            batches = []
            for bi in range(n_batches):
                part = samples[bi * BATCH:(bi + 1) * BATCH]
                loader = DataLoader(part, batch_size=BATCH)
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    vl, ade, fde = orig_train.validate(model, loader, criterion, device)
                batch = next(iter(DataLoader(part, batch_size=BATCH))).to(device)
                model.eval()  # the original validate() ends with model.train()
                with torch.no_grad():
                    out_ = model(batch)
                b = len(part)
                pred_inc, true_inc = out_.view(b, 60, 2), batch.y.view(b, 60, 2)
                # As in the original validate(): ego_history ([B*20, 7]) matches none of its branches, so the
                # trajectories are restored from ego_last_pos (global UTM coordinates) in float32.
                origin = batch.ego_last_pos.view(b, 2)
                err = (orig_train.restore_trajectory_from_increments(origin, pred_inc)
                       - orig_train.restore_trajectory_from_increments(origin, true_inc)).norm(dim=2)
                s_ade, s_fde = err.mean(1).tolist(), err[:, -1].tolist()
                assert abs(np.mean(s_ade) - ade) < 1e-4 and abs(np.mean(s_fde) - fde) < 1e-4, (bi, ade, fde)
                err_ego = (pred_inc.cumsum(1) - true_inc.cumsum(1)).norm(dim=2)  # same, in the ego frame
                batches.append({"n": b, "val_loss": vl, "ade": ade, "fde": fde, "sample_ade": s_ade, "sample_fde": s_fde,
                                "sample_ade_ego_frame": err_ego.mean(1).tolist(),
                                "sample_fde_ego_frame": err_ego[:, -1].tolist()})

            N = sum(x["n"] for x in batches)
            agg = {k: sum(x[k] * x["n"] for x in batches) / N for k in ("val_loss", "ade", "fde")}
            rec_file = glob.glob(os.path.join(os.path.dirname(os.path.dirname(ckpt[0])), "results", "loss_record_*.json"))[0]
            rec = json.load(open(rec_file))
            i = int(np.argmin(rec["val_loss"]))
            record = {"epoch": i + 1, "val_loss": rec["val_loss"][i], "ade": rec["ade_history"][i],
                      "fde": rec["fde_history"][i]}
            print(f"{sc:26s} {topo:12s} batch {chosen:3d}: {batches[chosen]['ade']:.4f}/{batches[chosen]['fde']:.4f}   "
                  f"all {N} samples: {agg['val_loss']:.4f}/{agg['ade']:.4f}/{agg['fde']:.4f}   "
                  f"record ep{record['epoch']}: {record['val_loss']:.4f}/{record['ade']:.4f}/{record['fde']:.4f}", flush=True)
            sc_ref["runs"][topo] = {"checkpoint": os.path.relpath(ckpt[0], full), "record_at_ckpt_epoch": record,
                                    "aggregate": agg, "batches": batches}
        reference["scenarios"][sc] = sc_ref

    os.makedirs(os.path.join(REPO, "reference"), exist_ok=True)
    with open(os.path.join(REPO, "reference", "full_validation_reference.json"), "w") as fh:
        json.dump(reference, fh)


if __name__ == "__main__":
    main()
