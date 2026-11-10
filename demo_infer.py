"""Inference demo and reproduction check for the original MAGNET experiments (book chapter version).

The repository ships one complete evaluation batch (8 samples) of the original validation split for each of the two
scenarios whose original data survives, the eight checkpoints of those scenarios, and reference values computed on
the complete validation splits (reference/full_validation_reference.json). All computation uses the original code in
code/ (training-time versions; comments translated, logic unchanged).

  verify  Runs the original train.validate() on each shipped batch with each of the four checkpoints and compares
          val_loss / ADE / FDE with the reference values of that batch. Then shows that the reference values of all
          batches average to the training record of the checkpoint epoch (the number train.validate() returned on the
          complete split during training).
  show    Step-by-step walk-through of the inference of one shipped sample, plus a bird's-eye-view figure.

Examples
  python demo_infer.py verify --device cuda:0
  python demo_infer.py show --scenario high_lateral_acceleration --sample 3 --fig_dir figures/demo
  python demo_infer.py show --random --fig_dir figures/demo
"""
import argparse
import contextlib
import glob
import inspect
import io
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(REPO, "code")
# Original code first on sys.path, so nothing else with the same module names can be picked up.
sys.path.insert(0, os.path.join(CODE_DIR, "utils"))
sys.path.insert(0, CODE_DIR)

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import config as orig_config  # noqa: E402
    import train as orig_train  # noqa: E402
    from transformer_predictor import TransformerPredictor  # noqa: E402
    from eval_inputs import load_eval_inputs  # noqa: E402

for _obj in (orig_train, TransformerPredictor):
    assert inspect.getfile(_obj).startswith(CODE_DIR), f"not the shipped original: {inspect.getfile(_obj)}"

CKPT_DIR = os.path.join(REPO, "checkpoints", "magnet")
SAMPLE_DIR = os.path.join(REPO, "data", "samples")
REFERENCE = os.path.join(REPO, "reference", "full_validation_reference.json")
TOPOLOGIES = ("default", "ego_only", "nearby_only", "no_ego_road")
SCENARIOS = ("high_lateral_acceleration", "near_multiple_vehicles")
# Values printed in Table 2 of the chapter (ADE, FDE) for the two scenarios shipped here.
PAPER_TABLE2 = {
    ("high_lateral_acceleration", "default"): (2.13, 4.20), ("high_lateral_acceleration", "ego_only"): (2.06, 3.87),
    ("high_lateral_acceleration", "nearby_only"): (1.58, 3.31), ("high_lateral_acceleration", "no_ego_road"): (2.34, 4.57),
    ("near_multiple_vehicles", "default"): (1.51, 2.77), ("near_multiple_vehicles", "ego_only"): (1.38, 2.43),
    ("near_multiple_vehicles", "nearby_only"): (1.80, 3.35), ("near_multiple_vehicles", "no_ego_road"): (1.62, 2.94),
}
TOL = {"val_loss": 1e-4, "ade": 1e-4, "fde": 1e-4}


def ckpt_path(scenario, topology):
    hits = sorted(glob.glob(os.path.join(CKPT_DIR, f"results_type_{scenario}_{topology}_adjacency", "training_*",
                                         "models", f"best_model_{topology}.pth")))
    assert len(hits) == 1, f"expected one checkpoint for {scenario}/{topology}, found {hits}"
    return hits[0]


def load_model(ckpt, device):
    with contextlib.redirect_stdout(io.StringIO()):
        model = TransformerPredictor()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()


def load_batch(scenario):
    """The shipped batch of a scenario: (list of 8 data objects, sample names, batch index in the original split)."""
    hits = glob.glob(os.path.join(SAMPLE_DIR, f"{scenario}_batch*.pt"))
    assert len(hits) == 1, hits
    samples, names = load_eval_inputs(hits[0])
    batch_index = torch.load(hits[0], map_location="cpu")["batch_index"]
    return samples, names, batch_index


def run_validate(model, samples, device, log_path):
    """Call the original train.validate() on one batch; its verbose debug output goes to a log file."""
    criterion = orig_train.ImprovedTrajectoryLoss(ade_weight=0.5, fde_weight=3.0, smoothness_weight=0.3, direction_weight=0.5)
    loader = DataLoader(samples, batch_size=orig_config.BATCH_SIZE)
    with open(log_path, "a") as fh, contextlib.redirect_stdout(fh):
        return orig_train.validate(model, loader, criterion, device)


def cmd_verify(args, device):
    ref = json.load(open(REFERENCE))
    print(f"[split] original split train_test_split(test_size={orig_config.VALIDATION_SPLIT}, "
          f"random_state={orig_config.SEED}); batch_size={orig_config.BATCH_SIZE}, no shuffle")
    print("\n(1) Shipped batches, original validate(), compared with the reference values of the same batch")
    print(f"{'scenario':26s} {'run label':12s} {'batch':>5s}  {'reproduced val_loss/ADE/FDE':>28s}  "
          f"{'reference val_loss/ADE/FDE':>28s}  match")
    n_ok = n_all = 0
    for sc in SCENARIOS:
        samples, names, bi = load_batch(sc)
        sc_ref = ref["scenarios"][sc]
        assert names == sc_ref["shipped_names"], "shipped samples differ from the reference batch"
        for topo in TOPOLOGIES:
            vl, ade, fde = run_validate(load_model(ckpt_path(sc, topo), device), samples, device, args.log)
            r = sc_ref["runs"][topo]["batches"][bi]
            ok = all(abs(x - r[k]) <= TOL[k] for x, k in ((vl, "val_loss"), (ade, "ade"), (fde, "fde")))
            n_ok += ok
            n_all += 1
            print(f"{sc:26s} {topo:12s} {bi:5d}  {vl:10.4f}/{ade:7.4f}/{fde:<8.4f}  "
                  f"{r['val_loss']:10.4f}/{r['ade']:7.4f}/{r['fde']:<8.4f}  {'YES' if ok else 'NO'}", flush=True)

    print("\n(2) Reference values of all batches of the complete validation split (sample-weighted mean),"
          "\n    compared with the training record of the checkpoint epoch and with Table 2 of the chapter")
    print(f"{'scenario':26s} {'run label':12s} {'n':>5s}  {'all batches val_loss/ADE/FDE':>29s}  "
          f"{'training record':>28s}  match  Table 2 (min over epochs)")
    n2_ok = n2_all = 0
    for sc in SCENARIOS:
        for topo in TOPOLOGIES:
            run = ref["scenarios"][sc]["runs"][topo]
            batches = run["batches"]
            n = sum(b["n"] for b in batches)
            agg = {k: sum(b[k] * b["n"] for b in batches) / n for k in ("val_loss", "ade", "fde")}
            rec = run["record_at_ckpt_epoch"]
            ok = all(round(agg[k], 4) == round(rec[k], 4) for k in ("val_loss", "ade", "fde"))
            n2_ok += ok
            n2_all += 1
            p = PAPER_TABLE2[(sc, topo)]
            print(f"{sc:26s} {topo:12s} {n:5d}  {agg['val_loss']:10.4f}/{agg['ade']:7.4f}/{agg['fde']:<8.4f}  "
                  f"ep{rec['epoch']:<3d} {rec['val_loss']:.4f}/{rec['ade']:.4f}/{rec['fde']:.4f}  "
                  f"{'YES' if ok else 'NO ':3s}    {p[0]:.2f}/{p[1]:.2f}")
    print(f"\n[verify] shipped batches reproduced: {n_ok}/{n_all}; reference = training record: {n2_ok}/{n2_all}")
    if n_ok != n_all or n2_ok != n2_all:
        sys.exit(1)


def cmd_show(args, device):
    """Walk through one shipped sample inside its batch of 8 (the batching used for the reported numbers)."""
    if args.random:
        args.scenario = args.scenario or random.choice(SCENARIOS)
        if args.sample is None:
            args.sample = random.randrange(orig_config.BATCH_SIZE)
    if not args.scenario:
        sys.exit("show: give --scenario (or --random)")
    if args.sample is None:
        args.sample = 0
    samples, names, bi = load_batch(args.scenario)
    k = args.sample
    if not 0 <= k < len(samples):
        sys.exit(f"--sample must be 0..{len(samples) - 1}")
    topo = args.topology
    val_index = bi * orig_config.BATCH_SIZE + k
    batch = next(iter(DataLoader(samples, batch_size=orig_config.BATCH_SIZE))).to(device)
    model = load_model(ckpt_path(args.scenario, topo), device)
    B = int(batch.batch.max()) + 1
    caught = {}

    def hook(name_):
        def fn(_m, inp, out):
            caught.setdefault(name_, []).append((inp, out))
        return fn

    for name_ in ("ego_history_encoder", "context_transformer", "gnn_encoder", "feature_fusion", "output_head"):
        getattr(model, name_).register_forward_hook(hook(name_))
    with torch.no_grad():
        out = model(batch)

    sel = batch.batch == k
    is_agent = model._create_feature_mask(batch.x[sel]).bool()
    print(f"\n=== Inference walk-through: {args.scenario}, run label {topo}, sample {k} of batch {bi} "
          f"(validation sample #{val_index}, {names[k]}) ===")
    print(f"[1] Input polylines: {int(sel.sum())} vector segments (12-d each); "
          f"{int(is_agent.sum())} agent segments, {int((~is_agent).sum())} map segments")
    print(f"    GNN graph (whole batch): gnn_x {tuple(batch.gnn_x.shape)}, edges {batch.gnn_edge_index.shape[1]}")
    print(f"    ego_history tensor in batch: {tuple(batch.ego_history.shape)}  (stored as [20,7] per sample)")
    if "ego_history_encoder" in caught:
        print("[2] Ego branch: executed")
    else:
        print("[2] Ego branch: NOT executed -> ego_history is [B*20,7]; view(B,20,2) raises inside try/except,"
              " branch output stays zero")
    ctx = caught.get("context_transformer", [])
    print(f"[3] Context Transformer: called {len(ctx)} times (once per sample); "
          f"sample {k}: tokens {tuple(ctx[k][1].shape)} -> mean-pooled to 128-d")
    g_out = caught["gnn_encoder"][0][1]
    print(f"[4] GNN (2x GCN 9->64->128): output {tuple(g_out.shape)}"
          + ("  -> ONE vector for the whole batch (no gnn_batch), repeated to every sample" if g_out.shape[0] == 1 else ""))
    z = caught["feature_fusion"][0][0][0][k]
    print(f"[5] Late fusion: concat [ego | context | gnn] = 384-d; L2 norms "
          f"{z[:128].norm():.3f} | {z[128:256].norm():.3f} | {z[256:].norm():.3f} -> Linear(384,128)")
    pred_inc, true_inc = out.view(B, 60, 2), batch.y.view(B, 60, 2)
    traj_pred, traj_true = pred_inc[k].cumsum(0), true_inc[k].cumsum(0)
    print("[6] Output head: 120 values -> 60 x 2 position increments (0.1 s, 6 s horizon) -> cumulative sum")
    print(f"    predicted endpoint ({traj_pred[-1, 0]:.2f}, {traj_pred[-1, 1]:.2f}) m, "
          f"ground truth ({traj_true[-1, 0]:.2f}, {traj_true[-1, 1]:.2f}) m (ego frame)")
    # As in the original validate(): trajectories restored from ego_last_pos (global UTM coordinates) in float32.
    origin = batch.ego_last_pos.view(B, 2)
    err = (orig_train.restore_trajectory_from_increments(origin, pred_inc)
           - orig_train.restore_trajectory_from_increments(origin, true_inc)).norm(dim=2)[k]
    err_ego = (traj_pred - traj_true).norm(dim=1)
    r = json.load(open(REFERENCE))["scenarios"][args.scenario]["runs"][topo]["batches"][bi]
    print(f"[7] Sample metrics as computed by the original validate(): ADE {err.mean():.3f} m, FDE {err[-1]:.3f} m "
          f"(reference {r['sample_ade'][k]:.3f} / {r['sample_fde'][k]:.3f})")
    print(f"    same prediction measured in the ego frame:           ADE {err_ego.mean():.3f} m, FDE {err_ego[-1]:.3f} m"
          f"  (see README, known issues)")

    if args.fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = batch.x[sel].cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        for row, ag in zip(x, is_agent.cpu().numpy()):
            ax.plot([row[0], row[2]], [row[1], row[3]], color="tab:blue" if ag else "0.75",
                    lw=1.6 if ag else 0.8, zorder=2 if ag else 1)
        tt, tp = traj_true.cpu().numpy(), traj_pred.cpu().numpy()
        ax.plot(tt[:, 0], tt[:, 1], color="tab:green", lw=2.2, label="Ground truth (6 s)", zorder=3)
        ax.plot(tp[:, 0], tp[:, 1], color="tab:red", lw=2.2, ls="--", label="Prediction", zorder=4)
        ax.scatter([0], [0], color="k", s=30, zorder=5, label="Ego (t = 0)")
        ax.plot([], [], color="tab:blue", label="Agent history segments")
        ax.plot([], [], color="0.75", label="Lane segments")
        ax.set_title(f"{args.scenario} / run label: {topo} (trained on default graph)\n"
                     f"validation sample #{val_index}: ADE {err_ego.mean():.2f} m, FDE {err_ego[-1]:.2f} m (ego frame)",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, ego frame)")
        ax.set_ylabel("y (m, ego frame)")
        ax.legend(fontsize=8, loc="best")
        os.makedirs(args.fig_dir, exist_ok=True)
        stem = os.path.join(args.fig_dir, f"{args.scenario}_{topo}_val{val_index:04d}")
        fig.savefig(stem + ".png", dpi=160, bbox_inches="tight")
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        print(f"[fig] {stem}.png\n      {stem}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["verify", "show"])
    ap.add_argument("--scenario", choices=SCENARIOS, help="show: scenario (random with --random if omitted)")
    ap.add_argument("--topology", default="nearby_only", choices=TOPOLOGIES, help="show: run label (default nearby_only)")
    ap.add_argument("--sample", type=int, default=None, help="show: position 0-7 in the shipped batch (default 0)")
    ap.add_argument("--random", action="store_true", help="show: pick a random shipped sample")
    ap.add_argument("--fig_dir", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--log", default=os.path.join(REPO, "validate_debug.log"),
                    help="where the verbose debug output of the original validate() goes")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    {"verify": cmd_verify, "show": cmd_show}[args.mode](args, device)


if __name__ == "__main__":
    main()
