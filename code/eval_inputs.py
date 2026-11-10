"""Loader for the evaluation samples shipped with this repository (data/samples/*.pt).

Each file holds one complete batch of 8 samples of the original validation split, in the original order
(with its batch_index and val_indices), stored as plain {attribute: tensor} dicts. They are rebuilt here into the original GraphData class (code/utils/dataset.py),
so batching and collation are the same as when the samples were read from the original graph cache.
"""
import torch


def load_eval_inputs(path, data_cls=None):
    """Return (list of data objects, list of sample file names) in the stored order."""
    payload = torch.load(path, map_location="cpu")
    if data_cls is None:
        from dataset import GraphData
        data_cls = GraphData
    objs = []
    for sample in payload["samples"]:
        d = data_cls()
        for key, value in sample.items():
            setattr(d, key, value)
        objs.append(d)
    return objs, payload["names"]
