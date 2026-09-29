"""Average snapshot checkpoints elementwise into a single checkpoint.

Usage:
    python average.py --snapshots runs/full/snapshot-005000.pt ... --output runs/full/checkpoint-avg.pt

Snapshots must share the implementation module and architecture. The averaged
checkpoint keeps the first snapshot's metadata (protocol, implementation,
config, seed); train_tokens is taken from the last snapshot.
"""
import argparse
import torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    first = torch.load(args.snapshots[0], map_location='cpu', weights_only=True)
    model = {k: v.float() for k, v in first['model'].items()}
    for path in args.snapshots[1:]:
        other = torch.load(path, map_location='cpu', weights_only=True)
        if other['implementation'] != first['implementation'] or other['config'] != first['config']:
            raise ValueError(f'Mismatched snapshot: {path}')
        for k in model:
            model[k] += other['model'][k].float()
    for k in model:
        model[k] /= len(args.snapshots)
    first['model'] = model
    first['train_tokens'] = torch.load(args.snapshots[-1], map_location='cpu',
                                       weights_only=True)['train_tokens']
    torch.save(first, args.output)
    print(f'Averaged {len(args.snapshots)} snapshots -> {args.output}')


if __name__ == '__main__':
    main()
