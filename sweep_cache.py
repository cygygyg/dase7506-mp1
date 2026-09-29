"""Sweep cache_lambda (within-window flat cache interpolation) on validation.

Usage:
    python sweep_cache.py --checkpoint runs/final-1600-s33/checkpoint-final.pt --lambdas 0.0,0.05,0.10,0.15,0.20,0.25

Rebuilds the checkpoint's model with the given cache_lambda in its config,
scores the validation split on GPU FP32 and prints the BPB per value.
"""
import argparse
import torch
from pathlib import Path
from common import load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--lambdas', default='0.0,0.05,0.10,0.15,0.20,0.25,0.30')
    p.add_argument('--betas', default='1.0')
    p.add_argument('--temps', default='1.10')
    args = p.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = load_data()
    for temp in [float(x) for x in args.temps.split(',')]:
        for beta in [float(x) for x in args.betas.split(',')]:
            for lam in [float(x) for x in args.lambdas.split(',')]:
                config = dict(checkpoint['config'])
                config['temperature'] = temp
                config['cache_lambda'] = lam
                config['cache_beta'] = beta
                model, _ = make_model(checkpoint['implementation'], config, device)
                model.load_state_dict(checkpoint['model'])
                result = score(model, *data['validation'], device, 'fp32')
                print(f'temp {temp:.2f} beta {beta:.2f} lambda {lam:.2f} -> validation bpb {result["bpb"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
