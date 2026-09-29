"""Sweep ngram_lambda (static backoff n-gram interpolation) on validation.

The checkpoint must already contain the n-gram buffers (see build_ngram.py
and the final-checkpoint assembly step). Optionally also sweeps temperature.

Usage:
    python sweep_ngram.py --checkpoint runs/final-1600-s33/ck-ngram.pt --lambdas 0.05,0.10,0.15,0.20 --temps 1.10
"""
import argparse
import torch
from pathlib import Path
from common import load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--lambdas', default='0.05,0.10,0.15,0.20')
    p.add_argument('--temps', default='1.10')
    args = p.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = load_data()
    for temp in [float(x) for x in args.temps.split(',')]:
        for lam in [float(x) for x in args.lambdas.split(',')]:
            config = dict(checkpoint['config'])
            config['temperature'] = temp
            config['ngram_lambda'] = lam
            model, _ = make_model(checkpoint['implementation'], config, device)
            model.load_state_dict(checkpoint['model'])
            result = score(model, *data['validation'], device, 'fp32')
            print(f'temp {temp:.2f} ngram {lam:.2f} -> validation bpb {result["bpb"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
