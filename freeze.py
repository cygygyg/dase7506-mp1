"""Freeze the final checkpoint: set the validation-chosen settings into the
checkpoint config and save under a new name.

Usage:
    python freeze.py --checkpoint runs/w208-4000/avg-XXXX-YYYY.pt \
        --output runs/w208-4000/checkpoint-final.pt \
        --temperature 1.05 --cache-lambda 0.09 --cache-beta 0.10
"""
import argparse
import torch
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--temperature', type=float, default=1.05)
    p.add_argument('--cache-lambda', type=float, default=0.0)
    p.add_argument('--cache-beta', type=float, default=1.0)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'])
    config['temperature'] = args.temperature
    config['cache_lambda'] = args.cache_lambda
    config['cache_beta'] = args.cache_beta
    checkpoint['config'] = config
    torch.save(checkpoint, args.output)
    print(f'frozen {args.output}: temperature={args.temperature}, '
          f'cache_lambda={args.cache_lambda}, cache_beta={args.cache_beta}')


if __name__ == '__main__':
    main()
