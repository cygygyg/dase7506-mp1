"""Sweep the temperature (logit-scaling) setting on validation only.

Usage:
    python sweep_temp.py --checkpoint runs/final-1600/checkpoint-avg.pt --temps 0.90,0.95,0.98,1.00,1.02,1.05,1.10

Rebuilds the checkpoint's model with the given temperature in its config,
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
    p.add_argument('--temps', default='0.90,0.95,0.98,1.00,1.02,1.05,1.10')
    args = p.parse_args()
    device, _ = setup('cuda', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = load_data()
    for temp in [float(x) for x in args.temps.split(',')]:
        config = dict(checkpoint['config'])
        config['temperature'] = temp
        model, _ = make_model(checkpoint['implementation'], config, device)
        model.load_state_dict(checkpoint['model'])
        result = score(model, *data['validation'], device, 'fp32')
        print(f'temperature {temp:.2f} -> validation bpb {result["bpb"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
