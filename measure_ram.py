"""Measure the peak RSS of a full evaluation run (same scoring as evaluate.py).

Usage:
    python measure_ram.py --checkpoint runs/final-1600-s33/checkpoint-final.pt --split test

Runs the fixed scorer in-process and samples the process RSS from a
background thread; prints the standard result JSON plus the peak RSS.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import psutil
import torch

from common import load_data, make_model, setup
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--split', default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', default='fp32')
    args = p.parse_args()
    device, precision = setup(args.device, args.precision, 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    data = load_data()
    process = psutil.Process()
    peak = {'rss': 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak['rss'] = max(peak['rss'], process.memory_info().rss)
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    result = score(model, *data[args.split], device, precision)
    stop.set()
    sampler.join()
    result.pop('window_nll_nats')
    print(json.dumps(result, indent=2))
    print(f'peak RSS: {peak["rss"] / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
