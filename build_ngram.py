"""Build the static backoff n-gram asset from the TRAINING text only.

Counts contexts of length 1..3 (bigram..4-gram) plus unigram counts over the
supplied WikiText-2 training split. Contexts are packed into int64 keys
(vocab < 2^11, 11 bits per token, little-endian) sorted per order for
binary-search lookup. The student model loads this asset into its state
buffers and interpolates the backoff prior into its predictions with weight
`ngram_lambda`, tuned on validation.

Usage:
    python build_ngram.py --output runs/ngram-asset.pt
"""
import argparse
import collections
import numpy as np
import torch
from pathlib import Path

from common import ROOT, load_data

BITS = 11  # vocab 2048 < 2^11


def pack(tokens):
    value = 0
    for i, t in enumerate(tokens):
        value |= t << (BITS * i)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/ngram-asset.pt')
    p.add_argument('--max-order', type=int, default=4)
    p.add_argument('--min-count', type=int, default=2,
                   help='Drop (context, next) pairs seen fewer times; singleton counts are noise.')
    args = p.parse_args()
    tokens = load_data()['train'][0].tolist()
    vocab = 2048
    unigram = np.zeros(vocab, dtype=np.int64)
    for t in tokens:
        unigram[t] += 1
    tables = []
    offsets = [0]
    ctx_all, next_all, cnt_all = [], [], []
    for order in range(1, args.max_order):  # context lengths 1..3
        counts = collections.defaultdict(int)
        for i in range(len(tokens) - order):
            counts[(tuple(tokens[i:i + order]), tokens[i + order])] += 1
        rows = sorted((pack(ctx), nxt, c) for (ctx, nxt), c in counts.items() if c >= args.min_count)
        ctx = np.asarray([r[0] for r in rows], dtype=np.int64)
        nxt = np.asarray([r[1] for r in rows], dtype=np.int16)
        cnt = np.asarray([r[2] for r in rows], dtype=np.int32)
        ctx_all.append(ctx); next_all.append(nxt); cnt_all.append(cnt)
        offsets.append(offsets[-1] + len(rows))
        print(f'context length {order}: {len(rows):,} rows', flush=True)
    asset = {'unigram': torch.from_numpy(unigram),
             'total': sum(unigram.tolist()),
             'ctx': torch.from_numpy(np.concatenate(ctx_all)),
             'next': torch.from_numpy(np.concatenate(next_all)),
             'cnt': torch.from_numpy(np.concatenate(cnt_all)),
             'offsets': torch.tensor(offsets, dtype=torch.int64),
             'max_order': args.max_order}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asset, args.output)
    size = args.output.stat().st_size / 2**20
    print(f'saved {args.output} ({size:.1f} MiB)')


if __name__ == '__main__':
    main()
