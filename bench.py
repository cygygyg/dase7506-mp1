"""Quick GPU step-time benchmark for candidate configs (30 steps each)."""
import argparse
import time
import torch
from torch.nn import functional as F

from common import load_data, make_model, setup

p = argparse.ArgumentParser()
p.add_argument('--implementation', default='student')
p.add_argument('--configs', default='configs/student.json,configs/student-208.json')
args = p.parse_args()

device, precision = setup('cuda', 'bf16', 4)
tokens = load_data()['train'][0].to(device)
for path in args.configs.split(','):
    import json
    config = json.loads(open(path).read())
    torch.manual_seed(17)
    model, _ = make_model(args.implementation, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    rng = torch.Generator().manual_seed(17)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(30):
        starts = torch.randint(len(tokens) - 257, (128,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    params = sum(p.numel() for p in model.parameters())
    print(f'{path}: {elapsed / 30 * 1000:.0f} ms/step, {params / 1e6:.2f}M params')
