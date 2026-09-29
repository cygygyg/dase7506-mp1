# MP1 — Small Language Model Challenge (DASE7506)

Student: `3036808630` · Final submission: 30 September 2026 (UTC+8)

**Result: test BPB 1.55596** (FP32, CPU; classroom baseline 2.10273 on the
same machine → −0.54677 BPB, −26%). See REPORT.md for method, experiments,
comparisons, ablation and critical analysis.

## Final model

- Architecture (`student.py`): causal GPT, context 256, vocab 2048, tied
  embeddings. Width 208, depth 6, heads 4, SwiGLU mult 555, RoPE, RMSNorm,
  no biases, depth-scaled residual init, dropout 0.05
  (`configs/student-208.json`). ≈3.54M parameters, ≈4.0× baseline FLOPs.
- Training: seed 33, batch 128×256, 4000 steps (131.1M targets), AdamW
  (lr 1e-3, warmup 100, cosine to 10%, weight decay 0.2), grad clip 1.0,
  BF16 on an RTX 3060 Laptop GPU.
- Post-processing (all selected on validation only): uniform average of the
  6 snapshots at steps 1200–3200 (`average.py`), logit temperature τ=1.12,
  within-window flat cache (Grave et al. 2017) with λ=0.07, β=0.08.
- Frozen checkpoint: `runs/w208-4000/checkpoint-final.pt` (also attached to
  the GitHub release).

## Reproduce

Environment used: Windows 11, Python 3.12.10, PyTorch 2.7.1+cu126, RTX 3060
Laptop (6 GiB, driver 596). All commands run from the repository root
(which contains `train.py`, `evaluate.py`, `data/`, ... — the original
course package's `code/` directory). The unmodified course README is kept as
`UPSTREAM_README.md`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Evaluate the submitted checkpoint (ranked setting — CPU, FP32):

```powershell
python evaluate.py --checkpoint runs/w208-4000/checkpoint-final.pt --device cpu --precision fp32 --split test
```

Expected output: `"bpb": 1.5559579447880931` (≈1.55596; reproduced twice on
the author's machine; 17.1 s on idle CPU vs 5.86 s for the baseline).

Retrain from scratch (new output directory per run):

```powershell
python train.py --implementation student --config configs/student-208.json --device cuda --precision bf16 --seed 33 --batch-size 128 --steps 4000 --weight-decay 0.2 --checkpoint-every 400 --eval-every 400 --run-dir runs/w208-repro
python average.py --snapshots runs/w208-repro/snapshot-001200.pt runs/w208-repro/snapshot-001600.pt runs/w208-repro/snapshot-002000.pt runs/w208-repro/snapshot-002400.pt runs/w208-repro/snapshot-002800.pt runs/w208-repro/snapshot-003200.pt --output runs/w208-repro/checkpoint-avg.pt
python freeze.py --checkpoint runs/w208-repro/checkpoint-avg.pt --output runs/w208-repro/checkpoint-final.pt --temperature 1.12 --cache-lambda 0.07 --cache-beta 0.08
```

Other experiments (baseline, same-token comparison, ablation) use the
unmodified `model.py` / `configs/ablation.json` with the commands listed in
RUN_LOG.csv.

## Budget compliance (same frozen predictor, same machine)

| Limit | Measured |
|---|---|
| CPU scoring ≤5× baseline | 17.1 s vs 5.86 s = 2.9× |
| Peak eval RAM ≤4 GiB | 1097 MiB |
| Inference assets ≤64 MiB | 15 MiB (`checkpoint-final.pt`) |

## AI assistance disclosure

This submission was developed with substantial assistance from Claude Code
(Anthropic), which contributed to: implementation of the student
architecture (RoPE / RMSNorm / SwiGLU / depth-scaled init), the
within-window flat cache, training-script extensions (`--lr`,
`--weight-decay`, `--checkpoint-every`), snapshot averaging, temperature and
cache-parameter sweeps, experiment planning and execution, debugging, and
drafting of this README and REPORT.md. All design choices were reviewed by
the author, every reported number was produced by the supplied pipeline on
the author's own hardware, and the author can explain each component.

## Data attribution

WikiText-2 (Merity et al., [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, text by Wikipedia
contributors under CC BY-SA 3.0 and GFDL 1.3. See the upstream
[dataset](https://huggingface.co/datasets/Salesforce/wikitext) for notices.
