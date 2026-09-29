# MP1 Report — A Scaled Modernized GPT with Within-Window Memory

Student: `TODO` · DASE7506 · 30 September 2026

## 1. Method

### 1.1 Architecture

The student model is a causal GPT (context 256, vocab 2048, tied embeddings)
that scales and modernizes the classroom baseline. Four architectural
changes are applied, each individually switchable through the configuration
(`pos`, `mlp`, `norm`, and the bias/init convention):

1. **RoPE** replaces learned absolute position embeddings.
2. **RMSNorm** replaces LayerNorm in both pre-norm positions.
3. **SwiGLU** replaces the GELU MLP.
4. **No biases + depth-scaled residual initialization** (each block's two
   residual projections initialized at std 0.02/√(2·depth)).

The final model is width 208, depth 6, heads 4 (head dim 52), SwiGLU mult
555: 3.54M parameters and ~4.0× the baseline's per-token FLOPs (a smaller
width-192/heads-6 variant is used for the paired comparisons and ablations).
Head dims that are not multiples of 4–8 keep `scaled_dot_product_attention`
on slow code paths on the training GPU: width-208/heads-8 (head dim 26) ran
~3× slower per step for only 1.17× more FLOPs than width-192/heads-6, which
ruled that variant out.

### 1.2 Training recipe

AdamW (β1=0.9, β2=0.999), gradient clipping at 1.0, weight decay 0.1 (0.2 for
the final run, with dropout 0.05 on both residual branches), BF16 autocast on
an RTX 3060 Laptop GPU. Batch 128 sequences × 256 targets; LR 1e-3 with
100-step linear warmup and cosine decay to 10%. Checkpoints are snapshotted
every 200–400 steps and **uniformly averaged** over a window selected on
validation. All runs use the supplied training text only; seed 17
(comparison runs) or 33 (final).

### 1.3 Inference-time components (tuned on validation only)

- **Temperature** τ rescales logits before the softmax (τ=1.12).
- **Within-window flat cache** (Grave et al. 2017): at each position t the
  model distribution is interpolated with an empirical distribution over
  tokens that followed cosine-similar hidden states earlier in the SAME
  window (keys j<t, values ids[j+1]). Strictly causal; the cache is rebuilt
  from scratch on every call, so windows, examples and scoring passes never
  share state. Interpolation weight λ=0.07, sharpness β=0.08.

## 2. Experiments

All BPB values are FP32 evaluations with the supplied scorer (validation for
selection, test for reporting). Training used one laptop GPU.

| Experiment | Params | Train tokens | val BPB | test BPB |
|---|---|---|---|---|
| Baseline (official recipe, seed 17) | 1.09M | 9.83M | 2.073 | 2.103 |
| Scale-only ablation (192×6, upgrades off) | 3.05M | 9.83M | 1.955 | 1.984 |
| Student 192×6 (upgrades on) | 3.05M | 9.83M | 1.735 | 1.763 |
| Student 192×6, long run (seed 33) | 3.05M | 52.4M | 1.671 | — |
| + snapshot averaging (600–1600) | 3.05M | 52.4M | 1.655 | — |
| + window cache (τ=1.05, λ=0.09, β=0.08) | 3.05M | 52.4M | 1.598 | — |
| Student 208×4, long run (wd 0.2, dropout 0.05) | 3.54M | 131.1M | 1.617 | — |
| + snapshot averaging (1200–3200) | 3.54M | 131.1M | 1.594 | — |
| + temperature τ=1.12 + window cache (λ=0.07, β=0.08) | 3.54M | 131.1M | **1.539** | **1.556** |

The same-token comparison (row 1 vs 3) isolates the architecture changes:
−0.34 BPB at identical data. The mechanism ablation (row 5 vs 6/7) isolates
snapshot averaging (−0.02) and the window cache (−0.05). Additional ablations:

- **Length scan** (192×6, batch 128): validation BPB reaches a minimum of
  1.685 at 39M tokens (1200 steps) and overfits steeply afterward (1.796 at
  131M tokens). The 208×4 model with weight decay 0.2 and dropout 0.05
  overfits much later (minimum 1.617 at 105M tokens, 3200 steps), showing
  regularization buys usable training length at fixed data.
- **Cross-seed weight averaging** (mean of the seed-17 and seed-33 averaged
  checkpoints) collapses to 3.6 BPB: the two solutions are not in a
  permutation-aligned basin, so naive weight-space averaging destroys the
  model. Checkpoint averaging is only applied within a single training run.
- **Static backoff n-gram prior** (order-4 with pseudo-count backoff, built
  from training text): even at λ=0.02 it raises validation BPB (1.605 vs
  1.598). The transformer already encodes global n-gram statistics, so the
  static prior adds redundant mass; the window cache instead captures the
  current document's local statistics, which is the complementary signal.

## 3. Critical analysis

**Why the components help.** RoPE, RMSNorm and SwiGLU are well-established
substitutions that improve parameter efficiency in small GPTs; the
depth-scaled initialization keeps the deeper, wider stack stable early in
training. Their combined effect (−0.22 BPB at fixed tokens) exceeds the pure
scale-up effect (−0.12), showing the gains are architectural, not just
capacity. The within-window cache is the single largest inference-time
component (−0.05 BPB): WikiText documents repeat names, dates and phrasings
within a few hundred tokens, and a 3M-parameter model cannot retain such
local statistics in its weights; a flat cache over the window's own hidden
states supplies exactly that information. It only works with a sharp
similarity temperature (β≈0.1): at β=1 the retrieved distribution is too
diffuse and contributes almost nothing (−0.004). Temperature τ>1 partially
corrects the overconfident outputs of a model trained on heavily repeated
data.

**What fails and why.** Two plausible additions did not work. (1) A static
backoff n-gram prior from the training text raised BPB at every interpolation
weight: the transformer already encodes global n-gram statistics, so the
prior contributes redundant mass and dilutes the model. The complementary
signal is local (within-window), which is precisely what the flat cache
captures. (2) Cross-seed weight averaging collapsed the model (3.6 BPB): two
independently trained networks occupy different, non-permutation-aligned
basins in weight space, so naive averaging destroys both. Within-run
snapshot averaging is safe because consecutive checkpoints stay in the same
basin.

**Costs.** All training ran on one laptop GPU (RTX 3060); the full search,
including the discarded experiments, took about 1.5 GPU-hours (disclosed in
RUN_LOG.csv). Inference cost of the final predictor: 17.1 s CPU FP32 on the
full test split versus 5.86 s for the baseline on the same idle machine
(ratio 2.9×, limit 5×); peak RSS 1097 MiB (limit 4 GiB); inference assets
~15 MiB (limit 64 MiB). The flat cache adds a few percent to scoring time;
its O(T²·W) cost is small at T=256. CPU timings vary with machine load
(laptop power/thermal coupling with the GPU); all limit checks use
same-session, same-machine comparisons.

**Limitations.** Evaluation windows are independent 256-token slices, so the
method can exploit locality only within a window; cross-window statistics
are off-limits by design. Training is data-limited: the curve overfits past
~40M targets and the recipe compensates with early stopping, weight decay,
dropout and snapshot averaging rather than more data. The validation–test
BPB gap (+0.028) is consistent across models and is plausibly a property of
the split composition.

## 4. References

1. J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.
2. B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," NeurIPS, 2019.
3. N. Shazeer, "GLU Variants Improve Transformer," arXiv:2002.05202, 2020.
4. E. Grave, A. Joulin, N. Usunier, "Improving Neural Language Models with a Continuous Cache," ICLR, 2017.
5. S. Merity et al., "Pointer Sentinel Mixture Models," arXiv:1609.07843, 2016.
6. R. Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time," ICML, 2022.
7. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.

## 5. AI assistance

Substantial AI assistance (Claude Code) was used for implementation,
experiment orchestration and drafting; see the repository README for the
full disclosure. All experiments were executed locally with the supplied
pipeline, and the author verifies every reported number.
