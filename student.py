"""Student language model: a scaled GPT with rotary positions, RMSNorm, SwiGLU,
no biases and depth-scaled residual initialization.

Differences from the classroom baseline (model.py):

1. Rotary position embeddings (RoPE, Su et al. 2021) replace learned absolute
   position embeddings.
2. RMSNorm (Zhang & Sennrich 2019) replaces LayerNorm in the pre-norm blocks.
3. SwiGLU (Shazeer 2020) replaces the GELU MLP.
4. Biases are removed and the two residual projections of every block are
   scaled by 1/sqrt(2*depth) at initialization so deeper stacks stay stable.
5. The output head is tied to the token embedding matrix.

Each upgrade is switchable through the configuration
(pos: "rope"/"learned", mlp: "swiglu"/"gelu", norm: "rms"/"ln"), which is how
the ablation experiments in the report are produced.

The default configuration (configs/student.json) is width=192, depth=6,
heads=6, mult=512: about 3.7M parameters and about 3.4x the baseline's
per-token FLOPs, inside the 5x evaluation-time budget. The evaluator calls
forward(ids) for unnormalized logits and predict_log_probs(ids) for normalized
finite log probabilities; both return [batch, time, 2048].
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BITS = 11  # vocab 2048 < 2^11; n-gram contexts are packed into int64 keys


def pack(tokens):
    """Pack a token sequence into one int64 key (11 bits per token)."""
    value = 0
    for i, t in enumerate(tokens):
        value |= int(t) << (BITS * i)
    return value


def rope_cos_sin(head_dim, positions, base=10000.0):
    """RoPE tables [1, 1, T, head_dim] for a head of dimension `head_dim`."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    theta = torch.einsum('i,j->ij', positions.float(), inv)
    emb = torch.cat((theta, theta), dim=-1)
    return emb.unsqueeze(0).unsqueeze(0)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = width = config['width']
        self.heads = config['heads']
        norm = RMSNorm if config.get('norm', 'rms') == 'rms' else nn.LayerNorm
        self.norm1, self.norm2 = norm(width), norm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        drop = config.get('dropout', 0.0)
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        if config.get('mlp', 'swiglu') == 'swiglu':
            self.mlp = 'swiglu'
            mult = config.get('mult', math.ceil(8 / 3 * width))
            self.gate = nn.Linear(width, mult, bias=False)
            self.up = nn.Linear(width, mult, bias=False)
            self.down = nn.Linear(mult, width, bias=False)
        else:
            self.mlp = 'gelu'
            mult = config.get('mult', math.ceil(8 / 3 * width))
            hidden = config.get('gelu_mult', int(1.5 * mult))  # FLOP-matched to SwiGLU
            self.fc1 = nn.Linear(width, hidden, bias=False)
            self.fc2 = nn.Linear(hidden, width, bias=False)

    def forward(self, x, cos=None, sin=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if cos is not None:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        if self.mlp == 'swiglu':
            hidden = F.silu(self.gate(self.norm2(x))) * self.up(self.norm2(x))
            return x + self.dropout(self.down(hidden))
        return x + self.dropout(self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.pos = config.get('pos', 'rope')
        self.token = nn.Embedding(config['vocab'], width)
        self.pos_emb = nn.Embedding(self.context, width) if self.pos == 'learned' else None
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['depth'])])
        self.norm = RMSNorm(width) if config.get('norm', 'rms') == 'rms' else nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        head_dim = width // config['heads']
        table = rope_cos_sin(head_dim, torch.arange(self.context))
        self.register_buffer('rope_cos', torch.cos(table), persistent=False)
        self.register_buffer('rope_sin', torch.sin(table), persistent=False)
        # Optional training-derived backoff n-gram asset (see build_ngram.py).
        # Buffers are created when config['ngram_n'] is present; they are
        # filled by load_state_dict from the checkpoint and used only in
        # predict_log_probs when config['ngram_lambda'] > 0.
        self._ngram = None
        if config.get('ngram_n'):
            self.register_buffer('ngram_ctx', torch.zeros(config['ngram_n'], dtype=torch.int64))
            self.register_buffer('ngram_next', torch.zeros(config['ngram_n'], dtype=torch.int16))
            self.register_buffer('ngram_cnt', torch.zeros(config['ngram_n'], dtype=torch.int32))
            self.register_buffer('ngram_offsets', torch.zeros(4, dtype=torch.int64))
            self.register_buffer('ngram_unigram', torch.zeros(config['vocab'], dtype=torch.int64))
            self.register_buffer('ngram_total', torch.tensor(0, dtype=torch.int64))
        self.apply(self.initialize)
        # Depth-scaled residual initialization: keeps hidden-state magnitude
        # stable through the stack for RMSNorm / no-bias GPTs.
        scale = (2 * config['depth']) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=0.02 * scale)
            if block.mlp == 'swiglu':
                nn.init.normal_(block.down.weight, std=0.02 * scale)
            else:
                nn.init.normal_(block.fc2.weight, std=0.02 * scale)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def features(self, ids):
        x = self.token(ids)
        if self.pos == 'learned':
            x = x + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))
        use_rope = self.pos == 'rope'
        cos = self.rope_cos[:, :, :ids.shape[1]] if use_rope else None
        sin = self.rope_sin[:, :, :ids.shape[1]] if use_rope else None
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def within_window_cache(self, hidden, ids):
        """Strictly causal flat cache over the current window (Grave et al. 2017).

        A prediction at position t blends the model distribution with an
        empirical distribution over tokens that followed SIMILAR hidden states
        earlier in the same window: keys are the hidden states at positions
        j < t, values the observed next tokens ids[j+1]. Similarity is cosine,
        passed through a softmax; the resulting distribution is interpolated
        with the model with weight cache_lambda. Positions t=0 keep the pure
        model distribution. Nothing is stored across calls, so every
        evaluation window starts with an empty cache.
        """
        lam = float(self.config.get('cache_lambda', 0.0))
        if lam <= 0.0:
            return None
        batch, length, width = hidden.shape
        vocab = self.config['vocab']
        beta = float(self.config.get('cache_beta', 1.0))
        h = F.normalize(hidden.float(), dim=-1)
        q, k = h[:, 1:], h[:, :-1]                    # [B, T-1, W]
        values = ids[:, 1:]                           # [B, T-1] observed next tokens
        sim = torch.bmm(q, k.transpose(1, 2)).div(beta)  # [B, T-1, T-1]
        keep = torch.tril(torch.ones(length - 1, length - 1, dtype=torch.bool, device=ids.device))
        sim = sim.masked_fill(~keep, float('-inf'))
        weights = sim.softmax(dim=-1).nan_to_num(0.0)  # rows are distributions over j <= t-1
        counts = torch.zeros(batch, length - 1, vocab, device=ids.device)
        counts.scatter_add_(2, values[:, None, :].expand(batch, length - 1, length - 1), weights)
        return counts                                  # [B, T-1, V], valid distributions

    def ngram_prior(self, ids):
        """Backoff n-gram prior for the current window (training-derived asset).

        For each position t, the context is the up-to-3 observed previous
        tokens of the same window (strictly causal). The prior is a cascade of
        MLE estimates with pseudo-count interpolation: order 4 blends toward
        order 3 (alpha 32), order 3 toward order 2 (alpha 16), order 2 toward
        a Laplace unigram (alpha 8). Missing higher orders fall through
        unchanged. The returned tensor is a valid distribution per position.
        """
        lam = float(self.config.get('ngram_lambda', 0.0))
        if lam <= 0.0:
            return None
        batch, length = ids.shape
        vocab = self.config['vocab']
        if self._ngram is None:
            offsets = self.ngram_offsets.cpu().numpy()
            self._ngram = {
                'ctx': [self.ngram_ctx.cpu().numpy()[offsets[o - 1]:offsets[o]] for o in (1, 2, 3)],
                'next': [self.ngram_next.cpu().numpy()[offsets[o - 1]:offsets[o]] for o in (1, 2, 3)],
                'cnt': [self.ngram_cnt.cpu().numpy()[offsets[o - 1]:offsets[o]] for o in (1, 2, 3)],
                'uni': self.ngram_unigram.cpu().numpy(),
                'total': int(self.ngram_total.item()),
                'alpha': {1: 8.0, 2: 16.0, 3: 32.0},
            }
        lookup = self._ngram
        p1 = (lookup['uni'] + 1.0) / (lookup['total'] + vocab)  # Laplace unigram [V]
        dense_w = np.zeros(batch * length, dtype=np.float64)
        idx, idv, vals = [], [], []
        rows = ids.cpu().numpy()
        for i in range(batch):
            row = rows[i]
            for t in range(length):
                context = row[max(0, t - 3):t]
                a_prod = 1.0
                for o in range(len(context), 0, -1):
                    key = pack(context[len(context) - o:])
                    arr = lookup['ctx'][o - 1]
                    lo = np.searchsorted(arr, key, side='left')
                    hi = np.searchsorted(arr, key, side='right')
                    if lo < hi:
                        nxt = lookup['next'][o - 1][lo:hi]
                        cnt = lookup['cnt'][o - 1][lo:hi].astype(np.float64)
                        total_c = cnt.sum()
                        a = lookup['alpha'][o] / (total_c + lookup['alpha'][o])
                        w = a_prod / (total_c + lookup['alpha'][o])
                        idx.extend([i * length + t] * len(nxt))
                        idv.extend(nxt.tolist())
                        vals.extend((w * cnt).tolist())
                        a_prod *= a
                dense_w[i * length + t] = a_prod
        prior = torch.zeros(batch * length, vocab, device=ids.device)
        if idx:
            prior.index_put_(
                (torch.tensor(idx, device=ids.device), torch.tensor(idv, device=ids.device)),
                torch.tensor(vals, device=ids.device, dtype=torch.float32), accumulate=True)
        prior += torch.tensor(dense_w, device=ids.device, dtype=torch.float32)[:, None] * torch.tensor(
            p1, device=ids.device, dtype=torch.float32)[None, :]
        return prior.view(batch, length, vocab)

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized finite log probabilities.

        Optional config keys: "temperature" rescales the logits before the
        softmax; "cache_lambda" > 0 enables the within-window flat cache
        (Grave et al. 2017); "ngram_lambda" > 0 interpolates the static
        backoff n-gram prior built from training text (build_ngram.py). All
        three settings are tuned on validation only; the returned
        distribution stays normalized.
        """
        tau = float(self.config.get('temperature', 1.0))
        hidden = self.features(ids)
        logp = F.log_softmax(self.head(hidden).float() / tau, dim=-1)
        p = logp.exp()
        counts = self.within_window_cache(hidden, ids)
        if counts is not None:
            # Position 0 has no earlier keys, so it keeps the pure model
            # distribution; positions t >= 1 blend with the flat cache.
            lam = torch.full((logp.shape[1],), float(self.config['cache_lambda']), device=ids.device)
            lam[0] = 0.0
            cache = torch.cat(
                (torch.zeros(logp.shape[0], 1, logp.shape[2], device=ids.device), counts), dim=1)
            p = (1.0 - lam[None, :, None]) * p + lam[None, :, None] * cache
        ngram = self.ngram_prior(ids)
        if ngram is not None:
            p = (1.0 - float(self.config['ngram_lambda'])) * p + float(self.config['ngram_lambda']) * ngram
        return torch.log(p)


def build_model(config):
    return GPT(config)
