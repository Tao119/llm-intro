"""
mixture_of_experts.py

Mixture-of-Experts Transformer LM (pure NumPy).

Replaces the FFN inside each TransformerBlock with a MoELayer that:
  - Routes tokens to the top-2 experts via a learned router.
  - Computes a weighted sum of the two expert outputs.
  - Adds a load-balancing loss (variance of expert selection counts).

Compares perplexity against a dense Transformer trained on the same corpus.
Saves expert-utilization bar chart to moe_utilization.png.
"""

import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup (reuse ch08 common utilities)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DL2      = os.path.join(_ROOT, "deep-learning-from-scratch-2")

sys.path.insert(0, _DL2)
sys.path.insert(0, os.path.join(_DL2, "ch08"))

from transformer_lm import (        # noqa: E402
    LayerNorm,
    CausalSelfAttention,
    positional_encoding,
    _softmax,
    _cross_entropy_seq,
    TransformerLM,
)
from common.util import preprocess, clip_grads   # noqa: E402
from common.optimizer import Adam                 # noqa: E402


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

TEXT = (
    "the dog ran . the cat sat . the dog sat . "
    "a cat ran . a dog ran . the cat ran . "
    "the dog ate . a cat ate . the cat ate ."
)


# ---------------------------------------------------------------------------
# GELU activation
# ---------------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def gelu_backward(x):
    c   = math.sqrt(2.0 / math.pi)
    t   = np.tanh(c * (x + 0.044715 * x**3))
    dt  = (1.0 - t**2) * c * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * dt


# ---------------------------------------------------------------------------
# Expert FFN with GELU
# ---------------------------------------------------------------------------

class ExpertFFN:
    """Single expert: Affine(d_model→d_ff) → GELU → Affine(d_ff→d_model)."""

    def __init__(self, d_model, d_ff, rng):
        scale = 1.0 / math.sqrt(d_model)
        self.W1 = (rng.normal(0, scale, (d_model, d_ff))).astype("f")
        self.b1 = np.zeros(d_ff, dtype="f")
        self.W2 = (rng.normal(0, 1.0 / math.sqrt(d_ff), (d_ff, d_model))).astype("f")
        self.b2 = np.zeros(d_model, dtype="f")
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.grads  = [np.zeros_like(p) for p in self.params]
        self.cache  = None

    def forward(self, x):
        """x: (n_tokens, d_model)"""
        pre  = x @ self.W1 + self.b1     # (n_tokens, d_ff)
        h    = gelu(pre)
        out  = h @ self.W2 + self.b2     # (n_tokens, d_model)
        self.cache = (x, pre, h)
        return out

    def backward(self, dout):
        """dout: (n_tokens, d_model)"""
        x, pre, h = self.cache
        dW2 = h.T @ dout
        db2 = dout.sum(axis=0)
        dh  = dout @ self.W2.T           # (n_tokens, d_ff)
        dpre = dh * gelu_backward(pre)
        dW1  = x.T @ dpre
        db1  = dpre.sum(axis=0)
        dx   = dpre @ self.W1.T
        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2
        return dx


# ---------------------------------------------------------------------------
# MoELayer
# ---------------------------------------------------------------------------

class MoELayer:
    """
    Mixture-of-Experts layer.

    Router: Affine(d_model → n_experts) → Softmax → top-2 selection.
    Forward: weighted_sum(top2_expert_outputs).
    Load-balancing loss: variance of expert selection counts (stored, not
    added automatically — the caller should include it in the total loss).
    """

    def __init__(self, d_model, d_ff, n_experts=4, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.d_model   = d_model
        self.d_ff      = d_ff
        self.n_experts = n_experts

        scale            = 1.0 / math.sqrt(d_model)
        self.router_W    = rng.normal(0, scale, (d_model, n_experts)).astype("f")
        self.router_b    = np.zeros(n_experts, dtype="f")
        self.experts     = [ExpertFFN(d_model, d_ff, rng) for _ in range(n_experts)]

        self.params = [self.router_W, self.router_b]
        self.grads  = [np.zeros_like(self.router_W), np.zeros_like(self.router_b)]
        for exp in self.experts:
            self.params += exp.params
            self.grads  += exp.grads

        # populated during forward, used by caller
        self.load_balance_loss = 0.0
        self.expert_counts     = np.zeros(n_experts, dtype=np.int32)

        self.cache = None

    def forward(self, x):
        """
        x: (N, T, d_model)
        Returns (N, T, d_model).
        """
        N, T, D = x.shape
        x_flat  = x.reshape(N * T, D)   # (M, d_model) where M = N*T
        M       = N * T

        # Router scores and weights
        router_logits  = x_flat @ self.router_W + self.router_b  # (M, E)
        router_weights = _softmax(router_logits)                  # (M, E)

        # Top-2 expert indices and their weights
        top2_idx     = np.argsort(router_weights, axis=-1)[:, -2:]  # (M, 2)
        top2_weights = np.take_along_axis(router_weights, top2_idx, axis=-1)
        # Renormalise top-2 weights
        top2_weights = top2_weights / (top2_weights.sum(axis=-1, keepdims=True) + 1e-9)

        # Load balancing: variance of selection counts
        counts = np.zeros(self.n_experts, dtype=np.int32)
        for e_idx in top2_idx.reshape(-1):
            counts[e_idx] += 1
        self.expert_counts += counts
        self.load_balance_loss = float(counts.var())

        # Each expert processes tokens routed to it
        expert_outputs = np.zeros((M, D), dtype="f")
        # store which tokens each expert handled and their weights
        expert_token_map = [[] for _ in range(self.n_experts)]  # expert_id → list of (token_idx, weight)
        for m in range(M):
            for k in range(2):
                eid    = int(top2_idx[m, k])
                weight = float(top2_weights[m, k])
                expert_token_map[eid].append((m, weight))

        # Forward through each expert
        expert_inputs  = {}
        expert_results = {}
        for eid, assignments in enumerate(expert_token_map):
            if not assignments:
                self.experts[eid].cache = None
                continue
            indices = [a[0] for a in assignments]
            inp     = x_flat[indices]                     # (k, d_model)
            out     = self.experts[eid].forward(inp)      # (k, d_model)
            expert_inputs[eid]  = (indices, inp)
            expert_results[eid] = out
            for local_i, (global_i, w) in enumerate(assignments):
                expert_outputs[global_i] += w * out[local_i]

        self.cache = (x_flat, router_weights, top2_idx, top2_weights,
                      expert_token_map, expert_inputs, expert_results, N, T)
        return expert_outputs.reshape(N, T, D)

    def backward(self, dout):
        """dout: (N, T, d_model)"""
        (x_flat, router_weights, top2_idx, top2_weights,
         expert_token_map, expert_inputs, expert_results, N, T) = self.cache
        M, D = x_flat.shape
        E    = self.n_experts

        dout_flat   = dout.reshape(M, D)     # (M, d_model)
        dx_flat     = np.zeros_like(x_flat)
        d_router_w  = np.zeros((M, E), dtype="f")  # gradient w.r.t. router weights

        for eid, assignments in enumerate(expert_token_map):
            if not assignments:
                continue
            indices     = [a[0] for a in assignments]
            weights     = np.array([a[1] for a in assignments], dtype="f")  # (k,)
            out_k       = expert_results[eid]                               # (k, D)

            # gradient of expert output w.r.t. expert weights
            # expert_outputs[m] += w * expert_out[local_i]
            # d(loss)/d(expert_out[local_i]) = w * dout_flat[global_i]
            d_expert_out = weights[:, np.newaxis] * dout_flat[indices]     # (k, D)
            d_expert_in  = self.experts[eid].backward(d_expert_out)        # (k, D)
            np.add.at(dx_flat, indices, d_expert_in)

            # gradient w.r.t. top-2 weight for each assignment
            # d(loss)/d(w_k_at_m) = dot(dout_flat[m], out_k[local_i])
            for local_i, (global_i, w) in enumerate(assignments):
                k_pos = np.where(top2_idx[global_i] == eid)[0]
                if len(k_pos) == 0:
                    continue
                # raw weight gradient (chain through renorm is approximated)
                d_router_w[global_i, eid] += float(
                    np.dot(dout_flat[global_i], out_k[local_i])
                )

        # Gradient through router softmax
        # d_router_logits = router_weights * (d_router_w - (d_router_w * router_weights).sum(-1, keepdims=True))
        d_router_logits = router_weights * (
            d_router_w - (d_router_w * router_weights).sum(axis=-1, keepdims=True)
        )

        d_router_W = x_flat.T @ d_router_logits          # (D, E)
        d_router_b = d_router_logits.sum(axis=0)          # (E,)
        dx_flat   += d_router_logits @ self.router_W.T   # (M, D)

        self.grads[0][...] = d_router_W
        self.grads[1][...] = d_router_b
        return dx_flat.reshape(N, T, D)


# ---------------------------------------------------------------------------
# MoETransformerBlock (FFN replaced by MoELayer)
# ---------------------------------------------------------------------------

class MoETransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, n_experts=4, rng=None):
        self.norm1 = LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.moe   = MoELayer(d_model, d_ff, n_experts=n_experts, rng=rng)

        self.params = (self.norm1.params + self.attn.params
                       + self.norm2.params + self.moe.params)
        self.grads  = (self.norm1.grads  + self.attn.grads
                       + self.norm2.grads  + self.moe.grads)
        self.cache  = None

    def forward(self, x):
        h   = x + self.attn.forward(self.norm1.forward(x))
        out = h + self.moe.forward(self.norm2.forward(h))
        self.cache = (x, h)
        return out

    def backward(self, dout):
        x, h = self.cache
        dh_moe = self.moe.backward(self.norm2.backward(dout))
        dh     = dout + dh_moe
        dx_attn = self.attn.backward(self.norm1.backward(dh))
        return dh + dx_attn

    @property
    def load_balance_loss(self):
        return self.moe.load_balance_loss


# ---------------------------------------------------------------------------
# MoETransformerLM
# ---------------------------------------------------------------------------

class MoETransformerLM:
    """Same interface as TransformerLM but with MoE blocks."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 n_experts=4, max_len=256, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.d_model    = d_model
        self.vocab_size = vocab_size

        self.embed_W = (rng.normal(0, 1.0 / math.sqrt(d_model),
                                   (vocab_size, d_model))).astype("f")
        self.pe      = positional_encoding(max_len, d_model)
        self.blocks  = [MoETransformerBlock(d_model, n_heads, d_ff,
                                            n_experts=n_experts, rng=rng)
                        for _ in range(n_layers)]
        self.head_b  = np.zeros(vocab_size, dtype="f")

        self.params = [self.embed_W, self.head_b]
        self.grads  = [np.zeros_like(self.embed_W), np.zeros_like(self.head_b)]
        for blk in self.blocks:
            self.params += blk.params
            self.grads  += blk.grads

        self.cache = None

    def forward(self, xs, ts, lb_coeff=0.01):
        N, T = xs.shape
        x = self.embed_W[xs] + self.pe[:T]
        self.emb_in = xs

        for blk in self.blocks:
            x = blk.forward(x)

        logits = x @ self.embed_W.T + self.head_b
        loss, probs = _cross_entropy_seq(logits, ts)

        # Add load-balancing loss
        lb_loss = sum(blk.load_balance_loss for blk in self.blocks)
        total_loss = loss + lb_coeff * lb_loss

        self.cache = (x, probs, ts)
        return total_loss

    def backward(self, dout=1):
        x, probs, ts = self.cache
        N, T, V = probs.shape

        dlogits = probs.copy()
        dlogits[np.arange(N)[:, None], np.arange(T)[None, :], ts] -= 1
        dlogits *= dout / (N * T)

        self.grads[1][...] = dlogits.sum(axis=(0, 1))
        dlogits_2d = dlogits.reshape(N * T, V)
        x_2d       = x.reshape(N * T, self.d_model)
        dW_head    = dlogits_2d.T @ x_2d
        dx         = (dlogits_2d @ self.embed_W).reshape(N, T, self.d_model)

        for blk in reversed(self.blocks):
            dx = blk.backward(dx)

        dW_emb = np.zeros_like(self.embed_W)
        np.add.at(dW_emb, self.emb_in.reshape(-1), dx.reshape(N * T, self.d_model))
        self.grads[0][...] = dW_emb + dW_head

    def expert_counts(self):
        total = np.zeros(self.blocks[0].moe.n_experts, dtype=np.int64)
        for blk in self.blocks:
            total += blk.moe.expert_counts
        return total


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def run_training(model, corpus, max_epoch=300, batch_size=4, time_size=5,
                 label=""):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))
    optimizer = Adam(lr=1e-3)
    time_idx  = 0
    ppl_last  = None

    for epoch in range(max_epoch):
        total_loss = total_count = 0
        for _ in range(max_iter):
            batch_xs = np.zeros((batch_size, time_size), dtype=np.int32)
            batch_ts = np.zeros((batch_size, time_size), dtype=np.int32)
            offsets  = [data_size * i // batch_size for i in range(batch_size)]
            for t in range(time_size):
                for b in range(batch_size):
                    batch_xs[b, t] = xs[(offsets[b] + time_idx) % data_size]
                    batch_ts[b, t] = ts[(offsets[b] + time_idx) % data_size]
            time_idx = (time_idx + time_size) % data_size

            loss = model.forward(batch_xs, batch_ts)
            model.backward()
            clip_grads(model.grads, 1.0)
            optimizer.update(model.params, model.grads)
            total_loss  += loss
            total_count += 1

        ppl = float(np.exp(min(total_loss / total_count, 10.0)))
        ppl_last = ppl
        if (epoch + 1) % 100 == 0:
            print(f"  [{label}] epoch {epoch+1:>4}/{max_epoch}  ppl={ppl:.2f}")

    return ppl_last


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Mixture-of-Experts Transformer LM")
    print("=" * 60)

    corpus, word_to_id, id_to_word = preprocess(TEXT)
    vocab_size = len(word_to_id)
    print(f"Corpus length: {len(corpus)}  vocab size: {vocab_size}")

    D_MODEL   = 32
    N_HEADS   = 4
    N_LAYERS  = 2
    D_FF      = 64
    N_EXPERTS = 4
    MAX_EPOCH = 300
    BATCH     = 4
    TSIZE     = 5

    # ── Dense baseline ────────────────────────────────────────────────────
    print("\nTraining dense Transformer ...")
    np.random.seed(42)
    dense_model = TransformerLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF)
    dense_ppl   = run_training(dense_model, corpus,
                               max_epoch=MAX_EPOCH, batch_size=BATCH,
                               time_size=TSIZE, label="Dense")

    # ── MoE model ─────────────────────────────────────────────────────────
    print("\nTraining MoE Transformer ...")
    moe_rng   = np.random.default_rng(42)
    moe_model = MoETransformerLM(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF,
                                  n_experts=N_EXPERTS, rng=moe_rng)
    moe_ppl   = run_training(moe_model, corpus,
                             max_epoch=MAX_EPOCH, batch_size=BATCH,
                             time_size=TSIZE, label="MoE")

    # ── Results ───────────────────────────────────────────────────────────
    print("\n=== Perplexity Comparison ===")
    print(f"  Dense Transformer : {dense_ppl:.2f}")
    print(f"  MoE   Transformer : {moe_ppl:.2f}")

    counts = moe_model.expert_counts()
    total  = counts.sum()
    print("\n=== Expert Utilization ===")
    for i, c in enumerate(counts):
        bar = "#" * int(30 * c / max(total, 1))
        print(f"  Expert {i}: {c:6d}  ({100*c/max(total,1):.1f}%)  {bar}")

    # Load-balancing: variance of expert fractions
    fractions  = counts / max(total, 1)
    lb_variance = float(fractions.var())
    print(f"\n  Load-balance variance: {lb_variance:.6f}  "
          f"(0 = perfectly balanced)")

    # Router entropy (how decisive is routing)
    # Approximated as entropy of the count distribution
    p = fractions / (fractions.sum() + 1e-9)
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    max_entropy = math.log(N_EXPERTS)
    print(f"  Router entropy: {entropy:.4f}  (max={max_entropy:.4f})")

    # ── Bar chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(range(N_EXPERTS), counts, color=colors[:N_EXPERTS])
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Token selections (cumulative across layers & steps)")
    ax.set_title("MoE Expert Utilization")
    ax.set_xticks(range(N_EXPERTS))
    ax.set_xticklabels([f"Expert {i}" for i in range(N_EXPERTS)])
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", va="bottom",
                fontsize=9)
    # Ideal (uniform) line
    ax.axhline(total / N_EXPERTS, color="red", linestyle="--",
               linewidth=1.2, label="ideal (uniform)")
    ax.legend()
    plt.tight_layout()
    out_png = os.path.join(_THIS_DIR, "moe_utilization.png")
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"\nExpert utilization chart saved → {out_png}")
    print("=" * 60)


if __name__ == "__main__":
    main()
