"""
speculative_decoding.py

Speculative Decoding demo (Leviathan et al. 2023).

Uses the pure-NumPy TransformerLM from
  deep-learning-from-scratch-2/ch08/transformer_lm.py

Two models trained on the same small corpus:
  - Draft  : d_model=16, n_layers=1, n_heads=2  (fast, small)
  - Target : d_model=64, n_layers=4, n_heads=4  (slow, accurate)

Algorithm:
  1. Draft generates K tokens speculatively.
  2. Target evaluates all K+1 positions in one forward pass.
  3. Accept token i with prob  min(1, p_target[i] / p_draft[i]).
  4. On rejection: sample from corrected distribution, stop.
  5. If all accepted, append one bonus token from target.

Results saved to speculative_results.json.
"""

import os
import sys
import json
import time
import math

import numpy as np

# ---------------------------------------------------------------------------
# Path: make ch08/transformer_lm.py importable
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # imai-lab/
_DL2      = os.path.join(_ROOT, "deep-learning-from-scratch-2")

sys.path.insert(0, _DL2)           # for "common.*"
sys.path.insert(0, os.path.join(_DL2, "ch08"))  # for "transformer_lm"

from transformer_lm import (        # noqa: E402
    TransformerLM,
    _softmax,
    positional_encoding,
)
from common.util import preprocess, clip_grads   # noqa: E402
from common.optimizer import Adam                 # noqa: E402


# ---------------------------------------------------------------------------
# Corpus — identical to ch08/transformer_lm.py __main__
# ---------------------------------------------------------------------------

TEXT = (
    "the dog ran . the cat sat . the dog sat . "
    "a cat ran . a dog ran . the cat ran . "
    "the dog ate . a cat ate . the cat ate ."
)

SEED = 42
K    = 4       # draft tokens per step
MAX_NEW = 50   # tokens to generate for benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_token_probs(model, token_ids):
    """Run forward pass and return softmax probs for the next position."""
    xs = np.array(token_ids, dtype=np.int32)[np.newaxis]   # (1, T)
    ts = np.zeros_like(xs)                                   # dummy targets
    model.forward(xs, ts)                                    # fills model.cache
    x, probs, _ = model.cache
    # x is the last hidden state (1, T, d_model), probs is (1, T, V)
    # We want logits at position -1; recompute from x
    logits_last = x[0, -1] @ model.embed_W.T + model.head_b  # (V,)
    p = _softmax(logits_last[np.newaxis])[0]
    return p


def _target_batch_probs(model, token_ids):
    """
    Run one forward pass covering all positions in token_ids.
    Returns probability matrix of shape (T, V) where row t is the
    distribution over the next token given token_ids[:t+1].
    """
    T = len(token_ids)
    xs = np.array(token_ids, dtype=np.int32)[np.newaxis]   # (1, T)
    ts = np.zeros_like(xs)
    model.forward(xs, ts)
    x, _, _ = model.cache                                   # x: (1, T, d_model)
    logits = x[0] @ model.embed_W.T + model.head_b          # (T, V)
    probs  = _softmax(logits)                               # (T, V)
    return probs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(corpus, vocab_size, d_model, n_heads, n_layers, d_ff,
          max_epoch=300, batch_size=4, time_size=5, label=""):
    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    max_iter  = max(1, data_size // (batch_size * time_size))

    model     = TransformerLM(vocab_size, d_model, n_heads, n_layers, d_ff)
    optimizer = Adam(lr=1e-3)

    time_idx = 0
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

        if (epoch + 1) % 100 == 0:
            ppl = float(np.exp(total_loss / total_count))
            print(f"  [{label}] epoch {epoch+1:>4}/{max_epoch}  ppl={ppl:.2f}")

    return model


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------

def speculative_decode(draft_model, target_model, start_id, K=4, max_len=20, rng=None):
    """
    Leviathan et al. (2023) Algorithm 1.

    Args:
        draft_model  : small fast model
        target_model : large accurate model
        start_id     : starting token id
        K            : draft tokens per step
        max_len      : total tokens to generate (not counting start)
        rng          : np.random.Generator

    Returns:
        tokens          : list of generated token ids (excluding start)
        n_accepted_total: total draft tokens accepted across all rounds
        n_drafted_total : total draft tokens generated across all rounds
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    generated     = [start_id]
    n_accepted_total = 0
    n_drafted_total  = 0

    while len(generated) - 1 < max_len:
        # ── Step 1: Draft K tokens ─────────────────────────────────────────
        draft_tokens = []
        draft_probs  = []
        ctx = list(generated)
        for _ in range(K):
            p_d = _next_token_probs(draft_model, ctx)
            tok = int(rng.choice(len(p_d), p=p_d))
            draft_tokens.append(tok)
            draft_probs.append(float(p_d[tok]))
            ctx.append(tok)

        n_drafted_total += len(draft_tokens)

        # ── Step 2: Target evaluates all K+1 positions in one pass ────────
        target_input = generated + draft_tokens          # length T + K
        t_probs = _target_batch_probs(target_model, target_input)
        # t_probs[i] is the distribution for position i+1,
        # i.e. given input[0..i] → next token distribution

        # ── Step 3–4: Accept / reject each draft token ───────────────────
        base = len(generated) - 1  # index into t_probs for first draft pos

        accepted_this_round = 0
        for i, (tok, p_d_tok) in enumerate(zip(draft_tokens, draft_probs)):
            pos      = base + i          # t_probs index predicting tok
            p_t      = t_probs[pos]      # (V,) target dist
            p_t_tok  = float(p_t[tok])

            ratio        = min(1.0, p_t_tok / (p_d_tok + 1e-12))
            if rng.random() < ratio:
                generated.append(tok)
                n_accepted_total += 1
                accepted_this_round += 1
            else:
                # Corrected distribution and sample bonus token, then stop
                p_draft_here = _next_token_probs(draft_model, generated)
                p_corr = np.maximum(p_t - p_draft_here, 0.0)
                z = p_corr.sum()
                if z > 1e-9:
                    p_corr /= z
                else:
                    p_corr = p_t
                bonus = int(rng.choice(len(p_corr), p=p_corr))
                generated.append(bonus)
                break
        else:
            # All K accepted: sample one bonus token from target
            bonus_pos = base + len(draft_tokens)
            if bonus_pos < len(t_probs):
                p_bonus = t_probs[bonus_pos]
                bonus   = int(rng.choice(len(p_bonus), p=p_bonus))
                generated.append(bonus)

    # Trim to exactly max_len new tokens
    return generated[1: 1 + max_len], n_accepted_total, n_drafted_total


def naive_decode(target_model, start_id, max_len=20, rng=None):
    """Standard auto-regressive decoding with target model."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    generated = [start_id]
    for _ in range(max_len):
        p = _next_token_probs(target_model, generated)
        tok = int(rng.choice(len(p), p=p))
        generated.append(tok)
    return generated[1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Speculative Decoding  (Leviathan et al. 2023)")
    print("=" * 60)

    corpus, word_to_id, id_to_word = preprocess(TEXT)
    vocab_size = len(word_to_id)
    print(f"Corpus length: {len(corpus)}  vocab size: {vocab_size}")
    print(f"Vocabulary: {list(word_to_id.keys())}")

    # ── Train draft model ──────────────────────────────────────────────────
    print("\nTraining draft model  (d_model=16, n_layers=1, n_heads=2) ...")
    draft_model = train(
        corpus, vocab_size,
        d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_epoch=300, label="Draft"
    )

    # ── Train target model ─────────────────────────────────────────────────
    print("\nTraining target model (d_model=64, n_layers=4, n_heads=4) ...")
    target_model = train(
        corpus, vocab_size,
        d_model=64, n_heads=4, n_layers=4, d_ff=128,
        max_epoch=300, label="Target"
    )

    # ── Benchmark setup ────────────────────────────────────────────────────
    start_word = "the"
    start_id   = word_to_id[start_word]
    rng        = np.random.default_rng(SEED)
    N_RUNS     = 5

    print(f"\nBenchmark: start='{start_word}'  max_new={MAX_NEW}  K={K}  "
          f"runs={N_RUNS}")

    # ── Naive AR timing ────────────────────────────────────────────────────
    naive_times = []
    naive_out   = None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        naive_out = naive_decode(target_model, start_id,
                                 max_len=MAX_NEW, rng=rng)
        t1 = time.perf_counter()
        naive_times.append(t1 - t0)
    naive_elapsed = float(np.mean(naive_times))
    naive_tps     = MAX_NEW / naive_elapsed

    naive_words = " ".join(id_to_word[i] for i in naive_out)
    print(f"\nNaive AR   : {naive_tps:7.2f} tok/s   "
          f"elapsed={naive_elapsed*1000:.1f} ms")
    print(f"  sample : {naive_words}")

    # ── Speculative timing ─────────────────────────────────────────────────
    spec_times    = []
    all_accepted  = []
    all_drafted   = []
    spec_out      = None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        spec_out, n_acc, n_draft = speculative_decode(
            draft_model, target_model, start_id,
            K=K, max_len=MAX_NEW, rng=rng
        )
        t1 = time.perf_counter()
        spec_times.append(t1 - t0)
        all_accepted.append(n_acc)
        all_drafted.append(n_draft)

    spec_elapsed    = float(np.mean(spec_times))
    spec_tps        = MAX_NEW / spec_elapsed
    acceptance_rate = float(np.mean(all_accepted)) / max(float(np.mean(all_drafted)), 1)

    spec_words = " ".join(id_to_word[i] for i in spec_out)
    print(f"Speculative: {spec_tps:7.2f} tok/s   "
          f"elapsed={spec_elapsed*1000:.1f} ms")
    print(f"  sample : {spec_words}")
    print(f"  accept : {acceptance_rate:.2%}  "
          f"avg acc/drafted = {np.mean(all_accepted):.1f} / "
          f"{np.mean(all_drafted):.1f}")

    speedup = spec_tps / max(naive_tps, 1e-9)
    print(f"\nSpeedup: {speedup:.3f}x")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "algorithm": "Speculative Decoding (Leviathan et al. 2023)",
        "corpus": TEXT,
        "K": K,
        "max_new_tokens": MAX_NEW,
        "n_benchmark_runs": N_RUNS,
        "start_token": start_word,
        "draft_model": {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32},
        "target_model": {"d_model": 64, "n_layers": 4, "n_heads": 4, "d_ff": 128},
        "naive_ar": {
            "tokens_per_sec": round(naive_tps, 3),
            "elapsed_ms": round(naive_elapsed * 1000, 2),
            "sample": naive_words,
        },
        "speculative": {
            "tokens_per_sec": round(spec_tps, 3),
            "elapsed_ms": round(spec_elapsed * 1000, 2),
            "acceptance_rate": round(acceptance_rate, 4),
            "avg_accepted_per_run": round(float(np.mean(all_accepted)), 2),
            "avg_drafted_per_run": round(float(np.mean(all_drafted)), 2),
            "sample": spec_words,
        },
        "speedup_ratio": round(speedup, 3),
    }

    out_path = os.path.join(_THIS_DIR, "speculative_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
