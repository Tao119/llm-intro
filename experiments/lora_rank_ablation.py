"""
lora_rank_ablation.py

LoRA rank ablation study on cl-tohoku/bert-base-japanese-v3.
Fine-tunes on synthetic MARC-ja style sentiment data and records,
for each rank r in [2, 4, 8, 16, 32]:
  - Number of trainable parameters
  - Validation accuracy after 2 epochs

Saves a summary table and a PNG plot.
"""

import os
import json
import random
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
LORA_RANKS = [2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Synthetic dataset (MARC-ja style)
# ---------------------------------------------------------------------------

POSITIVE = [
    "この映画は感動的で素晴らしかった",
    "演技が上手く涙が出た",
    "ストーリーが深く心に刺さった",
    "映像美が圧倒的で見惚れた",
    "音楽が素晴らしく気持ちが高まった",
    "キャストの演技力が見事だった",
    "脚本が巧みで伏線の回収が見事",
    "感情移入できて最後まで楽しめた",
    "監督のセンスが光る傑作",
    "久しぶりに映画館で泣いた",
    "主人公の成長が感動的だった",
    "テンポが良く飽きずに見られた",
    "予想外の展開に驚かされた",
    "エンディングが美しく余韻が残った",
    "全キャラクターに愛着が湧いた",
    "何度でも見返したい名作",
    "細部まで丁寧に作られた作品",
    "友達にも強くお薦めしたい",
    "公開初日に見て良かった",
    "人生観が変わるような映画",
    "良い製品です",
    "最高でした",
    "とても満足",
    "おすすめします",
    "品質が良い",
    "素晴らしいサービス",
    "また買います",
    "期待以上でした",
    "コスパ最高",
    "大満足",
    "最高のショッピング体験でした",
    "親切なスタッフで助かりました",
    "商品の品質が期待を超えていた",
    "すぐに届いて嬉しかった",
    "また利用したいと思います",
    "丁寧な梱包で感謝します",
    "説明通りの商品で満足",
    "値段以上の価値がありました",
    "家族みんな喜んでいます",
    "プレゼントにも最適でした",
]

NEGATIVE = [
    "退屈でつまらない映画",
    "ストーリーが弱く失望した",
    "期待外れで時間の無駄",
    "主人公の行動が理解できない",
    "脚本に矛盾が多すぎる",
    "演技が棒読みで感情移入できない",
    "終わり方が意味不明だった",
    "テンポが遅く眠くなった",
    "キャラクターに魅力がなかった",
    "映像が安っぽく見るに堪えない",
    "最悪でした",
    "使えません",
    "がっかりした",
    "返品したい",
    "品質が悪い",
    "二度と買わない",
    "時間の無駄",
    "全く役立たない",
    "最低品質",
    "お金の無駄",
    "届いた商品が壊れていた",
    "説明と全く違う商品でした",
    "対応が最悪で腹が立った",
    "返品手続きが面倒だった",
    "全く使えない代物",
    "サポートが全然繋がらない",
    "梱包がひどく商品が傷ついていた",
    "偽物を送られた",
    "品質管理ができていない",
    "二度と利用しない",
]


def make_dataset(n_per_class=100):
    pos = (POSITIVE * ((n_per_class // len(POSITIVE)) + 1))[:n_per_class]
    neg = (NEGATIVE * ((n_per_class // len(NEGATIVE)) + 1))[:n_per_class]
    texts  = pos + neg
    labels = [1] * n_per_class + [0] * n_per_class
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    split = int(0.8 * len(texts))
    train = {"text": list(texts[:split]), "label": list(labels[:split])}
    val   = {"text": list(texts[split:]), "label": list(labels[split:])}
    return train, val


# ---------------------------------------------------------------------------
# LoRA rank ablation
# ---------------------------------------------------------------------------

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_rank_experiment(rank, train_data, val_data, tokenizer, device):
    import torch
    from transformers import AutoModelForSequenceClassification
    from peft import get_peft_model, LoraConfig, TaskType

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    model.to(device)
    n_trainable = count_trainable_params(model)

    # Simple manual training loop (no Trainer dependency issues)
    from torch.optim import AdamW

    def tokenize(batch_texts):
        enc = tokenizer(
            batch_texts,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    batch_size = 16
    n_epochs   = 2

    for epoch in range(n_epochs):
        model.train()
        idx = list(range(len(train_data["text"])))
        random.shuffle(idx)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            batch_texts  = [train_data["text"][i] for i in batch_idx]
            batch_labels = torch.tensor([train_data["label"][i] for i in batch_idx],
                                        dtype=torch.long, device=device)
            enc = tokenize(batch_texts)
            out = model(**enc, labels=batch_labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    # Evaluate
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for start in range(0, len(val_data["text"]), batch_size):
            batch_texts  = val_data["text"][start:start + batch_size]
            batch_labels = [val_data["label"][i] for i in range(start, min(start + batch_size, len(val_data["label"])))]
            enc = tokenize(batch_texts)
            out = model(**enc)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            correct += sum(p == l for p, l in zip(preds, batch_labels))
            total   += len(batch_labels)

    val_acc = correct / max(total, 1)
    return n_trainable, val_acc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(records, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranks  = [r["rank"] for r in records]
        accs   = [r["val_accuracy"] for r in records]
        params = [r["trainable_params"] for r in records]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(ranks, accs, "o-", color="steelblue", linewidth=2, markersize=8)
        ax1.set_xlabel("LoRA rank (r)")
        ax1.set_ylabel("Validation accuracy")
        ax1.set_title("LoRA rank vs Validation accuracy")
        ax1.set_xticks(ranks)
        ax1.grid(True, alpha=0.4)
        for r, a in zip(ranks, accs):
            ax1.annotate(f"{a:.3f}", (r, a), textcoords="offset points", xytext=(0, 8), ha="center")

        ax2.plot(ranks, params, "s-", color="coral", linewidth=2, markersize=8)
        ax2.set_xlabel("LoRA rank (r)")
        ax2.set_ylabel("Trainable parameters")
        ax2.set_title("LoRA rank vs Trainable parameters")
        ax2.set_xticks(ranks)
        ax2.grid(True, alpha=0.4)
        for r, p in zip(ranks, params):
            ax2.annotate(f"{p:,}", (r, p), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nPlot saved: {out_path}")
    except Exception as e:
        print(f"Plot skipped ({e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    random.seed(42)
    np.random.seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load tokenizer once
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Tokenizer loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"Failed to load tokenizer ({e}), aborting.")
        return

    train_data, val_data = make_dataset(n_per_class=100)
    print(f"Train: {len(train_data['text'])}  Val: {len(val_data['text'])}\n")

    records = []
    for rank in LORA_RANKS:
        print(f"--- LoRA rank r={rank} ---")
        try:
            n_params, val_acc = run_rank_experiment(rank, train_data, val_data, tokenizer, device)
        except Exception as e:
            print(f"  Experiment failed: {e}")
            n_params, val_acc = 0, 0.0
        records.append({"rank": rank, "trainable_params": n_params, "val_accuracy": val_acc})
        print(f"  Trainable params: {n_params:,}")
        print(f"  Val accuracy:     {val_acc:.4f}\n")

    # Print summary table
    print("=" * 50)
    print(f"{'rank':>6}  {'trainable_params':>18}  {'accuracy':>10}")
    print("-" * 40)
    for r in records:
        print(f"{r['rank']:>6}  {r['trainable_params']:>18,}  {r['val_accuracy']:>10.4f}")
    print("=" * 50)

    # Plot
    plot_path = os.path.join(OUT_DIR, "lora_rank_ablation.png")
    plot_results(records, plot_path)

    # Save JSON
    out_path = os.path.join(OUT_DIR, "lora_rank_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "epochs": 2, "ranks": records}, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
