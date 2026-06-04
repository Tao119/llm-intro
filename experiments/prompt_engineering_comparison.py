"""
prompt_engineering_comparison.py

Systematic Prompt Engineering Study — deterministic simulation (no LLM required).

Compares 5 strategies on 20 medical QA test cases:
  1. Zero-shot
  2. Role prompting
  3. Chain-of-thought
  4. Few-shot (3 examples)
  5. Self-consistency (majority vote, N=5)

Metrics: exact_match, token_f1, rouge_l
Saves radar chart as PNG and results as JSON.
"""

import os
import json
import re
import numpy as np
import sys

OUT_DIR = os.path.join(os.path.dirname(__file__), "03-prompt-engineering")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# 20 medical QA test cases with expected answers
# ---------------------------------------------------------------------------

QA_CASES = [
    {
        "question": "高血圧の第一選択薬は何ですか？",
        "answer":   "カルシウム拮抗薬またはARB（アンジオテンシンII受容体拮抗薬）",
        "keywords": ["カルシウム", "ARB", "降圧薬"],
        "category": "薬理",
    },
    {
        "question": "糖尿病の診断基準における空腹時血糖値は？",
        "answer":   "空腹時血糖126mg/dL以上",
        "keywords": ["126", "空腹時", "血糖"],
        "category": "診断",
    },
    {
        "question": "心筋梗塞の典型的な症状を3つ挙げてください。",
        "answer":   "胸痛、冷汗、左肩への放散痛",
        "keywords": ["胸痛", "冷汗", "放散痛"],
        "category": "症状",
    },
    {
        "question": "敗血症のqSOFAスコアの構成要素は？",
        "answer":   "呼吸数≥22/分、意識変容、収縮期血圧≤100mmHg",
        "keywords": ["呼吸数", "意識", "血圧", "qSOFA"],
        "category": "スコア",
    },
    {
        "question": "アナフィラキシーショックの第一選択治療は？",
        "answer":   "エピネフリン（アドレナリン）筋注0.3mg",
        "keywords": ["エピネフリン", "アドレナリン", "筋注"],
        "category": "治療",
    },
    {
        "question": "CRP上昇が示す主な病態は何ですか？",
        "answer":   "炎症または感染症の存在を示す急性相タンパクの上昇",
        "keywords": ["炎症", "感染", "急性相"],
        "category": "検査",
    },
    {
        "question": "肺炎球菌肺炎の抗菌薬第一選択は？",
        "answer":   "アモキシシリンまたはペニシリンG",
        "keywords": ["アモキシシリン", "ペニシリン"],
        "category": "治療",
    },
    {
        "question": "INR延長が示す凝固因子の異常はどれか？",
        "answer":   "外因系凝固因子（第VII因子）または共通経路の異常",
        "keywords": ["外因系", "第VII因子", "凝固"],
        "category": "検査",
    },
    {
        "question": "STEMI（ST上昇型心筋梗塞）の治療目標時間は？",
        "answer":   "Door-to-Balloon時間90分以内",
        "keywords": ["90分", "Balloon", "Door"],
        "category": "治療",
    },
    {
        "question": "脳卒中のtPA投与の時間窓は？",
        "answer":   "症状出現から4.5時間以内",
        "keywords": ["4.5時間", "tPA", "時間窓"],
        "category": "治療",
    },
    {
        "question": "HbA1cの正常値と糖尿病診断基準値は？",
        "answer":   "正常値6.2%未満、糖尿病診断基準6.5%以上",
        "keywords": ["6.5", "HbA1c", "糖尿病"],
        "category": "診断",
    },
    {
        "question": "ワルファリンの作用機序を説明してください。",
        "answer":   "ビタミンK依存性凝固因子（II、VII、IX、X）の産生を阻害する",
        "keywords": ["ビタミンK", "凝固因子", "阻害"],
        "category": "薬理",
    },
    {
        "question": "心房細動に対する抗凝固療法の適応を判定するスコアは？",
        "answer":   "CHA2DS2-VAScスコア",
        "keywords": ["CHA2DS2", "VASc", "抗凝固"],
        "category": "スコア",
    },
    {
        "question": "急性腎障害（AKI）のKDIGO基準を述べてください。",
        "answer":   "血清クレアチニン48時間内に0.3mg/dL以上上昇、または7日以内に基準値の1.5倍以上",
        "keywords": ["クレアチニン", "0.3mg", "48時間", "1.5倍"],
        "category": "診断",
    },
    {
        "question": "褥瘡の危険因子を3つ述べてください。",
        "answer":   "不動、栄養不良、皮膚の湿潤",
        "keywords": ["不動", "栄養", "湿潤"],
        "category": "予防",
    },
    {
        "question": "DKA（糖尿病性ケトアシドーシス）の診断三徴は？",
        "answer":   "高血糖、代謝性アシドーシス、ケトン血症",
        "keywords": ["高血糖", "アシドーシス", "ケトン"],
        "category": "診断",
    },
    {
        "question": "オピオイド過量投与の解毒薬は何ですか？",
        "answer":   "ナロキソン（オピオイド受容体拮抗薬）",
        "keywords": ["ナロキソン", "拮抗薬"],
        "category": "薬理",
    },
    {
        "question": "肺塞栓症の診断に使用するD-ダイマーの役割は？",
        "answer":   "陰性的中率が高く除外診断に有用だが、特異性は低い",
        "keywords": ["D-ダイマー", "陰性的中率", "除外"],
        "category": "検査",
    },
    {
        "question": "小児の発熱に対するアセトアミノフェンの用量は？",
        "answer":   "体重1kgあたり10〜15mg、6時間毎に投与",
        "keywords": ["10〜15mg", "体重", "アセトアミノフェン"],
        "category": "薬理",
    },
    {
        "question": "心不全の治療薬として推奨されるACE阻害薬の例を挙げてください。",
        "answer":   "エナラプリル、ラミプリル、カプトプリルなどのACE阻害薬",
        "keywords": ["エナラプリル", "ACE阻害薬", "カプトプリル"],
        "category": "薬理",
    },
]

# ---------------------------------------------------------------------------
# Simulated LLM response engine (deterministic extraction)
# ---------------------------------------------------------------------------
# Maps each strategy to an extraction function.
# Simulate "imperfect but pattern-based" responses that partially match keywords.

def _keyword_score(answer_text, keywords):
    """How many keywords appear in the answer."""
    return sum(1 for kw in keywords if kw in answer_text)


class SimulatedLLM:
    """
    Deterministic simulator.
    Returns responses with different accuracy profiles per strategy.
    """

    def __init__(self, rng):
        self.rng = rng

    def _make_response(self, case, hit_prob, noise_ratio=0.1):
        """
        hit_prob: probability that each keyword is included.
        noise_ratio: fraction of unrelated tokens added.
        """
        answer = case["answer"]
        keywords = case["keywords"]
        parts = []
        for kw in keywords:
            if self.rng.random() < hit_prob:
                parts.append(kw)
        # Combine with some surrounding answer text
        if parts:
            resp = "、".join(parts)
            # Sometimes include full answer fragments
            if self.rng.random() < hit_prob * 0.6:
                resp = answer
        else:
            resp = case["question"][:10] + "については専門的な判断が必要です"
        return resp

    def zero_shot(self, case):
        return self._make_response(case, hit_prob=0.55)

    def role_prompted(self, case):
        return self._make_response(case, hit_prob=0.68)

    def chain_of_thought(self, case):
        return self._make_response(case, hit_prob=0.73)

    def few_shot(self, case):
        return self._make_response(case, hit_prob=0.80)

    def self_consistency(self, case, n=5):
        """Majority vote from N generations with slightly higher accuracy."""
        generations = [self._make_response(case, hit_prob=0.77) for _ in range(n)]
        # Pick the generation that shares the most keywords with other generations
        best = generations[0]
        best_score = 0
        for g in generations:
            score = sum(1 for other in generations if g == other)
            if score > best_score:
                best_score = score
                best = g
        return best


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tokenize(text):
    """Simple character-level tokenizer for Japanese."""
    return list(text.replace(" ", "").replace("　", ""))


def exact_match(pred, gold):
    return 1.0 if pred.strip() == gold.strip() else 0.0


def token_f1(pred, gold):
    pred_tokens = set(tokenize(pred))
    gold_tokens = set(tokenize(gold))
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a, b):
    """Longest common subsequence length (character-level)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use O(n) space DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(pred, gold):
    pred_chars = list(pred)
    gold_chars = list(gold)
    lcs = _lcs_length(pred_chars, gold_chars)
    if len(pred_chars) == 0 or len(gold_chars) == 0:
        return 0.0
    precision = lcs / len(pred_chars)
    recall    = lcs / len(gold_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_response(pred, case):
    gold = case["answer"]
    return {
        "exact_match": exact_match(pred, gold),
        "token_f1":    token_f1(pred, gold),
        "rouge_l":     rouge_l(pred, gold),
    }


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

STRATEGIES = ["zero_shot", "role_prompted", "chain_of_thought", "few_shot", "self_consistency"]
STRATEGY_LABELS = {
    "zero_shot":        "Zero-shot",
    "role_prompted":    "Role Prompting",
    "chain_of_thought": "Chain-of-thought",
    "few_shot":         "Few-shot (3-ex)",
    "self_consistency": "Self-consistency",
}


def evaluate_all():
    rng = np.random.default_rng(42)
    llm = SimulatedLLM(rng)

    results = {s: {"scores": [], "predictions": []} for s in STRATEGIES}

    for case in QA_CASES:
        for strategy in STRATEGIES:
            if strategy == "zero_shot":
                pred = llm.zero_shot(case)
            elif strategy == "role_prompted":
                pred = llm.role_prompted(case)
            elif strategy == "chain_of_thought":
                pred = llm.chain_of_thought(case)
            elif strategy == "few_shot":
                pred = llm.few_shot(case)
            elif strategy == "self_consistency":
                pred = llm.self_consistency(case)
            else:
                pred = ""

            sc = score_response(pred, case)
            results[strategy]["scores"].append(sc)
            results[strategy]["predictions"].append(pred)

    # Aggregate
    summary = {}
    for strategy in STRATEGIES:
        scores = results[strategy]["scores"]
        summary[strategy] = {
            "exact_match": float(np.mean([s["exact_match"] for s in scores])),
            "token_f1":    float(np.mean([s["token_f1"]    for s in scores])),
            "rouge_l":     float(np.mean([s["rouge_l"]     for s in scores])),
        }

    return results, summary


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def save_radar_chart(summary, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = ["exact_match", "token_f1", "rouge_l"]
        metric_labels = ["Exact Match", "Token F1", "ROUGE-L"]
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for (strategy, color) in zip(STRATEGIES, colors):
            values = [summary[strategy][m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=STRATEGY_LABELS[strategy],
                    color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.set_title("Prompt Strategy Comparison\n(Medical QA — 20 Cases)", size=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Radar chart saved to {out_path}")
    except ImportError:
        print("[Warning] matplotlib not available — skipping radar chart")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Engineering Strategy Comparison")
    print("Medical QA — 20 test cases, 5 strategies")
    print("=" * 60)

    results, summary = evaluate_all()

    print("\n--- Metric Summary ---")
    header = f"{'Strategy':<20}  {'ExactMatch':>10}  {'Token F1':>10}  {'ROUGE-L':>10}"
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        s = summary[strategy]
        print(f"{STRATEGY_LABELS[strategy]:<20}  "
              f"{s['exact_match']:>10.3f}  "
              f"{s['token_f1']:>10.3f}  "
              f"{s['rouge_l']:>10.3f}")

    # Per-case preview
    print("\n--- Sample Predictions (first 3 cases, zero_shot vs self_consistency) ---")
    for i, case in enumerate(QA_CASES[:3]):
        print(f"\nQ{i+1}: {case['question']}")
        print(f"  Gold:             {case['answer'][:60]}")
        print(f"  Zero-shot pred:   {results['zero_shot']['predictions'][i][:60]}")
        print(f"  Self-cons pred:   {results['self_consistency']['predictions'][i][:60]}")

    # Save radar chart
    radar_path = os.path.join(OUT_DIR, "prompt_strategy_radar.png")
    save_radar_chart(summary, radar_path)

    # Save results JSON
    output = {
        "summary": summary,
        "per_case": [
            {
                "question": case["question"],
                "answer":   case["answer"],
                "category": case["category"],
                "predictions": {
                    s: {
                        "text":   results[s]["predictions"][i],
                        "scores": results[s]["scores"][i],
                    }
                    for s in STRATEGIES
                },
            }
            for i, case in enumerate(QA_CASES)
        ],
    }
    json_path = os.path.join(OUT_DIR, "prompt_engineering_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {json_path}")
