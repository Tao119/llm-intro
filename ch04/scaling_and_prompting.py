import numpy as np
import matplotlib.pyplot as plt


def plot_scaling_law():
    params = np.logspace(7, 11, 100)
    A, B, alpha = 406.4, 1.69, 0.34
    loss = A / (params ** alpha) + B
    plt.figure(figsize=(7, 4))
    plt.loglog(params, loss, "b-", linewidth=2, label=r"$L = A/N^\alpha + B$ (Chinchilla近似)")
    for n, label in [(1e8, "100M"), (1e9, "1B"), (1e10, "10B")]:
        l = A / n**alpha + B
        plt.loglog(n, l, "ro", markersize=8)
        plt.annotate(label, (n, l), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("パラメータ数 N")
    plt.ylabel("テスト損失 L")
    plt.title("スケーリング則: パラメータ数と損失の関係")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scaling_law.png", dpi=120)
    plt.close()
    print("saved: scaling_law.png")


def show_prompt_patterns():
    print("\n" + "="*60)
    print("プロンプト設計パターン")
    print("="*60)

    print("\n[Zero-Shot]")
    print("感情分析してください。\n入力: 今日は天気が良くて気持ちいい。\n感情:")

    print("\n[One-Shot]")
    print("感情分析してください。\n例)\n入力: 仕事が忙しくてストレスがたまる。\n感情: ネガティブ\n---\n入力: 今日は天気が良くて気持ちいい。\n感情:")

    print("\n[Few-Shot]")
    print("感情分析してください。\n例1)\n入力: 誕生日パーティーがとても楽しかった。\n感情: ポジティブ\n例2)\n入力: 電車が遅延して遅刻した。\n感情: ネガティブ\n例3)\n入力: 今日は特に何もなかった。\n感情: ニュートラル\n---\n入力: 今日は天気が良くて気持ちいい。\n感情:")

    print("\n[Chain-of-Thought]")
    print("問題: 太郎は3個のりんごを持っていた。花子から2個もらい、次に5個買った。今何個ある？")
    print("考え方: まず最初に3個。花子から2個もらうと3+2=5個。さらに5個買うと5+5=10個。")
    print("答え: 10個")

    print("\n[System + User (ChatGPT形式)]")
    print("System: あなたは日本語の専門家です。丁寧に回答してください。")
    print("User: 「行く」の尊敬語は何ですか？")
    print("Assistant:")


def show_alignment_concepts():
    print("\n" + "="*60)
    print("アライメント: Helpful / Honest / Harmless")
    print("="*60)
    examples = [
        ("役立つ(Helpful)", "ユーザーの意図を正確に把握し、有益な情報を提供"),
        ("正直(Honest)",    "知らないことは知らないと言う。ハルシネーションを避ける"),
        ("無害(Harmless)",  "差別・暴力・違法な内容を生成しない"),
    ]
    for label, desc in examples:
        print(f"  {label}: {desc}")


if __name__ == "__main__":
    plot_scaling_law()
    show_prompt_patterns()
    show_alignment_concepts()
