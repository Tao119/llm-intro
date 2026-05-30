# 大規模言語モデル入門

理論編（ch01-ch04）は アーキテクチャの純NumPy/PyTorch実装。  
実装編（ch05-ch09）は HuggingFace `transformers` を用いた日本語NLP。

## 章構成

| 章 | テーマ | 実装対象 | 状態 |
|----|--------|----------|------|
| ch01 | はじめに | transformers デモ、word embedding、LLM概要 | 実装中 |
| ch02 | Transformer | Self-Attention・Multi-Head・FFN・LayerNorm を純実装 | 実装中 |
| ch03 | 大規模言語モデルの基礎 | GPT/BERT/T5 アーキテクチャ解説 + transformers使用例 | 予定 |
| ch04 | LLMの進展 | スケーリング則・CoT・アライメント・指示チューニング理論 | 予定 |
| ch05 | ファインチューニング | JGLUE(感情分析・NLI・意味類似度)、LoRA | 予定 |
| ch06 | 固有表現認識 | BERTファインチューニング・CRF | 予定 |
| ch07 | 要約生成 | T5ファインチューニング・ROUGE/BLEU | 予定 |
| ch08 | 文埋め込み | SimCSE・Faiss類似文検索 | 予定 |
| ch09 | 質問応答 | OpenAI API・RAG with BPR | 予定 |

## 実行環境

- 理論実装: 純NumPy + Matplotlib
- 実装編: Python 3.10+, transformers, torch, datasets, scikit-learn
- ch09: OpenAI API key 必要

## ディレクトリ構成

```
llm-intro/
├── ch01/          # transformers基礎・word embedding
├── ch02/          # Transformer実装 (純NumPy)
├── ch03/          # GPT/BERT/T5
├── ch04/          # スケーリング・CoT・RLHF理論
├── ch05/          # ファインチューニング実装
├── ch06/          # NER
├── ch07/          # 要約生成
├── ch08/          # 文埋め込み
├── ch09/          # 質問応答
├── common/        # 共通ユーティリティ
├── dataset/       # データセット管理
└── experiments/   # 実験レポート
```
