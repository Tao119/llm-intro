# 大規模言語モデル入門

理論編（ch01-ch04）+ 実装編（ch05-ch09）。

## 章構成と実装内容

| 章 | テーマ | 実装 | 実行方法 |
|----|--------|------|---------|
| ch01 | はじめに | *(概念説明、ch02-ch09の布石)* | — |
| **ch02** | **Transformer** | 純NumPy実装（Self-Attention・Multi-Head・LayerNorm・Positional Encoding） | `python3 ch02/transformer.py` |
| **ch03** | GPT/BERT/T5 | transformers API デモ（生成・穴埋め・翻訳） | `python3 ch03/transformers_intro.py` |
| **ch04** | LLMの進展 | スケーリング則プロット・プロンプト設計・CoT・アライメント | `python3 ch04/scaling_and_prompting.py` |
| **ch05** | ファインチューニング | MARC-ja感情分析（BERT + Trainer）← JGLUE | `python3 ch05/finetune_sentiment.py` |
| **ch06** | 固有表現認識 | IOB2ラベリング + BERTトークン分類 + seqeval | `python3 ch06/ner.py` |
| **ch07** | 要約生成 | T5ファインチューニング + ROUGE評価 | `python3 ch07/summarization.py` |
| **ch08** | 文埋め込み | 教師なしSimCSE + Faiss類似文検索 | `python3 ch08/simcse.py` |
| **ch09** | 質問応答 | 抽出型QA(BERT) + RAGパイプライン | `python3 ch09/qa_rag.py` |

## 環境構築

```bash
pip install transformers datasets sentencepiece fugashi ipadic \
            langchain-community langchain-core peft faiss-cpu evaluate seqeval
```

MPS（Apple Silicon）/ CUDA / CPU を自動選択。  
ch05・ch06・ch07 の日本語ファインチューニングは MPS で動作確認済み。

## 主要モデル

| 用途 | モデル |
|------|--------|
| 日本語分類/NER | `cl-tohoku/bert-base-japanese-v3` |
| 日本語生成 | `rinna/japanese-gpt2-medium` |
| 多言語T5 | `google/mt5-small` |
| 日本語T5 | `sonoisa/t5-base-japanese` |
