import os
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_synthetic_dataset():
    articles = [
        ("東京都知事選挙が来月行われる予定で、複数の候補者が名乗りを上げている。", "都知事選来月実施へ"),
        ("新型コロナウイルスのワクチン接種が全国で進んでいる。", "コロナワクチン接種進む"),
        ("日本代表がワールドカップ予選で勝利し、本戦出場に近づいた。", "日本代表W杯予選突破へ"),
        ("東京証券取引所で日経平均株価が大幅上昇した。", "日経平均大幅上昇"),
        ("政府は少子化対策として新たな支援策を発表した。", "政府が少子化対策発表"),
        ("大型台風が九州に上陸し、各地で被害が出ている。", "台風九州上陸で被害"),
        ("日本銀行が金融政策を現状維持することを決定した。", "日銀金融政策を維持"),
        ("国内の電気自動車販売台数が過去最高を記録した。", "EV販売台数が過去最高"),
        ("文部科学省が新しい学習指導要領の改訂案を公表した。", "学習指導要領改訂案公表"),
        ("東京オリンピック施設の後利用計画が明らかになった。", "五輪施設後利用計画判明"),
        ("政府は再生可能エネルギーの普及促進策を強化する方針を示した。", "再エネ普及策を強化"),
        ("国内の物価上昇が続き、消費者物価指数が前年比で上昇した。", "消費者物価指数が上昇"),
        ("大手自動車メーカーが電動化戦略を加速すると発表した。", "自動車大手が電動化加速"),
        ("農林水産省が食料安全保障強化に向けた政策を打ち出した。", "食料安保強化策を発表"),
        ("東京都内で住宅価格の上昇が続いており、購入困難な状況が深刻化している。", "都内住宅価格高騰が深刻"),
        ("国会で予算委員会が開かれ、来年度予算案の審議が始まった。", "来年度予算案審議開始"),
        ("気象庁が今年の夏は平年より気温が高くなると予測した。", "今夏は平年より高温予測"),
        ("地方創生の取り組みとして移住促進策が各地で広がっている。", "移住促進策が各地で拡大"),
        ("観光庁が訪日外国人数の新たな目標を設定した。", "訪日客数の新目標設定"),
        ("医療のデジタル化推進のため、電子カルテの標準化が進む。", "電子カルテ標準化が進展"),
    ]
    full = []
    for i in range(100):
        art, title = articles[i % len(articles)]
        full.append({"article": art + f"（{i+1}報）", "title": title})
    return full


def load_dataset_with_fallback():
    try:
        from datasets import load_dataset
        ds = load_dataset("llm-book/livedoor-news-corpus", trust_remote_code=True)
        train = ds["train"].select(range(min(200, len(ds["train"]))))
        val = ds["validation"].select(range(min(50, len(ds["validation"]))))

        def rename(example):
            return {"article": example.get("text", example.get("article", "")),
                    "title": example.get("title", "")}

        train = train.map(rename)
        val = val.map(rename)
        return train, val
    except Exception:
        pass

    from datasets import Dataset
    data = make_synthetic_dataset()
    random.shuffle(data)
    train_data = data[:80]
    val_data = data[80:]
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def load_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    for model_name in ["sonoisa/t5-base-japanese", "google/mt5-small"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Loaded model: {model_name}")
            return tokenizer, model, model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    raise RuntimeError("Could not load any model")


def preprocess(examples, tokenizer, max_input=512, max_target=64):
    inputs = tokenizer(
        examples["article"],
        max_length=max_input,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["title"],
            max_length=max_target,
            truncation=True,
            padding="max_length",
        )
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in tgt]
        for tgt in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs


def compute_rouge(eval_pred, tokenizer):
    import evaluate
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
    return {k: round(v, 4) for k, v in result.items()}


def main():
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

    train_ds, val_ds = load_dataset_with_fallback()
    tokenizer, model, model_name = load_model_and_tokenizer()

    train_tokenized = train_ds.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tokenized = val_ds.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    use_mps = torch.backends.mps.is_available()
    output_dir = os.path.join(os.path.dirname(__file__), "results")

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=64,
        fp16=False,
        no_cuda=not torch.cuda.is_available(),
        use_mps_device=use_mps,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda p: compute_rouge(p, tokenizer),
    )

    trainer.train()

    results = trainer.evaluate()
    print("\n=== ROUGE Scores ===")
    for k in ["eval_rouge1", "eval_rouge2", "eval_rougeL"]:
        if k in results:
            label = k.replace("eval_", "").upper()
            print(f"{label}: {results[k]:.4f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
