import os
import sys
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
import evaluate

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
OUTPUT_DIR = "results/marc-ja"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_marc_ja():
    try:
        ds = load_dataset("shunk031/JGLUE", name="MARC-ja", trust_remote_code=True)
        label_map = {"negative": 0, "positive": 1}
        def fix_label(example):
            lbl = example.get("label", example.get("labels", 0))
            if isinstance(lbl, str):
                example["label"] = label_map.get(lbl, 0)
            return example
        ds = ds.map(fix_label)
        return ds["train"], ds["validation"]
    except Exception as e:
        print(f"JGLUE load failed ({e}), using synthetic data")
        pos = ["良い製品です", "最高でした", "とても満足", "おすすめします", "品質が良い",
               "素晴らしいサービス", "また買います", "期待以上でした", "コスパ最高", "大満足"]
        neg = ["最悪でした", "使えません", "がっかりした", "返品したい", "品質が悪い",
               "二度と買わない", "時間の無駄", "全く役立たない", "最低品質", "お金の無駄"]
        texts  = pos * 10 + neg * 10
        labels = [1] * 100 + [0] * 100
        import random; idx = list(range(200)); random.shuffle(idx)
        texts = [texts[i] for i in idx]; labels = [labels[i] for i in idx]
        train_ds = Dataset.from_dict({"sentence": texts[:160], "label": labels[:160]})
        val_ds   = Dataset.from_dict({"sentence": texts[160:], "label": labels[160:]})
        return train_ds, val_ds


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds = load_marc_ja()
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    text_col = "sentence" if "sentence" in train_ds.column_names else "review"

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=128)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ["label"]])
    val_tok   = val_ds.map(tokenize,   batched=True, remove_columns=[c for c in val_ds.column_names   if c not in ["label"]])
    train_tok = train_tok.rename_column("label", "labels") if "label" in train_tok.column_names else train_tok
    val_tok   = val_tok.rename_column("label", "labels")   if "label" in val_tok.column_names   else val_tok
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return accuracy.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        use_mps_device=(device == "mps"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"\nfinal accuracy: {results['eval_accuracy']:.4f}")
    trainer.save_model(OUTPUT_DIR + "/best")
    print(f"model saved: {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()
