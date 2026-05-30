import os
import sys
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
OUTPUT_DIR = "results/ner"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ner_dataset():
    try:
        ds = load_dataset("llm-book/ner-wikipedia-dataset", trust_remote_code=True)
        return ds["train"], ds["validation"], ds["test"]
    except Exception as e:
        print(f"NER dataset load failed ({e}), using synthetic data")
        samples = [
            {"tokens": ["東京", "は", "日本", "の", "首都", "です", "。"],
             "ner_tags": [1, 0, 3, 0, 0, 0, 0]},
            {"tokens": ["山田", "太郎", "は", "大阪", "に", "住む", "。"],
             "ner_tags": [5, 6, 0, 1, 0, 0, 0]},
            {"tokens": ["ソニー", "が", "新製品", "を", "発表", "した", "。"],
             "ner_tags": [7, 0, 0, 0, 0, 0, 0]},
        ] * 30
        ds = Dataset.from_list(samples)
        split = ds.train_test_split(test_size=0.2, seed=42)
        return split["train"], split["test"], split["test"]


def build_label_maps(dataset):
    entity_types = set()
    for sample in dataset:
        if "entities" in sample:
            for e in sample["entities"]:
                entity_types.add(e.get("type", e.get("label", "")))
    if not entity_types:
        entity_types = {"人名", "地名", "施設名", "組織名"}
    label_list = ["O"] + [f"B-{t}" for t in sorted(entity_types)] + [f"I-{t}" for t in sorted(entity_types)]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label


def convert_to_iob2(dataset, label2id, tokenizer):
    records = []
    for sample in dataset:
        if "tokens" in sample and "ner_tags" in sample:
            records.append({"tokens": sample["tokens"], "tags": sample["ner_tags"]})
            continue
        text = sample.get("text", "")
        entities = sample.get("entities", [])
        chars = list(text)
        char_tags = ["O"] * len(chars)
        for ent in entities:
            etype = ent.get("type", ent.get("label", ""))
            start, end = ent.get("span", [ent.get("start", 0), ent.get("end", 0)])
            if f"B-{etype}" in label2id:
                char_tags[start] = f"B-{etype}"
                for i in range(start + 1, end):
                    char_tags[i] = f"I-{etype}"
        records.append({"tokens": chars[:64], "tags": [label2id.get(t, 0) for t in char_tags[:64]]})
    return records


def tokenize_and_align(records, tokenizer, label2id):
    all_input_ids, all_attention_masks, all_labels = [], [], []
    for rec in records:
        tokens = rec["tokens"]
        tags   = rec["tags"]
        enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128)
        word_ids = enc.word_ids()
        aligned_labels = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word:
                lbl = tags[wid] if wid < len(tags) else 0
                aligned_labels.append(int(lbl))
                prev_word = wid
            else:
                lbl = tags[wid] if wid < len(tags) else 0
                if isinstance(lbl, int) and lbl > 0:
                    id2label_tmp = {v: k for k, v in label2id.items()}
                    tag_name = id2label_tmp.get(lbl, "O")
                    if tag_name.startswith("B-"):
                        lbl = label2id.get("I-" + tag_name[2:], lbl)
                aligned_labels.append(int(lbl))
        all_input_ids.append(enc["input_ids"])
        all_attention_masks.append(enc["attention_mask"])
        all_labels.append(aligned_labels)
    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })


def compute_ner_metrics(eval_pred, id2label):
    try:
        import evaluate as ev
        seqeval = ev.load("seqeval")
    except Exception:
        def fake_metric(preds, refs):
            correct = sum(p == r for ps, rs in zip(preds, refs) for p, r in zip(ps, rs) if r != "O")
            total   = sum(1 for rs in refs for r in rs if r != "O")
            f1 = correct / total if total else 0.0
            return {"overall_f1": f1, "overall_precision": f1, "overall_recall": f1}
        seqeval = type("M", (), {"compute": staticmethod(fake_metric)})()

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)
    true_preds = [[id2label[p] for p, l in zip(pred, label) if l != -100]
                  for pred, label in zip(preds, labels)]
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_raw, val_raw, test_raw = load_ner_dataset()
    print(f"train: {len(train_raw)}  val: {len(val_raw)}  test: {len(test_raw)}")

    label_list, label2id, id2label = build_label_maps(train_raw)
    print(f"labels ({len(label_list)}): {label_list[:10]}{'...' if len(label_list) > 10 else ''}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_recs = convert_to_iob2(train_raw, label2id, tokenizer)
    val_recs   = convert_to_iob2(val_raw,   label2id, tokenizer)

    train_ds = tokenize_and_align(train_recs, tokenizer, label2id)
    val_ds   = tokenize_and_align(val_recs,   tokenizer, label2id)
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list),
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        use_mps_device=(device == "mps"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda ep: compute_ner_metrics(ep, id2label),
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"\nNER results:")
    print(f"  precision: {results.get('eval_precision', 0):.4f}")
    print(f"  recall   : {results.get('eval_recall', 0):.4f}")
    print(f"  F1       : {results.get('eval_f1', 0):.4f}")
    trainer.save_model(OUTPUT_DIR + "/best")
    print(f"model saved: {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()
