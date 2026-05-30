import sys
import torch
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, pipeline, set_seed
)

set_seed(42)


def demo_gpt2():
    print("\n=== GPT-2 Text Generation ===")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=30, do_sample=True, temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"prompt : {prompt}")
    print(f"output : {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    tokens = tokenizer.tokenize(prompt)
    ids    = tokenizer.encode(prompt)
    print(f"\ntokenize: {tokens}")
    print(f"input_ids: {ids}")

    logits = model(**inputs).logits
    print(f"logits shape (seq_len × vocab): {logits.shape}")


def demo_bert_mlm():
    print("\n=== BERT Masked LM (Japanese) ===")
    model_name = "cl-tohoku/bert-base-japanese-v3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        print(f"Japanese BERT load failed ({e}), using bert-base-multilingual-cased")
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()

    text = "東京は日本の[MASK]です。"
    inputs = tokenizer(text, return_tensors="pt")
    mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
    with torch.no_grad():
        logits = model(**inputs).logits
    top5 = logits[0, mask_idx].topk(5)
    print(f"input: {text}")
    print("top-5 predictions:")
    for score, idx in zip(top5.values, top5.indices):
        print(f"  {tokenizer.decode([idx.item()]):12s}  score={score.item():.3f}")

    print(f"\nModel type: {model.__class__.__name__}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")


def demo_mt5():
    print("\n=== mT5 Text-to-Text ===")
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    prompts = [
        "translate English to Japanese: The weather is nice today.",
        "summarize: The transformer architecture has revolutionized natural language processing.",
    ]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=40)
        print(f"input : {prompt}")
        print(f"output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        enc_shape = model.encoder(**inputs).last_hidden_state.shape
        print(f"encoder output shape: {enc_shape}\n")


def show_model_anatomy():
    print("\n=== Model Architecture Comparison ===")
    from transformers import GPT2Config, BertConfig, T5Config
    gpt_cfg  = GPT2Config(n_layer=12, n_head=12, n_embd=768)
    bert_cfg = BertConfig(num_hidden_layers=12, num_attention_heads=12, hidden_size=768)
    t5_cfg   = T5Config(num_layers=6, num_heads=8, d_model=512)
    print(f"{'Model':<10} {'Type':<20} {'Layers':<8} {'Heads':<7} {'d_model'}")
    print("-" * 55)
    print(f"{'GPT-2':<10} {'decoder-only':<20} {gpt_cfg.n_layer:<8} {gpt_cfg.n_head:<7} {gpt_cfg.n_embd}")
    print(f"{'BERT':<10} {'encoder-only':<20} {bert_cfg.num_hidden_layers:<8} {bert_cfg.num_attention_heads:<7} {bert_cfg.hidden_size}")
    print(f"{'T5':<10} {'encoder-decoder':<20} {t5_cfg.num_layers:<8} {t5_cfg.num_heads:<7} {t5_cfg.d_model}")


if __name__ == "__main__":
    demo_gpt2()
    demo_bert_mlm()
    demo_mt5()
    show_model_anatomy()
