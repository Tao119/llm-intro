import os
import json
import random
import string
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

KNOWLEDGE_BASE = [
    ("東京", "東京は日本の首都であり、世界最大級の都市圏を持つ。人口は約1400万人で、政治・経済・文化の中心地として機能している。"),
    ("富士山", "富士山は日本最高峰の山で、標高3776メートル。静岡県と山梨県にまたがり、2013年にユネスコ世界文化遺産に登録された。"),
    ("京都", "京都は794年から1869年まで日本の首都だった古都。金閣寺、清水寺、嵐山など多くの寺社仏閣があり、年間5000万人以上の観光客が訪れる。"),
    ("新幹線", "新幹線は1964年の東京オリンピックに合わせて開業した高速鉄道。東海道新幹線が最初で、現在は全国に路線網が広がっている。最高速度は320km/h。"),
    ("桜", "桜は日本を代表する花木で、毎年3月から5月にかけて開花する。花見の習慣は平安時代から続き、現代でも重要な文化行事として親しまれている。"),
    ("寿司", "寿司は日本を代表する料理で、酢飯に魚介類や野菜を合わせた食べ物。江戸時代に発展した握り寿司が世界的に有名で、ユネスコ無形文化遺産の和食の代表格。"),
    ("俳句", "俳句は5・7・5の17音からなる日本の詩形式。松尾芭蕉が確立し、与謝蕪村や小林一茶が発展させた。季語を含むことが伝統的な決まり。"),
    ("相撲", "相撲は日本の国技とされる格闘技で、2000年以上の歴史を持つ。力士が土俵の上で相手を押し出したり倒したりして勝負を競う。年6回の本場所が行われる。"),
    ("歌舞伎", "歌舞伎は江戸時代に発展した日本の伝統演劇。豪華な衣装と化粧、様式的な演技が特徴。2008年にユネスコ無形文化遺産に登録された。"),
    ("温泉", "日本には約3000ヶ所の温泉地があり、火山活動が豊富な地質が源泉。別府温泉、草津温泉、有馬温泉が三大名湯として知られる。"),
    ("アニメ", "日本のアニメーション産業は世界最大規模で、年間200タイトル以上が制作される。宮崎駿監督のスタジオジブリ作品は国際的に高く評価されている。"),
    ("日本語", "日本語は日本で話される言語で、話者数は約1億3千万人。ひらがな・カタカナ・漢字の3種の文字を使用する独特の文字体系を持つ。"),
    ("茶道", "茶道は抹茶を点てる日本の伝統的な儀式。千利休が16世紀に完成させ、「和敬清寂」の精神を重視する。ユネスコ無形文化遺産に登録されている。"),
    ("日本の経済", "日本は世界第3位の経済大国で、GDPは約4兆ドル。自動車・電機・精密機械などの製造業が強みで、トヨタ・ソニー・任天堂などの世界的企業がある。"),
    ("北海道", "北海道は日本最北の島で面積は日本の約22%を占める。農業・酪農・漁業が盛んで、雪まつりやラベンダー畑が観光地として知られる。"),
    ("沖縄", "沖縄は日本最南西端の県で、亜熱帯性気候の島々からなる。琉球王国の歴史文化が残り、美しいサンゴ礁と独自の食文化で知られる。"),
    ("日本の四季", "日本は四季がはっきりしており、春の桜、夏の海水浴、秋の紅葉、冬のスキーと季節ごとに異なる魅力がある。この豊かな自然が日本文化の基盤となっている。"),
    ("東大寺", "東大寺は奈良県にある8世紀創建の寺院。世界最大の木造建築物である大仏殿に、高さ約15メートルの奈良の大仏が安置されている。"),
    ("忍者", "忍者は日本の戦国時代に活躍した諜報員・工作員。特殊な訓練を受け、情報収集や暗殺などを行った。三重県伊賀市と滋賀県甲賀市が有名な忍者の里として知られる。"),
    ("日本の伝統音楽", "日本の伝統音楽には、三味線・琴・尺八などを使った楽曲がある。雅楽は世界最古のオーケストラ音楽のひとつとされ、宮廷で演奏されてきた。"),
]

SYNTHETIC_QA = [
    {"question": "東京の人口は何人ですか？", "context": "東京は日本の首都であり、世界最大級の都市圏を持つ。人口は約1400万人で、政治・経済・文化の中心地として機能している。", "answer": "約1400万人"},
    {"question": "富士山の標高は何メートルですか？", "context": "富士山は日本最高峰の山で、標高3776メートル。静岡県と山梨県にまたがり、2013年にユネスコ世界文化遺産に登録された。", "answer": "3776メートル"},
    {"question": "京都はいつまで首都でしたか？", "context": "京都は794年から1869年まで日本の首都だった古都。金閣寺、清水寺、嵐山など多くの寺社仏閣があり、年間5000万人以上の観光客が訪れる。", "answer": "1869年"},
    {"question": "新幹線はいつ開業しましたか？", "context": "新幹線は1964年の東京オリンピックに合わせて開業した高速鉄道。東海道新幹線が最初で、現在は全国に路線網が広がっている。最高速度は320km/h。", "answer": "1964年"},
    {"question": "俳句の音数は何音ですか？", "context": "俳句は5・7・5の17音からなる日本の詩形式。松尾芭蕉が確立し、与謝蕪村や小林一茶が発展させた。季語を含むことが伝統的な決まり。", "answer": "17音"},
    {"question": "相撲の本場所は年何回行われますか？", "context": "相撲は日本の国技とされる格闘技で、2000年以上の歴史を持つ。力士が土俵の上で相手を押し出したり倒したりして勝負を競う。年6回の本場所が行われる。", "answer": "年6回"},
    {"question": "東大寺の大仏の高さは？", "context": "東大寺は奈良県にある8世紀創建の寺院。世界最大の木造建築物である大仏殿に、高さ約15メートルの奈良の大仏が安置されている。", "answer": "約15メートル"},
    {"question": "日本のGDPは世界何位ですか？", "context": "日本は世界第3位の経済大国で、GDPは約4兆ドル。自動車・電機・精密機械などの製造業が強みで、トヨタ・ソニー・任天堂などの世界的企業がある。", "answer": "世界第3位"},
    {"question": "北海道の面積は日本の何パーセントですか？", "context": "北海道は日本最北の島で面積は日本の約22%を占める。農業・酪農・漁業が盛んで、雪まつりやラベンダー畑が観光地として知られる。", "answer": "約22%"},
    {"question": "茶道を完成させたのは誰ですか？", "context": "茶道は抹茶を点てる日本の伝統的な儀式。千利休が16世紀に完成させ、「和敬清寂」の精神を重視する。ユネスコ無形文化遺産に登録されている。", "answer": "千利休"},
]


def find_answer_span(context, answer):
    start = context.find(answer)
    if start == -1:
        return 0, 0
    return start, start + len(answer)


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.samples = []
        for item in data:
            enc = tokenizer(
                item["question"],
                item["context"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = enc["offset_mapping"].squeeze(0).tolist()
            char_start, char_end = find_answer_span(item["context"], item["answer"])

            question_end = enc.sequence_ids(0).index(1) if 1 in enc.sequence_ids(0) else 0

            start_pos, end_pos = 0, 0
            for i, (s, e) in enumerate(offset_mapping):
                if i < question_end:
                    continue
                if s <= char_start < e:
                    start_pos = i
                if s < char_end <= e:
                    end_pos = i
                    break

            sample = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "start_positions": torch.tensor(start_pos),
                "end_positions": torch.tensor(end_pos),
            }
            if "token_type_ids" in enc:
                sample["token_type_ids"] = enc["token_type_ids"].squeeze(0)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_qa_dataset(tokenizer):
    try:
        from datasets import load_dataset
        ds = load_dataset("llm-book/jsquad-for-bert-v1.1", trust_remote_code=True)

        def build_from_hf(examples):
            out = []
            for q, ctx, ans in zip(examples["question"], examples["context"], examples["answers"]):
                answer_text = ans["text"][0] if ans["text"] else ""
                out.append({"question": q, "context": ctx, "answer": answer_text})
            return out

        train_data = build_from_hf(ds["train"][:100])
        val_data = build_from_hf(ds["validation"][:30])
        print("Loaded llm-book/jsquad-for-bert-v1.1")
        return QADataset(train_data, tokenizer), QADataset(val_data, tokenizer)
    except Exception:
        pass

    print("Using synthetic QA dataset")
    random.shuffle(SYNTHETIC_QA)
    train_data = SYNTHETIC_QA[:8]
    val_data = SYNTHETIC_QA[8:]
    return QADataset(train_data, tokenizer), QADataset(val_data, tokenizer)


def compute_em_f1(predictions, references):
    def normalize(text):
        text = text.lower().strip()
        for punc in string.punctuation:
            text = text.replace(punc, " ")
        return " ".join(text.split())

    def f1_score(pred, ref):
        pred_tokens = normalize(pred).split()
        ref_tokens = normalize(ref).split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    em_scores = [float(normalize(p) == normalize(r)) for p, r in zip(predictions, references)]
    f1_scores = [f1_score(p, r) for p, r in zip(predictions, references)]
    return np.mean(em_scores), np.mean(f1_scores)


def mean_pool(hidden, mask):
    mask_expanded = mask.unsqueeze(-1).float()
    return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def get_embeddings(model, tokenizer, texts, device, batch_size=8):
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=256, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def part1_extractive_qa(encoder_name, device):
    print("\n" + "="*60)
    print("Part 1: Extractive QA")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    try:
        model = AutoModelForQuestionAnswering.from_pretrained(encoder_name)
    except Exception as e:
        print(f"Could not load QA model from {encoder_name}: {e}")
        print("Skipping Part 1 training, showing architecture only.")
        return

    train_ds, val_ds = load_qa_dataset(tokenizer)

    output_dir = os.path.join(os.path.dirname(__file__), "qa_results")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="no",
        no_cuda=not torch.cuda.is_available(),
        use_mps_device=torch.backends.mps.is_available(),
        report_to="none",
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    model.eval()
    predictions, references = [], []
    for item in SYNTHETIC_QA:
        enc = tokenizer(item["question"], item["context"], return_tensors="pt", truncation=True, max_length=384)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        start = out.start_logits.argmax().item()
        end = out.end_logits.argmax().item()
        if end < start:
            end = start
        tokens = enc["input_ids"][0][start:end+1]
        pred = tokenizer.decode(tokens, skip_special_tokens=True)
        predictions.append(pred)
        references.append(item["answer"])

    em, f1 = compute_em_f1(predictions, references)
    print(f"\nExtractive QA Results:")
    print(f"  Exact Match: {em:.4f}")
    print(f"  F1 Score:    {f1:.4f}")


def part2_rag_demo(encoder_name, device):
    print("\n" + "="*60)
    print("Part 2: RAG Demo")
    print("="*60)

    import faiss
    from transformers import AutoModelForCausalLM

    tokenizer_enc = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).to(device)

    kb_texts = [text for _, text in KNOWLEDGE_BASE]
    kb_titles = [title for title, _ in KNOWLEDGE_BASE]

    print("Building knowledge base embeddings...")
    kb_embeddings = get_embeddings(encoder, tokenizer_enc, kb_texts, device)
    kb_normed = kb_embeddings / (np.linalg.norm(kb_embeddings, axis=1, keepdims=True) + 1e-9)

    dim = kb_normed.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(kb_normed.astype(np.float32))

    gen_model_name = "rinna/japanese-gpt2-medium"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer as AT
        gen_tokenizer = AT.from_pretrained(gen_model_name)
        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
        can_generate = True
        print(f"Loaded generator: {gen_model_name}")
    except Exception as e:
        print(f"Could not load generator {gen_model_name}: {e}")
        can_generate = False

    qa_demos = [
        ("東京はどんな都市ですか？", "東京"),
        ("富士山の高さを教えてください。", "富士山"),
        ("新幹線はいつ開業しましたか？", "新幹線"),
    ]

    rag_results = []
    for query, expected_topic in qa_demos:
        q_emb = get_embeddings(encoder, tokenizer_enc, [query], device)
        q_normed = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        scores, indices = index.search(q_normed.astype(np.float32), 3)

        retrieved = [(kb_titles[i], kb_texts[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
        context = "\n".join([f"[{t}] {txt}" for t, txt, _ in retrieved])
        prompt = f"以下の文書を参考に質問に答えてください。\n文書:{context}\n質問:{query}\n回答:"

        print(f"\nQuery: {query}")
        print("Retrieved passages:")
        for rank, (title, _, score) in enumerate(retrieved, 1):
            hit = "✓" if title == expected_topic else " "
            print(f"  {rank}. [{score:.4f}] {title} {hit}")

        if can_generate:
            inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = gen_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=gen_tokenizer.eos_token_id,
                )
            generated = gen_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"Generated answer: {generated.strip()}")
        else:
            print(f"[Generator not available — prompt ready for external LLM]")

        correct_in_top3 = any(t == expected_topic for t, _, _ in retrieved)
        rag_results.append({"query": query, "correct_in_top3": correct_in_top3})

    precision = np.mean([r["correct_in_top3"] for r in rag_results])
    print(f"\nRetrieval Precision@3: {precision:.4f}")


def main():
    encoder_name = "cl-tohoku/bert-base-japanese-v3"
    try:
        AutoTokenizer.from_pretrained(encoder_name)
    except Exception:
        encoder_name = "bert-base-multilingual-cased"
        print(f"Fallback to {encoder_name}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    part1_extractive_qa(encoder_name, device)
    part2_rag_demo(encoder_name, device)


if __name__ == "__main__":
    main()
