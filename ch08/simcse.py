import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SENTENCES = [
    "東京の天気は晴れです。",
    "今日は気温が高く、夏日となりました。",
    "明日は雨が降る予報が出ています。",
    "大阪では梅雨入りが発表されました。",
    "北海道は涼しく過ごしやすい気候です。",
    "台風が接近しており、警戒が必要です。",
    "日本経済は緩やかな回復を続けています。",
    "株式市場では日経平均が上昇しました。",
    "円相場はドルに対して円安傾向が続いています。",
    "政府は新たな経済対策を発表しました。",
    "電気自動車の普及が世界的に加速しています。",
    "再生可能エネルギーの導入が進んでいます。",
    "人工知能技術の発展が目覚ましい進歩を見せています。",
    "機械学習モデルの精度が向上しています。",
    "自然言語処理の研究が活発に行われています。",
    "深層学習によって画像認識の精度が大幅に向上しました。",
    "日本の人口減少が社会問題となっています。",
    "少子化対策として政府が支援策を拡充しました。",
    "高齢化社会において医療費の増大が課題です。",
    "介護人材の不足が深刻な問題になっています。",
    "東京オリンピックの施設が有効活用されています。",
    "スポーツ振興のための政策が推進されています。",
    "プロ野球シーズンが開幕しました。",
    "サッカー日本代表がアジア予選を突破しました。",
    "国内旅行需要が回復傾向にあります。",
    "観光地への訪問者数が増加しています。",
    "京都の古都は国内外から多くの観光客を集めています。",
    "富士山は世界遺産として登録されています。",
    "日本食は世界中で人気を博しています。",
    "ラーメンは日本を代表するグルメのひとつです。",
    "寿司の需要が海外でも拡大しています。",
    "和食文化がユネスコ無形文化遺産に登録されました。",
    "新幹線は日本の高速鉄道網の中心です。",
    "リニア中央新幹線の建設が進められています。",
    "公共交通機関の利用促進が環境政策の柱です。",
    "電気バスの導入が各地で始まっています。",
    "教育分野でのデジタル化が加速しています。",
    "GIGAスクール構想でタブレットが全国に配布されました。",
    "オンライン授業が普及し学習スタイルが変化しています。",
    "大学入試改革が実施されました。",
    "医療技術の進歩により難病治療が可能になっています。",
    "ゲノム医療の実用化が進んでいます。",
    "新薬の開発競争が国際的に激化しています。",
    "ワクチン接種率の向上が感染症対策に貢献しています。",
    "気候変動対策として脱炭素社会の実現が急務です。",
    "二酸化炭素排出削減の国際目標が設定されました。",
    "電力の脱炭素化に向けた取り組みが進んでいます。",
    "海洋プラスチック問題への対応が求められています。",
    "宇宙開発競争が新たな局面を迎えています。",
    "民間企業による宇宙旅行が現実のものとなりました。",
    "月面探査計画が各国で進められています。",
    "火星への有人探査計画が検討されています。",
    "サイバーセキュリティの重要性が高まっています。",
    "個人情報の保護に関する法律が改正されました。",
    "デジタル庁が設置され行政のデジタル化が推進されています。",
    "マイナンバーカードの普及が進んでいます。",
    "農業のスマート化が生産性向上に貢献しています。",
    "ドローンを活用した農薬散布が普及しています。",
    "食料安全保障の観点から国産農産物の振興が重要です。",
    "有機農業への転換が農業政策の方向性となっています。",
    "日本の製造業は高い技術力を誇っています。",
    "半導体産業の国内回帰が政策的に支援されています。",
    "中小企業の技術革新を支援する取り組みが行われています。",
    "スタートアップ企業への投資が増加しています。",
    "文化芸術への支援が文化政策の重要な柱です。",
    "伝統工芸の担い手不足が課題となっています。",
    "アニメやマンガが日本のソフトパワーを高めています。",
    "ゲーム産業は日本の重要なコンテンツ産業です。",
    "地方移住を促進するための支援策が充実しています。",
    "テレワークの普及により働き方が多様化しています。",
    "ワークライフバランスの改善が求められています。",
    "女性活躍推進のための施策が拡充されています。",
    "外国人労働者の受け入れ拡大が検討されています。",
    "多文化共生社会の実現に向けた取り組みが進んでいます。",
    "障害者の就労支援が強化されています。",
    "ユニバーサルデザインの導入が各地で進んでいます。",
    "防災意識の向上が国民的課題となっています。",
    "自然災害への備えとして避難訓練が実施されています。",
    "気象予報技術の精度向上が防災に貢献しています。",
    "洪水対策として河川の整備が進められています。",
    "東日本大震災からの復興が続けられています。",
    "被災地の産業振興が重要な政策課題です。",
    "コミュニティの再生が復興の鍵となっています。",
    "防潮堤の建設が海岸線防護に役立っています。",
    "図書館のデジタル化が進み電子書籍の貸し出しが始まりました。",
    "出版業界における電子書籍市場が拡大しています。",
    "読書推進運動が子どもたちの学力向上に寄与しています。",
    "公共図書館の役割が地域コミュニティで高まっています。",
    "環境教育の充実が次世代への責任として重視されています。",
    "持続可能な開発目標SDGsの達成に向けた活動が広がっています。",
    "企業のCSR活動が社会貢献として評価されています。",
    "ESG投資が金融市場で注目を集めています。",
    "消費者の意識変化が持続可能な消費行動を促しています。",
    "フードロス削減に向けた取り組みが各地で行われています。",
    "シェアリングエコノミーが新たなビジネスモデルとして定着しています。",
    "サブスクリプションサービスの普及が消費スタイルを変えています。",
    "キャッシュレス決済の普及が進んでいます。",
    "フィンテック企業が金融サービスに革新をもたらしています。",
    "暗号資産市場の動向が注目されています。",
    "ブロックチェーン技術の実用化が進んでいます。",
]


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


class SimCSEModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def mean_pool(self, hidden, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return self.mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        z1 = self.encode(input_ids, attention_mask, token_type_ids)
        z2 = self.encode(input_ids, attention_mask, token_type_ids)
        return z1, z2


def nt_xent_loss(z1, z2, temperature=0.05):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size),
    ]).to(z1.device)
    return F.cross_entropy(sim, labels)


def train(model, loader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            z1, z2 = model(input_ids, attention_mask, token_type_ids)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")


def build_faiss_index(embeddings):
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    index.add(normed.astype(np.float32))
    return index, normed


def get_embeddings(model, tokenizer, sentences, device, batch_size=32):
    model.eval()
    all_embs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(
            batch,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            token_type_ids = enc.get("token_type_ids")
            emb = model.encode(enc["input_ids"], enc["attention_mask"], token_type_ids)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def main():
    model_name = "cl-tohoku/bert-base-japanese-v3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        print(f"Loaded: {model_name}")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}, using bert-base-multilingual-cased")
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    sentences = SENTENCES
    dataset = SentenceDataset(sentences, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimCSEModel(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train(model, loader, optimizer, device, epochs=3)

    print("\nBuilding Faiss index...")
    embeddings = get_embeddings(model, tokenizer, sentences, device)
    index, normed_embs = build_faiss_index(embeddings)

    query = "東京の天気は"
    query_enc = tokenizer(
        [query],
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    query_enc = {k: v.to(device) for k, v in query_enc.items()}
    with torch.no_grad():
        token_type_ids = query_enc.get("token_type_ids")
        q_emb = model.encode(query_enc["input_ids"], query_enc["attention_mask"], token_type_ids)
    q_emb = q_emb.cpu().numpy()
    q_normed = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

    scores, indices = index.search(q_normed.astype(np.float32), 3)

    print(f"\nQuery: {query}")
    print("Top-3 similar sentences:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        print(f"  {rank}. [{score:.4f}] {sentences[idx]}")

    out_dir = os.path.dirname(__file__)
    results = {
        "query": query,
        "results": [
            {"rank": i+1, "score": float(scores[0][i]), "sentence": sentences[indices[0][i]]}
            for i in range(len(indices[0]))
        ]
    }
    import json
    with open(os.path.join(out_dir, "simcse_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_dir}/simcse_results.json")


if __name__ == "__main__":
    main()
