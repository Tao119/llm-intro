"""
embedding_clustering.py

Semantic Clustering of Medical Terms — pure NumPy.

100 medical terms across 8 clusters:
  循環器, 神経, 呼吸器, 消化器, 感染症, 内分泌, 腎臓, 整形

Steps:
  1. Generate cluster-based synthetic embeddings (related terms close together)
  2. Apply K-means clustering (pure NumPy)
  3. PCA to 2D for visualization
  4. Evaluate: Adjusted Rand Index, Silhouette Score
  5. Save cluster visualization as PNG
"""

import os
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "04-embedding-clustering")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Medical term clusters (8 clusters × ~12-13 terms = 100 total)
# ---------------------------------------------------------------------------

CLUSTERS = {
    "循環器": [
        "心筋梗塞", "心房細動", "狭心症", "大動脈解離", "心不全",
        "冠動脈疾患", "心タンポナーデ", "弁膜症", "高血圧", "動脈硬化",
        "大動脈弁狭窄症", "心室細動", "肺高血圧症",
    ],
    "神経": [
        "脳梗塞", "くも膜下出血", "てんかん", "パーキンソン病", "アルツハイマー病",
        "多発性硬化症", "ギランバレー症候群", "重症筋無力症", "頭蓋内圧亢進",
        "脳出血", "髄膜炎", "脊髄損傷", "末梢神経障害",
    ],
    "呼吸器": [
        "肺炎", "気管支喘息", "COPD", "肺塞栓症", "気胸",
        "肺癌", "胸膜炎", "過換気症候群", "睡眠時無呼吸症候群",
        "急性呼吸窮迫症候群", "間質性肺炎", "肺結核", "気管支炎",
    ],
    "消化器": [
        "胃潰瘍", "十二指腸潰瘍", "潰瘍性大腸炎", "クローン病", "肝硬変",
        "膵炎", "腸閉塞", "逆流性食道炎", "肝炎", "胆石症",
        "大腸癌", "虫垂炎", "腹膜炎",
    ],
    "感染症": [
        "敗血症", "蜂窩織炎", "髄膜炎菌感染症", "インフルエンザ", "COVID-19",
        "尿路感染症", "肺炎球菌感染症", "MRSA感染症", "腸チフス", "マラリア",
        "ヘルペスウイルス感染症", "カンジダ症",
    ],
    "内分泌": [
        "糖尿病", "甲状腺機能亢進症", "甲状腺機能低下症", "副腎不全",
        "クッシング症候群", "原発性アルドステロン症", "褐色細胞腫",
        "下垂体腺腫", "インスリノーマ", "バセドウ病",
        "橋本病", "副甲状腺機能亢進症",
    ],
    "腎臓": [
        "急性腎障害", "慢性腎臓病", "ネフローゼ症候群", "糸球体腎炎",
        "腎盂腎炎", "水腎症", "腎細胞癌", "IgA腎症",
        "多発性嚢胞腎", "透析関連合併症", "高カリウム血症", "低ナトリウム血症",
    ],
    "整形": [
        "骨折", "変形性関節症", "椎間板ヘルニア", "脊柱管狭窄症", "骨粗鬆症",
        "関節リウマチ", "痛風", "半月板損傷", "腱板断裂",
        "大腿骨頸部骨折", "腰椎分離症", "肩関節脱臼",
    ],
}

CLUSTER_NAMES = list(CLUSTERS.keys())
NUM_CLUSTERS  = len(CLUSTER_NAMES)

# Flatten to list of (term, cluster_id)
TERMS = []
TRUE_LABELS = []
for cid, (cluster_name, terms) in enumerate(CLUSTERS.items()):
    for term in terms:
        TERMS.append(term)
        TRUE_LABELS.append(cid)

TRUE_LABELS = np.array(TRUE_LABELS)
N_TERMS = len(TERMS)
assert N_TERMS == 100, f"Expected 100 terms, got {N_TERMS}"

# ---------------------------------------------------------------------------
# Synthetic embedding generation
# ---------------------------------------------------------------------------

EMB_DIM = 64

def generate_embeddings(rng, dim=EMB_DIM, cluster_std=0.3, inter_cluster_sep=3.0):
    """
    Generate cluster-based embeddings.
    Each cluster has a random center; terms are sampled around it.
    """
    # Random cluster centers (well separated)
    centers = rng.normal(0, inter_cluster_sep, size=(NUM_CLUSTERS, dim)).astype(np.float32)

    embeddings = np.zeros((N_TERMS, dim), dtype=np.float32)
    for i, (term, cid) in enumerate(zip(TERMS, TRUE_LABELS)):
        noise = rng.normal(0, cluster_std, size=dim).astype(np.float32)
        embeddings[i] = centers[cid] + noise

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    return embeddings, centers


# ---------------------------------------------------------------------------
# Pure NumPy K-means
# ---------------------------------------------------------------------------

def kmeans(X, k, n_iter=200, n_restarts=3, rng=None):
    """K-means clustering with multiple restarts, pure NumPy."""
    if rng is None:
        rng = np.random.default_rng(0)

    best_inertia = np.inf
    best_labels  = None
    best_centers = None

    for _ in range(n_restarts):
        # K-means++ initialization
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X])
            probs = dists / dists.sum()
            centers.append(X[rng.choice(len(X), p=probs)])
        centers = np.array(centers, dtype=np.float32)

        labels = np.zeros(len(X), dtype=np.int32)
        for iteration in range(n_iter):
            # Assignment step
            dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels) and iteration > 0:
                break
            labels = new_labels
            # Update step
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centers[c] = X[mask].mean(axis=0)

        inertia = sum(
            np.sum((X[labels == c] - centers[c]) ** 2)
            for c in range(k)
            if (labels == c).sum() > 0
        )
        if inertia < best_inertia:
            best_inertia  = inertia
            best_labels   = labels.copy()
            best_centers  = centers.copy()

    return best_labels, best_centers, best_inertia


# ---------------------------------------------------------------------------
# Pure NumPy PCA
# ---------------------------------------------------------------------------

def pca_2d(X):
    """Reduce to 2D via PCA (pure NumPy)."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]
    return X_centered @ top2


# ---------------------------------------------------------------------------
# Evaluation metrics (pure NumPy)
# ---------------------------------------------------------------------------

def adjusted_rand_index(true_labels, pred_labels):
    """Compute Adjusted Rand Index."""
    n = len(true_labels)
    classes = np.unique(true_labels)
    clusters = np.unique(pred_labels)

    # Contingency table
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for ci, c in enumerate(classes):
        for ki, k in enumerate(clusters):
            contingency[ci, ki] = np.sum((true_labels == c) & (pred_labels == k))

    # Sum of combinations C(n_ij, 2)
    sum_comb_c = sum(np.sum(contingency[i] * (contingency[i] - 1)) / 2
                     for i in range(len(classes)))
    a_i = contingency.sum(axis=1)
    b_j = contingency.sum(axis=0)
    sum_comb_a = np.sum(a_i * (a_i - 1)) / 2
    sum_comb_b = np.sum(b_j * (b_j - 1)) / 2
    comb_n = n * (n - 1) / 2

    expected_index = sum_comb_a * sum_comb_b / comb_n
    max_index = (sum_comb_a + sum_comb_b) / 2
    if max_index - expected_index < 1e-10:
        return 0.0
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return float(ari)


def silhouette_score(X, labels):
    """Compute mean silhouette coefficient (pure NumPy)."""
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    scores = []
    for i in range(n):
        cluster_i = labels[i]
        same_mask  = (labels == cluster_i) & (np.arange(n) != i)
        other_clusters = [c for c in unique_labels if c != cluster_i]

        if same_mask.sum() == 0:
            scores.append(0.0)
            continue

        a = np.mean(np.linalg.norm(X[same_mask] - X[i], axis=1))
        b_vals = []
        for c in other_clusters:
            other_mask = labels == c
            if other_mask.sum() > 0:
                b_vals.append(np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1)))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        s = (b - a) / denom if denom > 0 else 0.0
        scores.append(s)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_cluster_plot(X_2d, pred_labels, true_labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
            "#9b59b6", "#1abc9c", "#e67e22", "#34495e"
        ]
        markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        for ax, labels, title in zip(
            axes,
            [true_labels, pred_labels],
            ["True Labels (Ground Truth)", f"K-means Predicted (k={NUM_CLUSTERS})"]
        ):
            for cid in range(NUM_CLUSTERS):
                mask = labels == cid
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[cid], marker=markers[cid],
                    s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
                    label=CLUSTER_NAMES[cid] if ax == axes[0] else None,
                )
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Shared legend
        patches = [
            mpatches.Patch(color=colors[i], label=CLUSTER_NAMES[i])
            for i in range(NUM_CLUSTERS)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.05), fontsize=9)
        fig.suptitle("Medical Term Clustering (100 terms, 8 clusters) — PCA 2D",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Cluster plot saved to {out_path}")
    except ImportError:
        print("[Warning] matplotlib not available — skipping cluster plot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Medical Term Semantic Clustering (Pure NumPy)")
    print(f"  {N_TERMS} terms  |  {NUM_CLUSTERS} clusters  |  embed_dim={EMB_DIM}")
    print("=" * 60)

    rng = np.random.default_rng(42)
    embeddings, true_centers = generate_embeddings(rng, dim=EMB_DIM)
    print(f"Embeddings shape: {embeddings.shape}")

    # K-means
    print("\nRunning K-means (3 restarts, up to 200 iterations)...")
    pred_labels, pred_centers, inertia = kmeans(
        embeddings, k=NUM_CLUSTERS, n_iter=200, n_restarts=3, rng=np.random.default_rng(0)
    )
    print(f"Final inertia: {inertia:.4f}")

    # Evaluation
    ari = adjusted_rand_index(TRUE_LABELS, pred_labels)
    sil = silhouette_score(embeddings, pred_labels)
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print(f"Silhouette Score:          {sil:.4f}")

    # Per-cluster assignment
    print("\n--- Cluster Assignment Summary ---")
    for cid in range(NUM_CLUSTERS):
        true_terms  = [TERMS[i] for i in range(N_TERMS) if TRUE_LABELS[i] == cid]
        pred_terms  = [TERMS[i] for i in range(N_TERMS) if pred_labels[i] == cid]
        # Purity: most common true label in predicted cluster
        if pred_terms:
            pred_true_labs = [TRUE_LABELS[i] for i in range(N_TERMS) if pred_labels[i] == cid]
            counts = np.bincount(pred_true_labs, minlength=NUM_CLUSTERS)
            dominant_true = CLUSTER_NAMES[np.argmax(counts)]
            purity = counts.max() / len(pred_true_labs)
        else:
            dominant_true = "—"
            purity = 0.0
        print(f"  Pred cluster {cid}: {len(pred_terms):>3} terms  "
              f"dominant={dominant_true:<8}  purity={purity:.2f}")

    # PCA for visualization
    X_2d = pca_2d(embeddings)

    out_path = os.path.join(OUT_DIR, "medical_term_clusters.png")
    save_cluster_plot(X_2d, pred_labels, TRUE_LABELS, out_path)

    print(f"\nSummary:")
    print(f"  ARI={ari:.3f}  (1.0=perfect, 0.0=random)")
    print(f"  Silhouette={sil:.3f}  (closer to 1.0 = better separated clusters)")
