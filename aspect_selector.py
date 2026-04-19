import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class AspectSentenceSelector:
    """Per-product, per-aspect sentence selection via relevance + dedup + MMR."""

    ASPECTS = ["kargo", "paketleme", "fiyat_performans", "kalite", "genel_memnuniyet"]

    ASPECT_QUERIES = {
        "kargo":            "kargo hızı teslimat süresi paket geldi",
        "paketleme":        "paket ambalaj kutu hasar koruma",
        "fiyat_performans": "fiyat performans değer para eder mi",
        "kalite":           "ürün kalitesi malzeme sağlamlık dayanıklılık",
        "genel_memnuniyet": "genel memnuniyet tavsiye eder tekrar alır",
    }

    def __init__(
        self,
        model_name: str = "trmteb/turkish-embedding-model",
        device: str = "cpu",
        text_col: str = "sentence_clean",
        top_k_candidates: int = 100,
        top_k_final: int = 30,
        mmr_lambda: float = 0.7,
        dup_threshold: float = 0.85,
        min_tokens: int = 4,
        min_relevance: float = 0.1,
        batch_size: int = 32,
    ):
        self.device = device
        self.text_col = text_col
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final
        self.mmr_lambda = mmr_lambda
        self.dup_threshold = dup_threshold
        self.min_tokens = min_tokens
        self.min_relevance = min_relevance
        self.batch_size = batch_size
        self.pred_cols = {a: f"pred_{a}" for a in self.ASPECTS}

        print(f"Loading sentence transformer {model_name} (device={device})...")
        self.model = SentenceTransformer(model_name, device=device)
        self.aspect_vecs = {
            aspect: self.model.encode(
                query, normalize_embeddings=True, convert_to_numpy=True
            )
            for aspect, query in self.ASPECT_QUERIES.items()
        }
        print("Sentence transformer ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_noisy(self, text: str) -> bool:
        if not isinstance(text, str):
            return True
        tokens = text.strip().split()
        if len(tokens) < self.min_tokens:
            return True
        if len(text.strip("!?. \n\t")) == 0:
            return True
        return False

    @staticmethod
    def _mmr(
        candidate_vecs: np.ndarray,
        query_vec: np.ndarray,
        top_k: int,
        lam: float,
    ) -> list[int]:
        relevance = candidate_vecs @ query_vec
        selected: list[int] = []
        remaining = list(range(len(candidate_vecs)))

        while len(selected) < top_k and remaining:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_vecs = candidate_vecs[selected]
                best_score = -np.inf
                best = None
                for i in remaining:
                    rel = relevance[i]
                    max_sim = float((candidate_vecs[i] @ sel_vecs.T).max())
                    score = lam * rel - (1 - lam) * max_sim
                    if score > best_score:
                        best_score = score
                        best = i
            selected.append(best)
            remaining.remove(best)
        return selected

    def _add_product_id(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["product_id"] = out["product_url"].str.extract(r"/dp/([A-Z0-9]{10})")
        out["product_id"] = out["product_id"].fillna(
            out["product_title"].str.slice(0, 50).str.strip()
        )
        return out.reset_index(drop=True)

    def _drop_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_noisy"] = df[self.text_col].apply(self._is_noisy)
        print(f"Gürültülü cümle: {df['is_noisy'].sum()} / {len(df)}")
        return df[~df["is_noisy"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_product_id(labeled_df)
        df = self._drop_noise(df)
        if df.empty:
            return pd.DataFrame()

        print("Cümleler encode ediliyor...")
        all_embeddings = self.model.encode(
            df[self.text_col].tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        records = []
        for product_id, grp in df.groupby("product_id"):
            for aspect in self.ASPECTS:
                pred_col = self.pred_cols[aspect]
                if pred_col not in grp.columns:
                    continue

                subset = grp[grp[pred_col] == 1]
                if subset.empty:
                    continue

                idxs = subset.index.values
                candidate_vecs = all_embeddings[idxs]
                query_vec = self.aspect_vecs[aspect]

                # 1. Relevance filter
                relevance_scores = candidate_vecs @ query_vec
                relevance_mask = relevance_scores >= self.min_relevance
                if relevance_mask.sum() == 0:
                    continue
                idxs = idxs[relevance_mask]
                candidate_vecs = candidate_vecs[relevance_mask]
                relevance_scores = relevance_scores[relevance_mask]

                # 2. Near-duplicate removal (keep higher-relevance copy)
                order = np.argsort(relevance_scores)[::-1]
                keep_mask = np.ones(len(idxs), dtype=bool)
                for i_pos, i in enumerate(order):
                    if not keep_mask[i]:
                        continue
                    for j in order[i_pos + 1:]:
                        if not keep_mask[j]:
                            continue
                        sim = float(candidate_vecs[i] @ candidate_vecs[j])
                        if sim >= self.dup_threshold:
                            keep_mask[j] = False
                idxs = idxs[keep_mask]
                candidate_vecs = candidate_vecs[keep_mask]
                relevance_scores = relevance_scores[keep_mask]

                # 3. Top-K candidates → MMR
                top_local = np.argsort(relevance_scores)[::-1][: self.top_k_candidates]
                candidate_vecs_top = candidate_vecs[top_local]
                idxs_top = idxs[top_local]

                selected_local = self._mmr(
                    candidate_vecs=candidate_vecs_top,
                    query_vec=query_vec,
                    top_k=self.top_k_final,
                    lam=self.mmr_lambda,
                )

                for rank, local_i in enumerate(selected_local):
                    global_i = idxs_top[local_i]
                    row = df.loc[global_i]
                    rel_score = float(relevance_scores[top_local[local_i]])
                    records.append({
                        "product_id":      product_id,
                        "product_title":   row.get("product_title"),
                        "aspect":          aspect,
                        "sentence":        row[self.text_col],
                        "relevance_score": round(rel_score, 4),
                        "mmr_rank":        rank + 1,
                        "rating":          row.get("rating"),
                        "review_id":       row.get("review_id"),
                        "sent_idx":        row.get("sent_idx"),
                        "source":          row.get("source"),
                    })

        return pd.DataFrame(records)
