import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class EvidenceRetriever:
    """Traces student summaries back to aspect-selected sentences and full reviews.

    For each (product_id, aspect) pair with a non-empty summary, encodes the
    summary and scores it against sentences belonging to that aspect in the
    MMR corpus (restricted to the same product_id). Keeps top-k hits whose
    cosine similarity >= tau and joins them to reviews for full-review text.
    """

    ASPECTS = ["kargo", "paketleme", "fiyat_performans", "kalite", "genel_memnuniyet"]
    EVIDENCE_COLUMNS = [
        "product_id",
        "product_title",
        "aspect",
        "summary_source",
        "summary_text",
        "evidence_rank",
        "similarity",
        "matched_sentence",
        "matched_sent_idx",
        "review_id",
        "rating",
        "mmr_rank",
        "relevance_score",
        "full_review_text",
    ]

    def __init__(
        self,
        model_name: str = "trmteb/turkish-embedding-model",
        device: str = "cpu",
        top_k: int = 15,
        tau: float = 0.25,
        batch_size: int = 32,
        candidate_text_col: str = "sentence",
        summary_col: str = "student_summary",
        summary_source_label: str = "student",
        review_text_col: str = "clean_body",
        sentence_model: SentenceTransformer | None = None,
    ):
        self.device = device
        self.top_k = top_k
        self.tau = tau
        self.batch_size = batch_size
        self.candidate_text_col = candidate_text_col
        self.summary_col = summary_col
        self.summary_source_label = summary_source_label
        self.review_text_col = review_text_col

        if sentence_model is not None:
            self.model = sentence_model
        else:
            print(f"Loading sentence transformer {model_name} for evidence retrieval (device={device})...")
            self.model = SentenceTransformer(model_name, device=device)
            print("Evidence retriever ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _build_review_lookup(self, reviews_df: pd.DataFrame | None) -> dict:
        if reviews_df is None or len(reviews_df) == 0:
            return {}
        cols = list(reviews_df.columns)
        if "review_id" not in cols or self.review_text_col not in cols:
            return {}

        # reviews.csv can have two columns named 'review_id' (preprocessor
        # reset_index collides with the scraper's Amazon review id). The
        # first occurrence is the integer id that MMR rows reference.
        rid_series = reviews_df.iloc[:, cols.index("review_id")]
        text_series = reviews_df[self.review_text_col]
        if isinstance(text_series, pd.DataFrame):
            text_series = text_series.iloc[:, 0]

        lookup: dict = {}
        for rid, text in zip(rid_series.tolist(), text_series.tolist()):
            if rid is None or (isinstance(rid, float) and pd.isna(rid)):
                continue
            if rid not in lookup:
                lookup[rid] = text
        return lookup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        summaries_df: pd.DataFrame,
        mmr_df: pd.DataFrame,
        reviews_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if summaries_df is None or summaries_df.empty or mmr_df is None or mmr_df.empty:
            return pd.DataFrame(columns=self.EVIDENCE_COLUMNS)

        s = summaries_df.copy()
        s[self.summary_col] = s[self.summary_col].astype("string")
        s = s[s[self.summary_col].notna() & s[self.summary_col].str.strip().ne("")]
        s = s.reset_index(drop=True)
        if s.empty:
            print("No non-empty summaries to retrieve evidence for.")
            return pd.DataFrame(columns=self.EVIDENCE_COLUMNS)

        candidates = mmr_df.dropna(subset=[self.candidate_text_col]).reset_index(drop=True)
        if candidates.empty:
            return pd.DataFrame(columns=self.EVIDENCE_COLUMNS)

        print(f"Encoding {len(candidates)} candidate sentences...")
        cand_vecs = self._encode(candidates[self.candidate_text_col].astype(str).tolist())

        print(f"Encoding {len(s)} summary sentences...")
        summary_vecs = self._encode(s[self.summary_col].str.strip().tolist())

        review_lookup = self._build_review_lookup(reviews_df)

        records = []
        for i, srow in s.iterrows():
            product_id = srow["product_id"]
            aspect = srow["aspect"]
            summary_text = srow[self.summary_col].strip()

            mask = (candidates["product_id"] == product_id) & (candidates["aspect"] == aspect)
            if not mask.any():
                continue

            sub_idx = np.where(mask.values)[0]
            sub_vecs = cand_vecs[sub_idx]
            sims = sub_vecs @ summary_vecs[i]

            order = np.argsort(sims)[::-1]
            kept = [j for j in order if sims[j] >= self.tau][: self.top_k]
            if not kept:
                continue

            for rank, local_j in enumerate(kept, start=1):
                crow = candidates.iloc[sub_idx[local_j]]
                rid = crow.get("review_id")
                records.append({
                    "product_id":       product_id,
                    "product_title":    crow.get("product_title") or srow.get("product_title"),
                    "aspect":           aspect,
                    "summary_source":   self.summary_source_label,
                    "summary_text":     summary_text,
                    "evidence_rank":    rank,
                    "similarity":       round(float(sims[local_j]), 4),
                    "matched_sentence": crow.get(self.candidate_text_col),
                    "matched_sent_idx": crow.get("sent_idx"),
                    "review_id":        rid,
                    "rating":           crow.get("rating"),
                    "mmr_rank":         crow.get("mmr_rank"),
                    "relevance_score":  crow.get("relevance_score"),
                    "full_review_text": review_lookup.get(rid),
                })

        return pd.DataFrame(records, columns=self.EVIDENCE_COLUMNS)

    def retrieve_and_save(
        self,
        summaries_df: pd.DataFrame,
        mmr_df: pd.DataFrame,
        reviews_df: pd.DataFrame | None,
        output_path: str,
    ) -> pd.DataFrame:
        out = self.retrieve(summaries_df, mmr_df, reviews_df)
        out.to_csv(output_path, index=False)
        print(f"Saved evidence to {output_path} ({out.shape})")
        return out
