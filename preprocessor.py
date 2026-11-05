import re
import unicodedata

import pandas as pd
from zemberek import TurkishSentenceExtractor

from punctuation_restorer import PunctuationRestorer


class ReviewPreprocessor:
    """End-to-end review preprocessor: punctuation restore -> clean -> sentence split."""

    ASPECT_COLS = ["kargo", "paketleme", "fiyat_performans", "kalite", "genel_memnuniyet"]

    _URL_RE = re.compile(
        r"""(?ix)\b(
          https?://[^\s<>()]+ |
          www\.[^\s<>()]+ |
          [a-z0-9.-]+\.(com|net|org|io|co|tr|de|uk|me|app|shop|store|gov|edu)(/[^\s<>()]*)?
        )\b"""
    )
    _EMAIL_RE = re.compile(r"""(?ix)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b""")
    _INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NL_RE = re.compile(r"\n{3,}")
    _REPEAT_EXCL_RE = re.compile(r"!{2,}")
    _REPEAT_Q_RE = re.compile(r"\?{2,}")
    _ELLIPSIS_RE = re.compile(r"(?:\.{2,}|…{2,})")
    _SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:.!?…])")
    _SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{«“])\s+")
    _SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([\)\]\}»”])")
    _NO_SPACE_AFTER_PUNCT_RE = re.compile(
        r"([.!?…])([A-Za-zÇĞİÖŞÜçğıöşü0-9\(\[\{“\"'])"
    )
    _DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
    _THOUSAND_DOT_RE = re.compile(r"(?<=\d)\.(?=\d{3}\b)")
    _DOT_TOKEN = "<DOT>"

    def __init__(self, punctuation_restorer: PunctuationRestorer | None = None):
        self.punct = punctuation_restorer or PunctuationRestorer()
        print("Loading Turkish sentence extractor (zemberek)...")
        self.sentence_extractor = TurkishSentenceExtractor()
        print("Sentence extractor ready.")

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    @classmethod
    def _protect_number_dots(cls, text: str) -> str:
        text = cls._DECIMAL_DOT_RE.sub(cls._DOT_TOKEN, text)
        text = cls._THOUSAND_DOT_RE.sub(cls._DOT_TOKEN, text)
        return text

    @classmethod
    def _restore_number_dots(cls, text: str) -> str:
        return text.replace(cls._DOT_TOKEN, ".")

    @classmethod
    def clean_text(
        cls,
        text,
        mask_urls: bool = True,
        mask_emails: bool = True,
        normalize_unicode: bool = True,
        normalize_ellipsis: bool = True,
        reduce_repeated_punct: bool = True,
        fix_punct_spacing: bool = True,
    ) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        text = str(text).replace("\r\n", "\n").replace("\r", "\n")
        text = cls._INVISIBLE_RE.sub("", text)

        if normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if mask_urls:
            text = cls._URL_RE.sub("<URL>", text)
        if mask_emails:
            text = cls._EMAIL_RE.sub("<EMAIL>", text)
        if normalize_ellipsis:
            text = cls._ELLIPSIS_RE.sub("…", text)
        if reduce_repeated_punct:
            text = cls._REPEAT_EXCL_RE.sub("!", text)
            text = cls._REPEAT_Q_RE.sub("?", text)

        text = cls._MULTI_SPACE_RE.sub(" ", text)
        text = cls._MULTI_NL_RE.sub("\n\n", text)
        text = text.strip()

        if fix_punct_spacing and text:
            text = cls._protect_number_dots(text)
            text = cls._SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
            text = cls._SPACE_AFTER_OPEN_RE.sub(r"\1", text)
            text = cls._SPACE_BEFORE_CLOSE_RE.sub(r"\1", text)
            text = cls._NO_SPACE_AFTER_PUNCT_RE.sub(r"\1 \2", text)
            text = cls._restore_number_dots(text)
            text = text.strip()

        return text

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _restore_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.punct.restore_column(
            df,
            col="body",
            out_col="punctuation_restored_body",
            sentence_mode=False,
            comma_to_period=True,
            comma_min_words=6,
            min_score=0.55,
            max_words_per_chunk=120,
            batch_size=8,
        )

    def _add_clean_body(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        out["body_raw"] = out[text_col]
        out["clean_body"] = out[text_col].apply(self.clean_text)
        return out

    def _split_sentences(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        reviews_df = df.reset_index().rename(columns={"index": "review_id"})
        sentence_rows = []

        meta_cols = [
            "source", "rating", "review_date", "collected_at",
            "product_title", "product_url", "title", "review_summary",
        ]

        for row in reviews_df.itertuples(index=False):
            text = str(getattr(row, "clean_body", "") or "").strip()
            if not text:
                continue

            sents = [s.strip() for s in self.sentence_extractor.from_paragraph(text) if s and s.strip()]
            if not sents:
                continue

            base = {
                "review_id": row.review_id,
                "n_sentences": len(sents),
                "sentence_text_variant": "clean_body",
            }
            for c in meta_cols:
                base[c] = getattr(row, c, None)

            for i, sent in enumerate(sents, start=1):
                sentence_rows.append({**base, "sent_idx": i, "sentence_clean": sent})

        sentences_df = pd.DataFrame(sentence_rows)
        for c in self.ASPECT_COLS:
            if c in sentences_df.columns:
                sentences_df[c] = sentences_df[c].astype("Int8")

        return reviews_df, sentences_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full pipeline. Returns (reviews_df, sentences_df)."""
        df = self._restore_punctuation(df)
        df = self._add_clean_body(df, text_col="punctuation_restored_body")
        return self._split_sentences(df)
