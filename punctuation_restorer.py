import re
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class PunctuationRestorer:
    """Turkish punctuation restorer wrapping a HuggingFace token-classification model."""

    DEFAULT_MODEL = "uygarkurt/electra-restore-punctuation-turkish"

    _PUNCT_CHARS = {",", ".", "?", "!", ";", ":"}
    _PUNCT_MAP = {
        "COMMA": ",", "PERIOD": ".", "QUESTION": "?",
        "EXCLAMATION": "!", "COLON": ":", "SEMICOLON": ";", "O": "",
    }
    _PREFER_FULL = ("PERIOD", "QUESTION", "EXCLAMATION", "COMMA", "COLON", "SEMICOLON")
    _PREFER_SENT = ("PERIOD", "QUESTION", "EXCLAMATION")

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        print(f"Loading punctuation model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="none",
            device=device,
        )
        print(f"Punctuation model ready (device={'cuda' if device == 0 else 'cpu'}).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_by_words(text: str, max_words: int = 160) -> list[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    @classmethod
    def _apply_punct(cls, text: str, pred, min_score: float, prefer: tuple[str, ...]) -> str:
        prefer_rank = {lab: i for i, lab in enumerate(prefer)}

        groups = []
        cur = None
        for p in pred:
            w = p["word"]
            start, end = int(p["start"]), int(p["end"])
            lab = p["entity"]
            score = float(p["score"])
            is_cont = w.startswith("##")

            if cur is None or (not is_cont and start > cur["end"]):
                cur = {"start": start, "end": end, "best_lab": "O", "best_score": -1.0}
                groups.append(cur)
            else:
                cur["end"] = max(cur["end"], end)

            if lab != "O" and score >= min_score:
                if (score > cur["best_score"]) or (
                    score == cur["best_score"]
                    and prefer_rank.get(lab, 999) < prefer_rank.get(cur["best_lab"], 999)
                ):
                    cur["best_lab"] = lab
                    cur["best_score"] = score

        inserts = []
        for g in groups:
            ch = cls._PUNCT_MAP.get(g["best_lab"], "")
            if not ch:
                continue
            pos = g["end"]
            if pos < len(text) and text[pos:pos + 1] in cls._PUNCT_CHARS:
                continue
            inserts.append((pos, ch))

        if not inserts:
            return text

        inserts.sort()
        out_parts = []
        last = 0
        for pos, ch in inserts:
            if pos < last:
                continue
            out_parts.append(text[last:pos])
            out_parts.append(ch)
            last = pos
        out_parts.append(text[last:])
        return "".join(out_parts)

    @staticmethod
    def _comma_to_period(restored: str, min_words: int = 12) -> str:
        parts = [p.strip() for p in restored.split(",") if p.strip()]
        out, buf, wc_sum = [], [], 0

        for p in parts:
            buf.append(p)
            wc_sum += len(p.split())
            if wc_sum >= min_words:
                sent = " ".join(buf).strip()
                if not re.search(r"[.!?]$", sent):
                    sent += "."
                out.append(sent)
                buf, wc_sum = [], 0

        if buf:
            sent = " ".join(buf).strip()
            if not re.search(r"[.!?]$", sent):
                sent += "."
            out.append(sent)

        return " ".join(out)

    def _prefer_labels(self, sentence_mode: bool) -> tuple[str, ...]:
        return self._PREFER_SENT if sentence_mode else self._PREFER_FULL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restore_text(
        self,
        text: str,
        max_words_per_chunk: int = 160,
        sentence_mode: bool = False,
        comma_to_period: bool = True,
        comma_min_words: int = 12,
        min_score: float = 0.55,
    ) -> str:
        if text is None:
            return ""
        text = str(text).strip()
        if not text:
            return ""

        prefer = self._prefer_labels(sentence_mode)
        chunks = self._chunk_by_words(text, max_words=max_words_per_chunk)
        restored_chunks = []

        for ch in chunks:
            pred = self.pipe(ch)
            restored = self._apply_punct(ch, pred, min_score=min_score, prefer=prefer)
            if comma_to_period:
                restored = self._comma_to_period(restored, min_words=comma_min_words)
            restored_chunks.append(restored)

        joined = " ".join(restored_chunks).strip()
        joined = re.sub(r"\s+([,.!?;:])", r"\1", joined)
        joined = re.sub(r"\s{2,}", " ", joined)
        return joined

    def restore_column(
        self,
        df: pd.DataFrame,
        col: str = "body",
        out_col: str | None = None,
        max_words_per_chunk: int = 160,
        sentence_mode: bool = False,
        comma_to_period: bool = True,
        comma_min_words: int = 12,
        min_score: float = 0.55,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        out = df.copy()
        target = out_col or col
        texts = out[col].fillna("").astype(str).tolist()
        prefer = self._prefer_labels(sentence_mode)

        tasks = []
        for i, t in enumerate(texts):
            t = t.strip()
            if not t:
                continue
            for ci, ch in enumerate(self._chunk_by_words(t, max_words=max_words_per_chunk)):
                tasks.append((i, ci, ch))

        if not tasks:
            out[target] = [""] * len(texts)
            return out

        chunk_texts = [t[2] for t in tasks]

        with torch.inference_mode():
            preds = self.pipe(chunk_texts, batch_size=batch_size)

        per_row = defaultdict(list)
        for (row_i, ci, ch_text), pred in zip(tasks, preds):
            restored = self._apply_punct(ch_text, pred, min_score=min_score, prefer=prefer)
            if comma_to_period:
                restored = self._comma_to_period(restored, min_words=comma_min_words)
            restored = re.sub(r"\s+([,.!?;:])", r"\1", restored)
            restored = re.sub(r"\s{2,}", " ", restored).strip()
            per_row[row_i].append((ci, restored))

        out_vals = [""] * len(texts)
        for i in range(len(texts)):
            chunks = per_row.get(i)
            if not chunks:
                continue
            chunks.sort(key=lambda x: x[0])
            out_vals[i] = " ".join(c for _, c in chunks).strip()

        out[target] = out_vals
        return out
