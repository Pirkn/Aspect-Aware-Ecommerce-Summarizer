import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AspectLabelingPipeline:
    """Multi-label aspect classifier over cleaned sentences. CPU-friendly."""

    def __init__(
        self,
        model_dir: str = "aspect-labeler",
        text_col: str = "sentence_clean",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        self.text_col = text_col
        self.batch_size = batch_size
        self.device = torch.device(device)

        with open(os.path.join(model_dir, "inference_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.labels = cfg["labels"]
        self.max_len = cfg["max_len"]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for i, l in enumerate(self.labels)}

        with open(os.path.join(model_dir, "thresholds.json"), "r", encoding="utf-8") as f:
            th = json.load(f)
        self.thresholds = np.array(th["thresholds"], dtype=np.float32)

        print(f"Loading aspect labeler from {model_dir} (device={device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("Aspect labeler ready.")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.text_col] = out[self.text_col].fillna("").astype(str).str.strip()
        return out

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        texts = df[self.text_col].tolist()
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)

        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.detach().cpu().numpy()
            chunks.append(_sigmoid(logits))
        return np.vstack(chunks).astype(np.float32)

    def predict(
        self,
        df: pd.DataFrame,
        return_probs: bool = True,
        return_binary: bool = True,
    ) -> pd.DataFrame:
        df_out = self._prepare_dataframe(df)
        probs = self.predict_proba(df_out)

        y_pred = (probs >= self.thresholds.reshape(1, -1)).astype(int)

        df_out["predicted_labels"] = [
            [self.labels[j] for j, v in enumerate(row) if v == 1] for row in y_pred
        ]

        if return_binary:
            for j, lab in enumerate(self.labels):
                df_out[f"pred_{lab}"] = y_pred[:, j]

        if return_probs:
            for j, lab in enumerate(self.labels):
                df_out[f"prob_{lab}"] = probs[:, j]

        return df_out
