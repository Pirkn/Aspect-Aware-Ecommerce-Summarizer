import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AspectSummarizer:
    """Calls the teacher (GPT-5.4-mini) and the fine-tuned student (RunPod-hosted
    Llama) on prompt_df rows and returns a flat per-product comparison DataFrame."""

    ASPECTS = ["kargo", "paketleme", "fiyat_performans", "kalite", "genel_memnuniyet"]

    RESPONSE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "aspect_summary_output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "aspect_summaries": {
                        "type": "object",
                        "properties": {
                            "kargo": {"type": ["string", "null"]},
                            "paketleme": {"type": ["string", "null"]},
                            "fiyat_performans": {"type": ["string", "null"]},
                            "kalite": {"type": ["string", "null"]},
                            "genel_memnuniyet": {"type": ["string", "null"]},
                        },
                        "required": [
                            "kargo",
                            "paketleme",
                            "fiyat_performans",
                            "kalite",
                            "genel_memnuniyet",
                        ],
                        "additionalProperties": False,
                    }
                },
                "required": ["aspect_summaries"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        openai_api_key: str | None = None,
        runpod_api_key: str | None = None,
        teacher_model: str = "gpt-5.4-mini",
        student_model: str = "berkhan/llama31-8b-turkish-sft-v3-merged",
        student_base_url: str = "https://api.runpod.ai/v2/7ja9cfymaef7os/openai/v1",
        teacher_temperature: float = 0.2,
        student_temperature: float = 0.0,
        student_max_tokens: int = 128,
    ):
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        runpod_key = runpod_api_key or os.getenv("RUNPOD_TOKEN")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        if not runpod_key:
            raise ValueError("RUNPOD_TOKEN is not set.")

        self.teacher_client = OpenAI(api_key=openai_key)
        self.student_client = OpenAI(api_key=runpod_key, base_url=student_base_url)

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.student_max_tokens = student_max_tokens

    # ------------------------------------------------------------------
    # Model calls
    # ------------------------------------------------------------------

    def _call_teacher(self, messages: list) -> dict:
        resp = self.teacher_client.chat.completions.create(
            model=self.teacher_model,
            temperature=self.teacher_temperature,
            response_format=self.RESPONSE_SCHEMA,
            messages=messages,
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw).get("aspect_summaries", {}) or {}
        except (json.JSONDecodeError, AttributeError):
            return {}

    def _call_student(self, messages: list) -> str | None:
        resp = self.student_client.chat.completions.create(
            model=self.student_model,
            temperature=self.student_temperature,
            max_tokens=self.student_max_tokens,
            messages=messages,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.lower() == "null":
            return None
        return text

    # ------------------------------------------------------------------
    # Per-product orchestration
    # ------------------------------------------------------------------

    def _summarize_row(self, row: pd.Series) -> dict:
        product_id = row["product_id"]
        evidence = row.get("evidence", {}) or {}

        print(f"  → Teacher call: {product_id}")
        try:
            teacher_summaries = self._call_teacher(row["teacher_messages"])
        except Exception as e:
            print(f"    teacher failed: {e}")
            teacher_summaries = {}

        student_summaries = {}
        for aspect in self.ASPECTS:
            if not evidence.get(aspect):
                student_summaries[aspect] = None
                continue
            print(f"  → Student call: {product_id} / {aspect}")
            try:
                student_summaries[aspect] = self._call_student(
                    row["student_messages"][aspect]
                )
            except Exception as e:
                print(f"    student failed ({aspect}): {e}")
                student_summaries[aspect] = None

        flat = {
            "product_id": product_id,
            "product_title": row.get("product_title", ""),
        }
        for aspect in self.ASPECTS:
            flat[f"teacher_{aspect}"] = teacher_summaries.get(aspect)
            flat[f"student_{aspect}"] = student_summaries.get(aspect)
        return flat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, prompt_df: pd.DataFrame) -> pd.DataFrame:
        required = {"product_id", "product_title", "evidence", "teacher_messages", "student_messages"}
        missing = required - set(prompt_df.columns)
        if missing:
            raise ValueError(f"prompt_df missing columns: {missing}")

        rows = [self._summarize_row(row) for _, row in prompt_df.iterrows()]
        flat = rows[0]  # single-product assumption

        data = {
            "Fine Tuned LLama Model": [flat.get(f"student_{a}") for a in self.ASPECTS],
            "Gpt 5.4 Mini": [flat.get(f"teacher_{a}") for a in self.ASPECTS],
        }
        return pd.DataFrame(data, index=self.ASPECTS)

    def summarize_and_save(self, prompt_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
        result_df = self.summarize(prompt_df)
        result_df.to_csv(output_path, index=True)
        print(f"Saved summaries to {output_path} ({result_df.shape})")
        return result_df
