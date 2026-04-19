import pandas as pd


class PromptBuilder:
    """Turns MMR-selected aspect sentences into chat messages for the
    teacher (GPT-5.4, all aspects at once, JSON) and the fine-tuned
    student (per-aspect, single Turkish sentence)."""

    ASPECTS = ["kargo", "paketleme", "fiyat_performans", "kalite", "genel_memnuniyet"]

    ASPECT_MEANINGS = {
        "kargo": "Kargo süreci, teslimat hızı, kurye deneyimi ve gönderim performansı.",
        "paketleme": "Ambalaj kalitesi, kutu/paket durumu, ürünün korunması ve hasar görüp görmediği.",
        "fiyat_performans": "Ürünün fiyatına değip değmediği, para-değer dengesi ve fiyat avantajı.",
        "kalite": "Ürünün malzeme kalitesi, sağlamlığı, dayanıklılığı ve işçiliği.",
        "genel_memnuniyet": "Genel kullanıcı memnuniyeti, tavsiye edilip edilmediği ve tekrar satın alma eğilimi.",
    }

    TEACHER_SYSTEM_PROMPT = """You summarize Turkish e-commerce reviews by aspect.

Task:
- You will receive review evidence for one product, grouped by aspect.
- Write exactly one short Turkish sentence for each aspect.
- Use only the evidence for that aspect.

Aspect meanings:
- kargo: Kargo süreci, teslimat hızı, kurye deneyimi ve gönderim performansı.
- paketleme: Ambalaj kalitesi, kutu/paket durumu, ürünün korunması ve hasar görüp görmediği.
- fiyat_performans: Ürünün fiyatına değip değmediği, para-değer dengesi ve fiyat avantajı.
- kalite: Ürünün malzeme kalitesi, sağlamlığı, dayanıklılığı ve işçiliği.
- genel_memnuniyet: Genel kullanıcı memnuniyeti, tavsiye edilip edilmediği ve tekrar satın alma eğilimi.

Core rules:
1. Output language: Turkish.
2. Maximum one sentence per aspect.
3. If evidence is weak, missing, off-topic, or insufficient, return null for that aspect.
4. Use the aspect meaning as a strict boundary.
5. Do not make numerical or quantity-based claims.
6. Do not use words like "çoğu", "birçok", "bazı", "kimi", "genelde", "çoğunlukla", or similar frequency wording.
7. Do not automatically mention both positive and negative evidence.
8. If one side is clearly dominant, write only the dominant side.
9. Do not mention minority, isolated, or one-off opposing opinions.
10. Only write a mixed summary if the evidence is strongly and clearly balanced.
11. Do not use contrast words like "ama", "ancak", "buna karşın", "öte yandan" unless the evidence is strongly balanced.
12. Keep aspects semantically pure.
13. Do not copy review text verbatim.
14. Return only the structured output that matches the provided schema."""

    STUDENT_SYSTEM_PROMPT = """You summarize Turkish e-commerce reviews for one target aspect only.

Rules:
- Output language: Turkish.
- Write exactly one short sentence.
- Use only the provided evidence.
- Stay strictly inside the target aspect meaning.
- Do not make numerical or quantity-based claims.
- Do not use words like "çoğu", "birçok", "bazı", "kimi", "genelde", "çoğunlukla" or similar frequency wording.
- Do not mention both positive and negative evidence unless the evidence is strongly and clearly balanced.
- Ignore minor opposing evidence.
- Do not copy review text verbatim.
- If evidence is weak, missing, off-topic, or insufficient, output exactly: null
- Return only the answer text. No JSON. No explanation."""

    def __init__(
        self,
        top_n_per_aspect: int = 20,
        min_sentences_per_aspect: int = 1,
        max_sent_len: int = 300,
    ):
        self.top_n_per_aspect = top_n_per_aspect
        self.min_sentences_per_aspect = min_sentences_per_aspect
        self.max_sent_len = max_sent_len

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_sentence(self, text) -> str:
        if not isinstance(text, str):
            return ""
        text = " ".join(text.strip().split())
        if not text:
            return ""
        return text[: self.max_sent_len].strip()

    def _collect_product_evidence(self, mmr_df: pd.DataFrame) -> pd.DataFrame:
        required = {"product_id", "product_title", "aspect", "sentence", "mmr_rank", "relevance_score"}
        missing = required - set(mmr_df.columns)
        if missing:
            raise ValueError(f"Eksik kolonlar: {missing}")

        df = mmr_df.copy()
        df["sentence"] = df["sentence"].apply(self._clean_sentence)
        df = df[df["sentence"].str.len() > 0].copy()

        df = df.sort_values(
            ["product_id", "aspect", "mmr_rank", "relevance_score"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        rows = []
        for product_id, grp in df.groupby("product_id", sort=False):
            titles = grp["product_title"].dropna().astype(str).str.strip()
            titles = titles[titles != ""]
            product_title = titles.iloc[0] if not titles.empty else ""

            evidence = {}
            evidence_meta = {}
            total = 0

            for aspect in self.ASPECTS:
                sub = (
                    grp[grp["aspect"] == aspect]
                    .sort_values(["mmr_rank", "relevance_score"], ascending=[True, False])
                    .head(self.top_n_per_aspect)
                )
                sentences = [s for s in sub["sentence"].tolist() if isinstance(s, str) and s.strip()]

                if len(sentences) < self.min_sentences_per_aspect:
                    evidence[aspect] = []
                    evidence_meta[aspect] = {
                        "n_sentences": 0,
                        "avg_relevance": None,
                        "max_relevance": None,
                    }
                else:
                    evidence[aspect] = sentences
                    evidence_meta[aspect] = {
                        "n_sentences": len(sentences),
                        "avg_relevance": round(float(sub["relevance_score"].mean()), 4),
                        "max_relevance": round(float(sub["relevance_score"].max()), 4),
                    }
                    total += len(sentences)

            rows.append({
                "product_id": product_id,
                "product_title": product_title,
                "total_evidence_sentences": total,
                "evidence": evidence,
                "evidence_meta": evidence_meta,
            })

        return pd.DataFrame(rows)

    def _render_all_aspects_block(self, evidence: dict) -> str:
        lines = []
        for aspect in self.ASPECTS:
            lines.append(f"[{aspect}]")
            sents = evidence.get(aspect, [])
            if not sents:
                lines.append("-")
            else:
                for i, sent in enumerate(sents, 1):
                    lines.append(f"{i}. {sent}")
            lines.append("")
        return "\n".join(lines).strip()

    def _render_single_aspect_block(self, aspect: str, sentences: list) -> str:
        if not sentences:
            return "-"
        return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))

    # ------------------------------------------------------------------
    # Prompt constructors
    # ------------------------------------------------------------------

    def _teacher_user_text(self, product_id: str, product_title: str, evidence: dict) -> str:
        block = self._render_all_aspects_block(evidence)
        return (
            "Below are aspect-grouped user review sentences for a single product.\n\n"
            f"Product ID: {product_id}\n"
            f"Product title: {product_title}\n\n"
            "Aspect evidence:\n"
            f"{block}\n\n"
            "Important:\n"
            "- Summarize the dominant supported view for each aspect.\n"
            "- Ignore minor opposing evidence.\n"
            "- Do not balance the sentence artificially.\n"
            "- Do not use contrast words unless the evidence is strongly and genuinely mixed.\n"
            "- Keep each aspect strictly within its own meaning.\n"
            "- Return only valid JSON."
        )

    def _student_user_text(
        self, product_id: str, product_title: str, aspect: str, sentences: list
    ) -> str:
        block = self._render_single_aspect_block(aspect, sentences)
        return (
            f"Product ID: {product_id}\n"
            f"Product title: {product_title}\n\n"
            f"Target aspect: {aspect}\n"
            f"Aspect meaning: {self.ASPECT_MEANINGS[aspect]}\n\n"
            "Aspect evidence:\n"
            f"{block}"
        )

    def _build_teacher_messages(self, product_id: str, product_title: str, evidence: dict) -> list:
        return [
            {"role": "developer", "content": self.TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": self._teacher_user_text(product_id, product_title, evidence)},
        ]

    def _build_student_messages(
        self, product_id: str, product_title: str, evidence: dict
    ) -> dict:
        return {
            aspect: [
                {"role": "system", "content": self.STUDENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._student_user_text(
                        product_id, product_title, aspect, evidence.get(aspect, [])
                    ),
                },
            ]
            for aspect in self.ASPECTS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, mmr_df: pd.DataFrame) -> pd.DataFrame:
        """Input: mmr_df from AspectSentenceSelector.select().
        Output: per-product DataFrame with evidence + teacher/student messages."""
        product_df = self._collect_product_evidence(mmr_df)

        product_df["teacher_messages"] = product_df.apply(
            lambda r: self._build_teacher_messages(r["product_id"], r["product_title"], r["evidence"]),
            axis=1,
        )
        product_df["student_messages"] = product_df.apply(
            lambda r: self._build_student_messages(r["product_id"], r["product_title"], r["evidence"]),
            axis=1,
        )
        return product_df
