"""
FastAPI server for the Aspect-Aware Review Summarizer capstone demo.

Models load once at startup (matching main.py). Each POST /api/run spawns a
background worker that runs the live pipeline end-to-end and streams real
stage-by-stage metrics over SSE. Act 3 reads the persisted Outputs/ CSVs.

Run:
    uvicorn web.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from amazon_scrape import AmazonScraper
from aspect_labeler import AspectLabelingPipeline
from aspect_selector import AspectSentenceSelector
from aspect_summarizer import AspectSummarizer
from evidence_retriever import EvidenceRetriever
from preprocessor import ReviewPreprocessor
from prompt_builder import PromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("review-evaluator")

ASPECTS = ["kargo", "paketleme", "fiyat_performans", "kalite", "genel_memnuniyet"]
SUMMARY_COLUMNS = [
    "product_id",
    "product_title",
    "aspect",
    "student_summary",
    "teacher_summary",
]
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
DIAGNOSTIC_DATASETS = {
    "B0C3BV19Q3": "HyperX Cloud III Oyuncu Kulaklığı",
    "B0C8VK2Y95": "Hakiki Deri Kartlık",
    "B0CHXF71NK": "Apple Watch Series 9 Akıllı Saat",
}

SCRAPER_TIMEOUT_SEC = 45

# ----------------------------------------------------------------------
# Run registry
# ----------------------------------------------------------------------


@dataclass
class RunState:
    run_id: str
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    product_id: str | None = None
    product_title: str | None = None
    review_count: int | None = None
    result_ready: bool = False
    error: str | None = None
    started_at: float = field(default_factory=time.time)

    def emit(self, event: str, data: dict[str, Any]) -> None:
        """Thread-safe event push into the asyncio queue."""
        payload = {"event": event, "data": data}
        self.loop.call_soon_threadsafe(self.events.put_nowait, payload)


RUNS: dict[str, RunState] = {}


# ----------------------------------------------------------------------
# Model bundle (loaded at startup)
# ----------------------------------------------------------------------


@dataclass
class ModelBundle:
    preprocessor: ReviewPreprocessor
    aspect_labeler: AspectLabelingPipeline
    aspect_selector: AspectSentenceSelector
    prompt_builder: PromptBuilder
    summarizer: AspectSummarizer
    evidence_retriever: EvidenceRetriever
    scraper: AmazonScraper | None
    scraper_error: str | None = None


MODELS: ModelBundle | None = None


def _load_models() -> ModelBundle:
    log.info("=" * 60)
    log.info("Loading models...")
    log.info("=" * 60)

    preprocessor = ReviewPreprocessor()
    aspect_labeler = AspectLabelingPipeline(
        model_dir=str(PROJECT_ROOT / "aspect-labeler"),
        text_col="sentence_clean",
        batch_size=32,
        device="cpu",
    )
    aspect_selector = AspectSentenceSelector(device="cpu", batch_size=32)
    prompt_builder = PromptBuilder()
    summarizer = AspectSummarizer()
    evidence_retriever = EvidenceRetriever(
        sentence_model=aspect_selector.model,
        batch_size=32,
        top_k=15,
        tau=0.25,
    )

    scraper: AmazonScraper | None = None
    scraper_error: str | None = None
    try:
        scraper = AmazonScraper()
        log.info("Amazon scraper initialized. Log in to Amazon in the opened browser.")
    except Exception as e:
        scraper_error = f"Scraper init failed: {e}"
        log.warning(scraper_error)

    log.info("=" * 60)
    log.info("All models loaded.")
    log.info("=" * 60)

    return ModelBundle(
        preprocessor=preprocessor,
        aspect_labeler=aspect_labeler,
        aspect_selector=aspect_selector,
        prompt_builder=prompt_builder,
        summarizer=summarizer,
        evidence_retriever=evidence_retriever,
        scraper=scraper,
        scraper_error=scraper_error,
    )


# ----------------------------------------------------------------------
# Pipeline orchestration (runs in worker thread)
# ----------------------------------------------------------------------


def _run_scrape_with_timeout(scraper: AmazonScraper, user_input: str, timeout: int) -> pd.DataFrame:
    """Run scraper.scrape on a helper thread and enforce a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(scraper.scrape, user_input)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as te:
            raise TimeoutError(
                f"Scraper exceeded {timeout}s. Use 'CSV Yükle' to continue."
            ) from te


def _load_diagnostic_dataset(asin: str) -> pd.DataFrame:
    asin = (asin or "").strip().upper()
    if asin not in DIAGNOSTIC_DATASETS:
        raise RuntimeError(f"Tanımlı hızlı tanı ürünü değil: {asin}")

    csv_path = DATA_DIR / f"{asin}_data.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Hazır veri dosyası bulunamadı: {csv_path.name}")

    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Hazır veri CSV'si okunamadı: {e}") from e


def _aspect_counts(labeled_df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for a in ASPECTS:
        col = f"pred_{a}"
        out[a] = int(labeled_df[col].sum()) if col in labeled_df.columns else 0
    return out


def _mmr_counts(mmr_df: pd.DataFrame) -> dict[str, int]:
    if mmr_df.empty or "aspect" not in mmr_df.columns:
        return {a: 0 for a in ASPECTS}
    counts = mmr_df.groupby("aspect").size().to_dict()
    return {a: int(counts.get(a, 0)) for a in ASPECTS}


def _append_user_comment(df: pd.DataFrame, comment: str) -> pd.DataFrame:
    """Append a single user-supplied comment as an Amazon-shaped review row."""
    text = (comment or "").strip()
    if not text:
        return df

    tz = pytz.timezone("Europe/Istanbul")
    now = datetime.now(tz)

    def _first(col: str):
        if col in df.columns:
            vals = df[col].dropna()
            if not vals.empty:
                return vals.iloc[0]
        return None

    row = {col: None for col in df.columns}
    row.update({
        "review_id": f"user_comment_{int(now.timestamp())}",
        "asin": _first("asin"),
        "title": None,
        "rating": None,
        "body": text,
        "review_date": now.date().isoformat(),
        "product_title": _first("product_title"),
        "product_description": _first("product_description"),
        "product_url": _first("product_url"),
        "source": "user_input",
        "review_summary": None,
        "collected_at": now.strftime("%Y-%m-%d %H:%M"),
    })

    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def run_pipeline(
    run_id: str,
    user_input: str | None,
    uploaded_csv: bytes | None,
    user_comment: str | None = None,
    data_asin: str | None = None,
) -> None:
    """Blocking pipeline worker. Emits SSE events into the run's asyncio queue."""
    assert MODELS is not None
    state = RUNS[run_id]
    scraper = MODELS.scraper

    try:
        # --- Stage 1: Data loading (scrape, upload, or diagnostic CSV) ---
        state.emit("stage_start", {"stage": 1, "name": "data_loading"})

        if data_asin:
            df = _load_diagnostic_dataset(data_asin)
        elif uploaded_csv is not None:
            try:
                df = pd.read_csv(io.BytesIO(uploaded_csv))
            except Exception as e:
                raise RuntimeError(f"CSV okunamadı: {e}") from e
        else:
            if scraper is None:
                raise RuntimeError(
                    "Scraper yüklenmedi. CSV yükleyerek devam edin."
                )
            df = _run_scrape_with_timeout(scraper, user_input or "", SCRAPER_TIMEOUT_SEC)

        if df is None or df.empty:
            raise RuntimeError("Bu ürün için yorum bulunamadı.")

        if user_comment and user_comment.strip():
            df = _append_user_comment(df, user_comment)

        product_title = None
        product_id = None
        if "product_title" in df.columns:
            titles = df["product_title"].dropna().astype(str)
            if not titles.empty:
                product_title = titles.iloc[0]
        if "product_url" in df.columns:
            m = df["product_url"].astype(str).str.extract(r"/dp/([A-Z0-9]{10})")
            if not m.empty and m.iloc[:, 0].dropna().size:
                product_id = m.iloc[:, 0].dropna().iloc[0]
        if product_id is None and "asin" in df.columns:
            asins = df["asin"].dropna().astype(str)
            if not asins.empty:
                product_id = asins.iloc[0]

        state.product_id = product_id
        state.product_title = product_title
        state.review_count = len(df)

        df.to_csv(OUTPUTS_DIR / "output.csv", index=False)

        state.emit(
            "stage_complete",
            {
                "stage": 1,
                "metric": {"reviews": len(df)},
                "product": {
                    "id": product_id,
                    "title": product_title,
                    "review_count": len(df),
                },
            },
        )

        # --- Stage 2: Preprocessing / sentence splitting ---
        state.emit("stage_start", {"stage": 2, "name": "preprocessing"})
        reviews_df, sentences_df = MODELS.preprocessor.process(df)
        reviews_df.to_csv(OUTPUTS_DIR / "reviews.csv", index=False)
        sentences_df.to_csv(OUTPUTS_DIR / "sentences.csv", index=False)
        state.emit(
            "stage_complete",
            {"stage": 2, "metric": {"sentences": int(sentences_df.shape[0])}},
        )

        # --- Stage 3: Aspect labeling ---
        state.emit("stage_start", {"stage": 3, "name": "aspect_labeling"})
        labeled_df = MODELS.aspect_labeler.predict(
            sentences_df, return_probs=True, return_binary=True
        )
        labeled_df.to_csv(OUTPUTS_DIR / "labeled_sentences.csv", index=False)
        state.emit(
            "stage_complete",
            {"stage": 3, "metric": {"per_aspect": _aspect_counts(labeled_df)}},
        )

        # --- Stage 4: Evidence selection (MMR) ---
        state.emit("stage_start", {"stage": 4, "name": "evidence_selection"})
        mmr_df = MODELS.aspect_selector.select(labeled_df)
        if mmr_df is None or mmr_df.empty:
            raise RuntimeError("Yön bazlı cümle seçimi boş döndü.")
        mmr_df.to_csv(OUTPUTS_DIR / "aspect_sentences_mmr.csv", index=False)
        state.emit(
            "stage_complete",
            {"stage": 4, "metric": {"per_aspect": _mmr_counts(mmr_df)}},
        )

        # --- Stage 5: Summary generation + evidence retrieval ---
        state.emit("stage_start", {"stage": 5, "name": "summarization"})
        prompt_df = MODELS.prompt_builder.build(mmr_df)
        summaries_df = MODELS.summarizer.summarize_and_save(
            prompt_df, str(OUTPUTS_DIR / "aspect_summaries.csv")
        )
        MODELS.evidence_retriever.retrieve_and_save(
            summaries_df=summaries_df,
            mmr_df=mmr_df,
            reviews_df=reviews_df,
            output_path=str(OUTPUTS_DIR / "evidence.csv"),
        )
        n_summaries = int(summaries_df["student_summary"].notna().sum())
        state.emit(
            "stage_complete",
            {"stage": 5, "metric": {"summaries": n_summaries, "aspects": len(ASPECTS)}},
        )

        state.result_ready = True
        state.emit(
            "pipeline_complete",
            {"result_url": f"/api/results/{run_id}"},
        )
    except Exception as e:
        tb = traceback.format_exc()
        log.error("Pipeline failed for run %s: %s\n%s", run_id, e, tb)
        state.error = str(e)
        state.emit("pipeline_error", {"message": str(e)})


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS
    MODELS = _load_models()
    yield
    if MODELS and MODELS.scraper:
        try:
            MODELS.scraper.close()
        except Exception:
            pass


app = FastAPI(title="Review Evaluator", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.get("/")
async def root():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": MODELS is not None,
        "scraper_ready": MODELS is not None and MODELS.scraper is not None,
        "scraper_error": MODELS.scraper_error if MODELS else None,
    }


@app.post("/api/run")
async def create_run(
    input: str | None = Form(None),
    comment: str | None = Form(None),
    data_asin: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if not input and file is None and not data_asin:
        raise HTTPException(status_code=400, detail="Provide 'input', 'file', or 'data_asin'.")

    run_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_running_loop()
    state = RunState(run_id=run_id, loop=loop)
    RUNS[run_id] = state

    csv_bytes: bytes | None = None
    if file is not None:
        csv_bytes = await file.read()

    # Worker runs on default thread executor so it doesn't block the event loop.
    loop.run_in_executor(None, run_pipeline, run_id, input, csv_bytes, comment, data_asin)

    return {"run_id": run_id}


def _sse_format(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/run/{run_id}/events")
async def stream_events(run_id: str):
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    async def event_gen():
        # Initial ping so the client opens the connection cleanly.
        yield _sse_format("ready", {"run_id": run_id})
        while True:
            try:
                payload = await asyncio.wait_for(state.events.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield _sse_format(payload["event"], payload["data"])
            if payload["event"] in ("pipeline_complete", "pipeline_error"):
                break

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/results/{run_id}")
async def get_results(run_id: str):
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if not state.result_ready:
        raise HTTPException(status_code=409, detail="Pipeline not complete.")
    return JSONResponse(_load_results_payload(state))


def _load_results_payload(state: RunState) -> dict[str, Any]:
    summaries_csv = OUTPUTS_DIR / "aspect_summaries.csv"
    evidence_csv = OUTPUTS_DIR / "evidence.csv"

    summaries = []
    if summaries_csv.exists():
        sdf = _read_csv_or_empty(summaries_csv, SUMMARY_COLUMNS)
        for _, row in sdf.iterrows():
            summaries.append({
                "aspect": row["aspect"],
                "student": _clean_cell(row.get("student_summary")),
                "teacher": _clean_cell(row.get("teacher_summary")),
                "bertscore_f1": None,
            })

    evidence = []
    if evidence_csv.exists():
        edf = _read_csv_or_empty(evidence_csv, EvidenceRetriever.EVIDENCE_COLUMNS)
        for _, row in edf.iterrows():
            evidence.append({
                "aspect": row["aspect"],
                "rank": int(row["evidence_rank"]),
                "similarity": float(row["similarity"]),
                "mmr_rank": int(row["mmr_rank"]) if pd.notna(row.get("mmr_rank")) else None,
                "relevance": float(row["relevance_score"]) if pd.notna(row.get("relevance_score")) else None,
                "matched": _clean_cell(row.get("matched_sentence")),
                "review_id": _safe_int(row.get("review_id")),
                "rating": float(row["rating"]) if pd.notna(row.get("rating")) else None,
                "full": _clean_cell(row.get("full_review_text")) or _clean_cell(row.get("matched_sentence")),
            })

    product_title = state.product_title
    if product_title is None and summaries_csv.exists():
        sdf = _read_csv_or_empty(summaries_csv, SUMMARY_COLUMNS)
        if "product_title" in sdf.columns and not sdf["product_title"].dropna().empty:
            product_title = sdf["product_title"].dropna().iloc[0]

    product_id = state.product_id
    if product_id is None and summaries_csv.exists():
        sdf = _read_csv_or_empty(summaries_csv, SUMMARY_COLUMNS)
        if "product_id" in sdf.columns and not sdf["product_id"].dropna().empty:
            product_id = sdf["product_id"].dropna().iloc[0]

    return {
        "product": {
            "id": product_id,
            "title": product_title,
            "review_count": state.review_count,
        },
        "summaries": summaries,
        "evidence": evidence,
    }


def _read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _clean_cell(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def _safe_int(v) -> int | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None
