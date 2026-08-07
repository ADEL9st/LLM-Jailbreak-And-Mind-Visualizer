"""Persistent experiment reports.

A "report" is a frozen snapshot of one piece of work the dashboard did — a single
white-box run, a batch benchmark, or a jailbreak-mode comparison — stored as a
self-contained JSON file so the run can be reopened, diffed or shipped alongside
a write-up without re-running the model.

Storage is one file per report under ``<project>/experiments``. Flat files (not a
database) on purpose: a report is meant to be copied into a paper repo, attached
to an issue, or diffed in git by hand.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Reports are keyed by a server-generated id. Client-supplied ids are never
# trusted for path building; every lookup re-validates against this pattern so a
# crafted id can't escape EXPERIMENTS_DIR.
ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{1,64}$")

REPORT_VERSION = 2

ExperimentKind = Literal["run", "benchmark", "compare", "model_compare", "knowledge", "sweep"]
ManualVerdict = Literal["unreviewed", "pass", "partial", "fail", "inconclusive"]
ReviewCategory = Literal[
    "hard_refusal", "soft_refusal", "safe_redirect", "partial_compliance",
    "complete_compliance", "technically_incorrect", "degenerate", "empty", "truncated",
    "inconclusive", "other",
]

# Config fields that must never reach disk. API keys arrive in the run request
# and would otherwise be copied verbatim into a file users are encouraged to
# share.
SECRET_FIELDS = ("api_key", "apiKey", "api_keys", "apiKeys")


class ManualReview(BaseModel):
    verdict: ManualVerdict = "unreviewed"
    category: ReviewCategory = "other"
    technical_accuracy: int | None = Field(default=None, ge=0, le=5)
    notes: str = Field(default="", max_length=4000)
    reviewer: str = Field(default="", max_length=120)
    reviewed_at: str = ""


class ExperimentReviewUpdate(BaseModel):
    review: ManualReview | None = None
    row_reviews: dict[str, ManualReview] = Field(default_factory=dict)


class ExperimentReport(BaseModel):
    """A saved experiment. `id` and `created_at` are assigned by the server."""

    id: str = ""
    created_at: str = ""
    version: int = REPORT_VERSION
    kind: ExperimentKind = "run"
    label: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=4000)
    # Free-form by design: the telemetry payload tracks whatever the adapters
    # emit, and pinning a schema here would mean touching this file every time a
    # panel gains a field.
    config: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    review: ManualReview | None = None
    row_reviews: dict[str, ManualReview] = Field(default_factory=dict)


class ExperimentSummary(BaseModel):
    """The listing row — everything the picker needs without loading telemetry."""

    id: str
    created_at: str
    kind: ExperimentKind
    label: str
    adapter: str = ""
    model: str = ""
    prompt: str = ""
    jailbreak: bool = False
    jailbreak_mode: str = ""
    safety_score: float | None = None
    refused: bool | None = None
    row_count: int = 0
    size_bytes: int = 0
    review_verdict: ManualVerdict = "unreviewed"
    output_category: str = ""


def _redact(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove credentials from shareable experiment snapshots."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in SECRET_FIELDS}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(config)


def _new_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _path_for(experiment_id: str) -> Path:
    if not ID_PATTERN.match(experiment_id):
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    return EXPERIMENTS_DIR / f"{experiment_id}.json"


def _default_label(report: ExperimentReport) -> str:
    prompt = str(report.config.get("prompt") or report.result.get("prompt") or "").strip()
    if report.kind == "benchmark":
        return f"Benchmark · {len(report.rows)} cases"
    if report.kind == "compare":
        return f"Compare · {prompt[:40]}" if prompt else "Compare"
    if report.kind == "model_compare":
        return f"Model compare · {prompt[:40]}" if prompt else "Model compare"
    if report.kind == "knowledge":
        return f"Knowledge test · {len(report.rows)} cases"
    if report.kind == "sweep":
        return f"Parameter sweep · {len(report.rows)} runs"
    return prompt[:60] if prompt else "Run"


def save(report: ExperimentReport) -> ExperimentReport:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    report.id = _new_id()
    report.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report.version = REPORT_VERSION
    report.config = _redact(report.config)
    if not report.label.strip():
        report.label = _default_label(report)

    path = _path_for(report.id)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def load(experiment_id: str) -> ExperimentReport | None:
    path = _path_for(experiment_id)
    if not path.is_file():
        return None
    try:
        return ExperimentReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete(experiment_id: str) -> bool:
    path = _path_for(experiment_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def update_review(experiment_id: str, update: ExperimentReviewUpdate) -> ExperimentReport | None:
    report = load(experiment_id)
    if report is None:
        return None
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if update.review is not None:
        update.review.reviewed_at = stamp
        report.review = update.review
    for row_id, review in update.row_reviews.items():
        if len(row_id) > 120:
            raise ValueError("row review id is too long")
        review.reviewed_at = stamp
        report.row_reviews[row_id] = review
    report.version = REPORT_VERSION
    path = _path_for(experiment_id)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    temp_path.replace(path)
    return report


def _summarize(report: ExperimentReport, size_bytes: int) -> ExperimentSummary:
    safety = report.telemetry.get("safety") or {}
    score = safety.get("score") if isinstance(safety, dict) else None
    return ExperimentSummary(
        id=report.id,
        created_at=report.created_at,
        kind=report.kind,
        label=report.label,
        adapter=str(report.config.get("adapter") or ""),
        model=str(report.config.get("model") or ""),
        prompt=str(report.config.get("prompt") or "")[:160],
        jailbreak=bool(report.config.get("jailbreak") or False),
        jailbreak_mode=str(report.config.get("jailbreak_mode") or ""),
        safety_score=float(score) if isinstance(score, (int, float)) else None,
        refused=report.result.get("refused") if isinstance(report.result.get("refused"), bool) else None,
        row_count=len(report.rows),
        size_bytes=size_bytes,
        review_verdict=report.review.verdict if report.review is not None else "unreviewed",
        output_category=str(
            (report.result.get("assessment") or {}).get("category")
            if isinstance(report.result.get("assessment"), dict)
            else report.result.get("outcome") or ""
        ),
    )


def list_summaries() -> list[ExperimentSummary]:
    if not EXPERIMENTS_DIR.is_dir():
        return []
    summaries: list[ExperimentSummary] = []
    for path in EXPERIMENTS_DIR.glob("*.json"):
        try:
            report = ExperimentReport.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            # A hand-edited or truncated file shouldn't take down the listing.
            continue
        summaries.append(_summarize(report, path.stat().st_size))
    summaries.sort(key=lambda item: item.created_at, reverse=True)
    return summaries


# ── Tabular export ────────────────────────────────────────────────────────────

BENCHMARK_COLUMNS = ["id", "category", "prompt", "expected_refusal", "refused", "verdict", "assessment", "finish_reason", "peak", "state", "elapsed", "text"]
COMPARE_COLUMNS = ["mode", "jailbreak", "refused", "outcome", "assessment", "finish_reason", "peak", "state", "elapsed", "text"]


def rows_to_csv(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render report rows as CSV. Written by hand rather than via `csv` so the
    output is always CRLF-free UTF-8 regardless of platform."""
    import csv
    import io

    if not rows:
        return ""
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _flatten(row.get(key)) for key in columns})
    return buffer.getvalue()


def _flatten(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        # Keep every record on one CSV line so `wc -l` and spreadsheet imports
        # agree on the row count.
        return value.replace("\n", " ").replace("\r", " ")
    return value


def columns_for(kind: str) -> list[str] | None:
    if kind == "benchmark":
        return BENCHMARK_COLUMNS
    if kind in ("compare", "model_compare", "sweep"):
        return COMPARE_COLUMNS
    if kind == "knowledge":
        return BENCHMARK_COLUMNS
    return None
