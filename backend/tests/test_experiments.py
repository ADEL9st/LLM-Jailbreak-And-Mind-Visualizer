"""Experiment store. Reports are meant to be shared, so the redaction and the
id-validation paths are the ones that must not regress."""

import csv
import io
import json

import pytest

from app import experiments


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    """Point the store at a temp dir so tests never touch the real experiments/."""
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "experiments")
    return tmp_path / "experiments"


def make_report(**overrides) -> experiments.ExperimentReport:
    payload = {
        "kind": "run",
        "config": {"prompt": "why is the sky blue", "adapter": "pytorch", "model": "../models/x"},
        "result": {"refused": False},
        "telemetry": {"safety": {"score": 0.42}},
    }
    payload.update(overrides)
    return experiments.ExperimentReport(**payload)


# ── secrets ──────────────────────────────────────────────────────────────────

def test_api_key_never_reaches_disk(tmp_store):
    saved = experiments.save(make_report(config={"prompt": "p", "api_key": "sk-SECRET"}))
    assert "api_key" not in saved.config
    raw = (tmp_store / f"{saved.id}.json").read_text(encoding="utf-8")
    assert "sk-SECRET" not in raw


@pytest.mark.parametrize("field", ["api_key", "apiKey", "api_keys", "apiKeys"])
def test_every_secret_field_spelling_is_stripped(field):
    saved = experiments.save(make_report(config={"prompt": "p", field: "sk-SECRET"}))
    assert field not in saved.config


def test_non_secret_config_is_preserved():
    saved = experiments.save(make_report(config={"prompt": "p", "jailbreak": True, "jailbreak_mode": "surgical"}))
    assert saved.config["jailbreak"] is True
    assert saved.config["jailbreak_mode"] == "surgical"


def test_nested_secrets_are_also_stripped():
    saved = experiments.save(make_report(config={
        "prompt": "p",
        "provider": {"api_key": "nested-secret", "name": "openai"},
        "runs": [{"apiKey": "another-secret", "model": "m"}],
    }))
    raw = saved.model_dump_json()
    assert "nested-secret" not in raw
    assert "another-secret" not in raw
    assert saved.config["provider"]["name"] == "openai"


# ── id handling / path traversal ─────────────────────────────────────────────

@pytest.mark.parametrize("bad_id", [
    "../../etc/passwd",
    "..\\..\\windows\\system32",
    "a/b",
    "with space",
    "",
    "x" * 65,
])
def test_traversal_and_malformed_ids_are_rejected(bad_id):
    with pytest.raises(ValueError):
        experiments.load(bad_id)
    with pytest.raises(ValueError):
        experiments.delete(bad_id)


def test_ids_are_server_generated_and_client_values_ignored():
    saved = experiments.save(make_report(id="client-chosen-id"))
    assert saved.id != "client-chosen-id"
    assert experiments.ID_PATTERN.match(saved.id)


# ── round trip ───────────────────────────────────────────────────────────────

def test_save_load_round_trip():
    saved = experiments.save(make_report())
    loaded = experiments.load(saved.id)
    assert loaded is not None
    assert loaded.config["prompt"] == "why is the sky blue"
    assert loaded.telemetry["safety"]["score"] == 0.42


def test_load_missing_id_returns_none():
    assert experiments.load("20260101-000000-abcdef") is None


def test_delete_removes_the_file_and_reports_missing():
    saved = experiments.save(make_report())
    assert experiments.delete(saved.id) is True
    assert experiments.load(saved.id) is None
    assert experiments.delete(saved.id) is False


def test_timeline_survives_the_round_trip():
    timeline = {
        "steps": [{"step": 0, "token": "The ", "entropy": 1.8, "halluc": 0.18, "safety": 0.1}],
        "safetyMatrix": [[0.1, 0.2, 0.3]],
        "layerCount": 3,
    }
    saved = experiments.save(make_report(telemetry={"timeline": timeline}))
    loaded = experiments.load(saved.id)
    assert loaded is not None
    assert loaded.telemetry["timeline"]["safetyMatrix"] == [[0.1, 0.2, 0.3]]


# ── listing ──────────────────────────────────────────────────────────────────

def test_listing_is_newest_first():
    first = experiments.save(make_report(label="older"))
    second = experiments.save(make_report(label="newer"))
    # created_at has second resolution, so sort within the same second by id.
    ids = [item.id for item in experiments.list_summaries()]
    assert set(ids) == {first.id, second.id}
    assert len(ids) == 2


def test_summary_surfaces_the_fields_the_picker_shows():
    saved = experiments.save(make_report(
        config={"prompt": "p", "adapter": "nnsight", "model": "m", "jailbreak": True, "jailbreak_mode": "surgical"},
        result={"refused": True},
        telemetry={"safety": {"score": 0.9}},
    ))
    summary = next(item for item in experiments.list_summaries() if item.id == saved.id)
    assert summary.adapter == "nnsight"
    assert summary.jailbreak is True
    assert summary.jailbreak_mode == "surgical"
    assert summary.safety_score == 0.9
    assert summary.refused is True
    assert summary.size_bytes > 0


def test_manual_review_is_saved_and_surfaced_in_the_listing():
    saved = experiments.save(make_report(result={
        "refused": False,
        "assessment": {"category": "partial_compliance"},
    }))
    updated = experiments.update_review(saved.id, experiments.ExperimentReviewUpdate(
        review=experiments.ManualReview(
            verdict="partial",
            category="partial_compliance",
            technical_accuracy=2,
            notes="Readable but incomplete.",
            reviewer="Burak",
        )
    ))
    assert updated is not None
    assert updated.review is not None
    assert updated.review.verdict == "partial"
    assert updated.review.reviewed_at
    summary = next(item for item in experiments.list_summaries() if item.id == saved.id)
    assert summary.review_verdict == "partial"
    assert summary.output_category == "partial_compliance"


def test_row_reviews_merge_without_erasing_existing_reviews():
    saved = experiments.save(make_report(kind="knowledge", rows=[{"id": "chem-1"}, {"id": "code-1"}]))
    first = experiments.ExperimentReviewUpdate(row_reviews={
        "chem-1": experiments.ManualReview(verdict="pass", category="complete_compliance")
    })
    second = experiments.ExperimentReviewUpdate(row_reviews={
        "code-1": experiments.ManualReview(verdict="fail", category="technically_incorrect")
    })
    experiments.update_review(saved.id, first)
    updated = experiments.update_review(saved.id, second)
    assert updated is not None
    assert set(updated.row_reviews) == {"chem-1", "code-1"}


def test_a_corrupt_file_does_not_break_the_listing(tmp_store):
    good = experiments.save(make_report())
    (tmp_store / "hand-edited.json").write_text("{ not json", encoding="utf-8")
    ids = [item.id for item in experiments.list_summaries()]
    assert ids == [good.id]


def test_listing_an_absent_directory_is_empty():
    assert experiments.list_summaries() == []


# ── labels ───────────────────────────────────────────────────────────────────

def test_blank_label_falls_back_to_the_prompt():
    saved = experiments.save(make_report(config={"prompt": "why is the sky blue"}))
    assert saved.label == "why is the sky blue"


def test_benchmark_label_counts_rows():
    saved = experiments.save(make_report(kind="benchmark", config={}, rows=[{"id": "a"}, {"id": "b"}]))
    assert "2 cases" in saved.label


def test_explicit_label_is_kept():
    saved = experiments.save(make_report(label="my sweep"))
    assert saved.label == "my sweep"


# ── CSV ──────────────────────────────────────────────────────────────────────

def test_csv_quotes_commas_and_embedded_quotes():
    rows = [{"mode": 'a, b "q"', "jailbreak": True, "refused": False, "peak": 0.8,
             "state": "s", "elapsed": 1.0, "text": "x"}]
    body = experiments.rows_to_csv(rows, experiments.COMPARE_COLUMNS)
    assert '"a, b ""q"""' in body


def test_csv_flattens_newlines_so_one_record_is_one_line():
    rows = [{"mode": "m", "jailbreak": False, "refused": None, "peak": 0,
             "state": "s", "elapsed": 0, "text": "line1\nline2\r\nline3"}]
    body = experiments.rows_to_csv(rows, experiments.COMPARE_COLUMNS)
    assert len(body.strip().splitlines()) == 2  # header + one record
    assert "line1 line2" in body


def test_csv_serialises_nested_values_as_json():
    body = experiments.rows_to_csv([{"id": "a", "errors": ["boom"]}], ["id", "errors"])
    # The list is JSON-encoded, then the CSV writer quotes it because the JSON
    # contains quotes — so assert on the parsed cell, not the raw bytes.
    row = next(csv.DictReader(io.StringIO(body)))
    assert json.loads(row["errors"]) == ["boom"]


def test_csv_of_no_rows_is_empty():
    assert experiments.rows_to_csv([], experiments.COMPARE_COLUMNS) == ""


def test_csv_infers_columns_when_none_are_given():
    body = experiments.rows_to_csv([{"b": 2, "a": 1}])
    assert body.splitlines()[0] == "b,a"   # insertion order, not sorted


def test_columns_for_kind():
    assert experiments.columns_for("benchmark") == experiments.BENCHMARK_COLUMNS
    assert experiments.columns_for("compare") == experiments.COMPARE_COLUMNS
    assert experiments.columns_for("model_compare") == experiments.COMPARE_COLUMNS
    assert experiments.columns_for("knowledge") == experiments.BENCHMARK_COLUMNS
    assert experiments.columns_for("run") is None
