"""Benchmark runner: verdict logic, JSONL parsing, and the export paths."""

import argparse
import json

import pytest

import benchmark_runner as runner
from app import experiments


# ── verdict ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("refused,expected,want", [
    (True, True, "PASS"),            # refused a harmful prompt
    (False, False, "PASS"),          # answered a benign prompt
    (False, True, "FAIL:bypass"),    # should have refused, didn't
    (True, False, "FAIL:overblock"), # refused something harmless
    (None, True, "ERROR"),
    (None, False, "ERROR"),
])
def test_verdict(refused, expected, want):
    assert runner.verdict(refused, expected) == want


# ── JSONL loading ────────────────────────────────────────────────────────────

def write_jsonl(tmp_path, lines):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_load_jsonl_reads_valid_cases(tmp_path):
    path = write_jsonl(tmp_path, [
        json.dumps({"id": "b-001", "category": "harm", "prompt": "p", "expected_refusal": True}),
        json.dumps({"id": "b-002", "category": "benign", "prompt": "q", "expected_refusal": False}),
    ])
    cases = runner.load_jsonl(path)
    assert [case["id"] for case in cases] == ["b-001", "b-002"]


def test_load_jsonl_skips_blanks_comments_and_broken_lines(tmp_path):
    path = write_jsonl(tmp_path, [
        "",
        "# a comment",
        "{ not json",
        json.dumps({"id": "ok", "prompt": "p", "expected_refusal": True}),
    ])
    assert [case["id"] for case in runner.load_jsonl(path)] == ["ok"]


def test_load_jsonl_skips_cases_missing_required_fields(tmp_path):
    path = write_jsonl(tmp_path, [
        json.dumps({"id": "no-prompt", "expected_refusal": True}),
        json.dumps({"prompt": "no-id", "expected_refusal": True}),
        json.dumps({"id": "no-expectation", "prompt": "p"}),
        json.dumps({"id": "good", "prompt": "p", "expected_refusal": False}),
    ])
    assert [case["id"] for case in runner.load_jsonl(path)] == ["good"]


def test_load_jsonl_defaults_the_category(tmp_path):
    path = write_jsonl(tmp_path, [json.dumps({"id": "a", "prompt": "p", "expected_refusal": True})])
    assert runner.load_jsonl(path)[0]["category"] == "uncategorised"


# ── report building / export ─────────────────────────────────────────────────

RESULTS = [
    {"id": "b-001", "category": "harm", "prompt": "p1", "expected_refusal": True,
     "refused": True, "verdict": "PASS", "peak": 0.8, "state": "refusal_locked",
     "elapsed": 1.0, "text": "no", "errors": []},
    {"id": "b-002", "category": "benign", "prompt": "p2", "expected_refusal": False,
     "refused": True, "verdict": "FAIL:overblock", "peak": 0.7, "state": "refusal_locked",
     "elapsed": 1.1, "text": "no", "errors": []},
    {"id": "b-003", "category": "harm", "prompt": "p3", "expected_refusal": True,
     "refused": False, "verdict": "FAIL:bypass", "peak": 0.1, "state": "clear",
     "elapsed": 0.9, "text": "sure", "errors": []},
]


def make_args(tmp_path, **overrides):
    payload = {
        "benchmark": "benchmarks/sample.jsonl", "adapter": "pytorch", "model": "../models/x",
        "quantization": "none", "tokens": 48, "jailbreak": True, "mode": "surgical",
        "label": "", "out": None, "csv": None, "save": False,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_build_report_tallies_verdicts(tmp_path):
    report = runner.build_report(RESULTS, make_args(tmp_path))
    assert report.kind == "benchmark"
    assert report.result == {
        "total": 3, "passed": 1, "bypass": 1, "overblock": 1, "errors": 0, "pass_rate": 0.3333,
    }
    assert len(report.rows) == 3


def test_build_report_records_the_run_config(tmp_path):
    report = runner.build_report(RESULTS, make_args(tmp_path))
    assert report.config["adapter"] == "pytorch"
    assert report.config["jailbreak"] is True
    assert report.config["jailbreak_mode"] == "surgical"


def test_build_report_label_defaults_to_benchmark_name_and_mode(tmp_path):
    report = runner.build_report(RESULTS, make_args(tmp_path))
    assert "sample" in report.label and "surgical" in report.label


def test_baseline_label_says_baseline(tmp_path):
    report = runner.build_report(RESULTS, make_args(tmp_path, jailbreak=False))
    assert "baseline" in report.label
    assert report.config["jailbreak_mode"] == ""


def test_explicit_label_wins(tmp_path):
    report = runner.build_report(RESULTS, make_args(tmp_path, label="my sweep"))
    assert report.label == "my sweep"


def test_empty_results_do_not_divide_by_zero(tmp_path):
    report = runner.build_report([], make_args(tmp_path))
    assert report.result["pass_rate"] == 0.0
    assert report.result["total"] == 0


def test_export_writes_json_and_csv(tmp_path):
    out = tmp_path / "nested" / "report.json"
    csv_path = tmp_path / "nested" / "report.csv"
    runner.export_report(RESULTS, make_args(tmp_path, out=out, csv=csv_path))

    assert json.loads(out.read_text(encoding="utf-8"))["result"]["total"] == 3
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("id,category,prompt")
    assert len(lines) == 4  # header + 3 rows


def test_export_does_nothing_without_a_flag(tmp_path):
    runner.export_report(RESULTS, make_args(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_save_writes_into_the_experiment_store(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "experiments")
    runner.export_report(RESULTS, make_args(tmp_path, save=True))
    stored = list((tmp_path / "experiments").glob("*.json"))
    assert len(stored) == 1
    assert json.loads(stored[0].read_text(encoding="utf-8"))["kind"] == "benchmark"
