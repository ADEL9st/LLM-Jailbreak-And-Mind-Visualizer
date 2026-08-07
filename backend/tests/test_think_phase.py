"""ThinkPhaseTracker splits a CoT model's stream into <think> and answer spans
and averages layer activity per phase."""

from app.analysis.think_phase import ThinkPhaseTracker


def feed_tokens(tracker: ThinkPhaseTracker, tokens, activities=None):
    for step, token in enumerate(tokens):
        tracker.feed(step, token, activities or [0.1, 0.2, 0.3])


def test_no_think_block_yields_no_summary():
    tracker = ThinkPhaseTracker()
    feed_tokens(tracker, ["The ", "answer ", "is ", "56."])
    assert tracker.summary() is None
    assert tracker.has_think_tokens is False
    assert tracker.phase == "answer"


def test_phase_flips_into_think_and_back():
    tracker = ThinkPhaseTracker()
    tracker.feed(0, "<think>", [0.1])
    assert tracker.phase == "think"
    tracker.feed(1, "reasoning", [0.1])
    assert tracker.phase == "think"
    tracker.feed(2, "</think>", [0.1])
    assert tracker.phase == "answer"


def test_summary_counts_steps_per_phase():
    tracker = ThinkPhaseTracker()
    feed_tokens(tracker, ["<think>", "a", "b", "c", "</think>", "x", "y"])
    summary = tracker.summary()
    assert summary is not None
    assert summary["has_think"] is True
    assert summary["think_steps"] == 4      # <think> a b c
    assert summary["answer_steps"] == 3     # </think> x y
    assert summary["think_steps"] + summary["answer_steps"] == 7


def test_tags_split_across_tokens_are_still_detected():
    """A tokenizer may emit "<th" + "ink>"; the tracker buffers, so it must
    still see the tag."""
    tracker = ThinkPhaseTracker()
    tracker.feed(0, "<th", [0.1])
    tracker.feed(1, "ink>", [0.1])
    assert tracker.phase == "think"
    tracker.feed(2, "</th", [0.1])
    tracker.feed(3, "ink>", [0.1])
    assert tracker.phase == "answer"
    assert tracker.has_think_tokens is True


def test_delta_is_think_average_minus_answer_average():
    tracker = ThinkPhaseTracker()
    # Layer 0 is hot while thinking, layer 1 while answering.
    tracker.feed(0, "<think>", [1.0, 0.0])
    tracker.feed(1, "reason", [1.0, 0.0])
    tracker.feed(2, "</think>", [0.0, 1.0])
    tracker.feed(3, "answer", [0.0, 1.0])

    summary = tracker.summary()
    assert summary is not None
    assert summary["think_avg"] == [1.0, 0.0]
    assert summary["answer_avg"] == [0.0, 1.0]
    assert summary["delta"] == [1.0, -1.0]
    # The layer that is hotter while thinking must rank first.
    assert summary["dominant_think_layers"][0] == 0


def test_unclosed_think_block_still_summarises():
    """Generation can hit the token cap mid-thought; the trailing span must be
    closed by summary() rather than dropped."""
    tracker = ThinkPhaseTracker()
    feed_tokens(tracker, ["<think>", "still", "reasoning"])
    summary = tracker.summary()
    assert summary is not None
    assert summary["think_steps"] == 3
    assert summary["answer_steps"] == 0
    assert summary["answer_avg"] == [0.0, 0.0, 0.0]


def test_spans_cover_every_step_exactly_once():
    tracker = ThinkPhaseTracker()
    feed_tokens(tracker, ["a", "<think>", "b", "c", "</think>", "d"])
    summary = tracker.summary()
    assert summary is not None
    assert sum(span["steps"] for span in summary["spans"]) == 6
    assert [span["phase"] for span in summary["spans"]] == ["answer", "think", "answer"]


def test_long_run_between_tags_does_not_break_detection():
    """The buffer is trimmed to 16 chars once it passes 32 — a closing tag after
    a long stretch of text must still register."""
    tracker = ThinkPhaseTracker()
    tracker.feed(0, "<think>", [0.1])
    for step in range(1, 40):
        tracker.feed(step, "reasoning words ", [0.1])
    tracker.feed(40, "</think>", [0.1])
    assert tracker.phase == "answer"
