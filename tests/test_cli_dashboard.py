from sentient_evals.cli import _compact_failure_reason


def test_compact_failure_reason_empty():
    assert _compact_failure_reason("") == "failed"


def test_compact_failure_reason_single_line():
    assert _compact_failure_reason("Token validation failed") == "Token validation failed"


def test_compact_failure_reason_multiline_truncates():
    raw = "Command exited with code 127\nline2\nline3\nline4"
    got = _compact_failure_reason(raw, max_chars=60)
    assert got.startswith("Command exited with code 127")
    assert "(+3 lines)" in got
    assert len(got) <= 60
