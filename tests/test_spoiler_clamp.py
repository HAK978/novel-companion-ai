"""The spoiler guarantee.

Every MCP content tool resolves the reader's stored progress and clamps the
requested chapter to it, so a client cannot widen its own window by asking for
a later chapter. These are the highest-value tests in the suite: this rule is
the product's core promise.
"""

import pytest


@pytest.fixture
def at_chapter(mcp_server, monkeypatch):
    """Pin stored reading progress to a known chapter."""

    def _set(chapter):
        monkeypatch.setattr(mcp_server, "_reading_progress", lambda novel_id: chapter)

    return _set


def test_request_beyond_progress_is_clamped(mcp_server, at_chapter):
    at_chapter(500)
    # A client asking about chapter 3000 must not widen its window
    assert mcp_server._scoped_chapter(novel_id=1, requested=3000) == 500


def test_request_below_progress_is_honoured(mcp_server, at_chapter):
    at_chapter(500)
    # Narrowing is allowed: "as of chapter 100, what did X know?"
    assert mcp_server._scoped_chapter(novel_id=1, requested=100) == 100


def test_no_request_defaults_to_progress(mcp_server, at_chapter):
    at_chapter(742)
    assert mcp_server._scoped_chapter(novel_id=1, requested=None) == 742


def test_request_equal_to_progress(mcp_server, at_chapter):
    at_chapter(500)
    assert mcp_server._scoped_chapter(novel_id=1, requested=500) == 500


def test_unset_progress_returns_error_not_a_chapter(mcp_server, at_chapter):
    at_chapter(None)
    result = mcp_server._scoped_chapter(novel_id=1, requested=None)

    # Failing open would expose the whole novel, so this must not be an int
    assert isinstance(result, dict)
    assert "error" in result


def test_unset_progress_refuses_even_an_explicit_chapter(mcp_server, at_chapter):
    at_chapter(None)
    result = mcp_server._scoped_chapter(novel_id=1, requested=50)

    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.parametrize(
    "progress,requested,expected",
    [
        (1, 9999, 1),
        (1000, 1001, 1000),
        (1000, 999, 999),
        (3026, 3026, 3026),
    ],
)
def test_clamp_is_a_minimum(mcp_server, at_chapter, progress, requested, expected):
    at_chapter(progress)
    assert mcp_server._scoped_chapter(novel_id=1, requested=requested) == expected
