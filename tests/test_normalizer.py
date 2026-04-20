"""
tests/test_normalizer.py
==========================
Unit tests for dep_extractor.core.normalizer.RequirementNormalizer.
"""

from __future__ import annotations

import pytest
from dep_extractor.core.normalizer import RequirementNormalizer


@pytest.fixture
def norm() -> RequirementNormalizer:
    return RequirementNormalizer()


class TestNormalizeStandard:
    def test_simple_name(self, norm):
        r = norm.normalize("torch", source_node="node_a")
        assert r is not None
        assert r.name == "torch"
        assert r.specifier == ""
        assert not r.is_url

    def test_pinned_version(self, norm):
        r = norm.normalize("torch==2.1.0", source_node="node_a")
        assert r is not None
        assert r.name == "torch"
        assert "2.1.0" in r.specifier

    def test_constrained_version(self, norm):
        r = norm.normalize("numpy>=1.20,<2.0", source_node="node_a")
        assert r is not None
        assert r.name == "numpy"

    def test_extras(self, norm):
        r = norm.normalize("requests[security]>=2.28", source_node="node_a")
        assert r is not None
        assert r.name == "requests"

    def test_whitespace_stripped(self, norm):
        r = norm.normalize("  transformers  ", source_node="node_a")
        assert r is not None
        assert r.name == "transformers"

    def test_comment_stripped(self, norm):
        r = norm.normalize("diffusers  # latest", source_node="node_a")
        assert r is not None
        assert r.name == "diffusers"

    def test_empty_line_returns_none(self, norm):
        assert norm.normalize("", source_node="node_a") is None

    def test_comment_line_returns_none(self, norm):
        assert norm.normalize("# this is a comment", source_node="node_a") is None

    def test_directive_returns_none(self, norm):
        assert norm.normalize("--extra-index-url https://example.com", source_node="node_a") is None


class TestNormalizeURL:
    def test_github_url(self, norm):
        r = norm.normalize(
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.6/flash_attn-2.3.6+cu118torch2.0.0-cp310-cp310-linux_x86_64.whl",
            source_node="node_a",
        )
        assert r is not None
        assert r.is_url

    def test_local_whl_path(self, norm):
        r = norm.normalize("./flash_attn-2.3.6-cp310-none-any.whl", source_node="node_a")
        # May be treated as a local path or URL depending on implementation
        assert r is not None


class TestNameNormalization:
    def test_case_insensitive(self, norm):
        r1 = norm.normalize("Pillow", source_node="node_a")
        r2 = norm.normalize("pillow", source_node="node_a")
        assert r1 is not None and r2 is not None
        assert r1.name == r2.name
