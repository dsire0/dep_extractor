"""
tests/test_conflict_detector.py
==================================
Unit tests for dep_extractor.analysis.conflict_detector.
"""

from __future__ import annotations

import pytest
from dep_extractor.analysis.conflict_detector import detect_static_conflicts
from dep_extractor.models import _specs_conflict


# ---------------------------------------------------------------------------
# _specs_conflict (lives in models.py, re-exported as package public API)
# ---------------------------------------------------------------------------

class TestSpecsConflict:
    """Low-level version spec conflict detection."""

    def test_non_overlapping_pins_conflict(self):
        assert _specs_conflict("==1.0", "==2.0") is True

    def test_same_pin_no_conflict(self):
        assert _specs_conflict("==1.0", "==1.0") is False

    def test_overlapping_range_no_conflict(self):
        # Both include the range [2.0, 3.0) — no conflict
        assert _specs_conflict(">=1.0,<3.0", ">=2.0,<4.0") is False

    def test_non_overlapping_range_no_conflict_as_currently_implemented(self):
        # NOTE: the current models._specs_conflict is a conservative heuristic.
        # It only flags hard pin conflicts (==x vs ==y). Range conflicts like
        # <1.0 vs >=2.0 return False (no conflict flagged) by design until the
        # SAT solver is invoked. This test documents that behavior.
        assert _specs_conflict("<1.0", ">=2.0") is False

    def test_empty_spec_no_conflict(self):
        assert _specs_conflict("", ">=1.0") is False

    def test_both_empty_no_conflict(self):
        assert _specs_conflict("", "") is False


# ---------------------------------------------------------------------------
# detect_static_conflicts
# ---------------------------------------------------------------------------

def _o(pkg: str, spec: str, node: str) -> dict:
    """Create an origin dict matching ScanResult.origin_map format."""
    return {"name": pkg, "spec": spec, "node": node, "raw": f"{pkg}{spec}"}


class TestDetectStaticConflicts:
    """Integration-level conflict detection from origin_map."""

    def test_no_conflicts_returns_empty(self):
        origin_map = {
            "torch": [
                _o("torch", ">=2.0", "node_a"),
                _o("torch", ">=1.0", "node_b"),
            ]
        }
        conflicts = detect_static_conflicts(origin_map)
        assert conflicts == []

    def test_pin_conflict_detected(self):
        origin_map = {
            "torch": [
                _o("torch", "==2.0.0", "node_a"),
                _o("torch", "==2.1.0", "node_b"),
            ]
        }
        conflicts = detect_static_conflicts(origin_map)
        assert len(conflicts) == 1
        assert conflicts[0].package == "torch"

    def test_multiple_packages_only_pin_conflicts_flagged(self):
        origin_map = {
            "numpy": [
                _o("numpy", ">=1.20", "node_a"),
                _o("numpy", ">=1.15", "node_b"),
            ],
            "torch": [
                _o("torch", "==2.0.0", "node_a"),
                _o("torch", "==2.1.0", "node_b"),
            ],
        }
        conflicts = detect_static_conflicts(origin_map)
        assert len(conflicts) == 1
        assert conflicts[0].package == "torch"

    def test_single_occurrence_no_conflict(self):
        origin_map = {
            "torch": [_o("torch", "==2.0.0", "only_node")]
        }
        conflicts = detect_static_conflicts(origin_map)
        assert conflicts == []

    def test_empty_origin_map_no_conflict(self):
        assert detect_static_conflicts({}) == []
