"""
tests/test_report_writer.py
=============================
Tests for dep_extractor.output.report_writer.write_combined_requirements.
"""

from __future__ import annotations

import pytest
from dep_extractor.models import AuditReport, NormalizedRequirement
from dep_extractor.output.report_writer import write_combined_requirements
import platform


def _make_req(name: str, specifier: str = "") -> NormalizedRequirement:
    raw = f"{name}{specifier}" if specifier else name
    return NormalizedRequirement(
        name=name,
        raw=raw,
        specifier=specifier,
        is_url=False,
        is_heavy=False,
        source_node="test_node",
    )


def _make_audit(**overrides):
    from dep_extractor.models import AuditData
    return AuditData(**overrides) if overrides else AuditData()


class TestWriteCombinedRequirements:

    def test_file_created(self, tmp_path):
        output = tmp_path / "combined_requirements.txt"
        packages = [_make_req("numpy", ">=1.20"), _make_req("torch", "==2.1.0")]
        write_combined_requirements(
            output_path=output,
            packages=packages,
            specialized=[],
            extra_indices=[],
            audit=_make_audit(),
            audit_report=AuditReport(),
        )
        assert output.exists()

    def test_packages_appear_in_output(self, tmp_path):
        output = tmp_path / "req.txt"
        packages = [_make_req("numpy", ">=1.20"), _make_req("accelerate")]
        write_combined_requirements(
            output_path=output,
            packages=packages,
            specialized=[],
            extra_indices=[],
            audit=_make_audit(),
            audit_report=AuditReport(),
        )
        content = output.read_text(encoding="utf-8")
        assert "numpy" in content
        assert "accelerate" in content

    def test_extra_index_url_present(self, tmp_path):
        output = tmp_path / "req.txt"
        write_combined_requirements(
            output_path=output,
            packages=[_make_req("torch")],
            specialized=[],
            extra_indices=["https://download.pytorch.org/whl/cu124"],
            audit=_make_audit(),
            audit_report=AuditReport(),
        )
        content = output.read_text(encoding="utf-8")
        assert "--extra-index-url https://download.pytorch.org/whl/cu124" in content

    def test_sorted_alphabetically(self, tmp_path):
        output = tmp_path / "req.txt"
        packages = [_make_req("z_pkg"), _make_req("a_pkg")]
        write_combined_requirements(
            output_path=output,
            packages=packages,
            specialized=[],
            extra_indices=[],
            audit=_make_audit(),
            audit_report=AuditReport(),
        )
        content = output.read_text(encoding="utf-8")
        idx_a = content.index("a_pkg")
        idx_z = content.index("z_pkg")
        assert idx_a < idx_z

    def test_audit_section_all_comments(self, tmp_path):
        """Every line in the audit section must start with # — pip-safe."""
        output = tmp_path / "req.txt"
        write_combined_requirements(
            output_path=output,
            packages=[_make_req("torch")],
            specialized=[],
            extra_indices=[],
            audit=_make_audit(),
            audit_report=AuditReport(untracked_wheels=["mystery.whl"]),
        )
        content = output.read_text(encoding="utf-8")
        split_marker = "# DIAGNOSTIC AUDIT REPORT"
        assert split_marker in content, "Audit section header not found"
        audit_section = content[content.index(split_marker):]
        for line in audit_section.splitlines():
            stripped = line.strip()
            if stripped:
                assert stripped.startswith("#"), (
                    f"Non-comment line in audit section: {line!r}"
                )
