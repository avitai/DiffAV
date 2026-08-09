"""Tests for MetricsDashboard."""

from __future__ import annotations

from pathlib import Path

import pytest

from diffav.core.types import MetricsReport
from diffav.evaluation.dashboard import MetricsDashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(*, with_violations: bool = False) -> MetricsReport:
    metric_values = {
        "VEHICLE/minADE": 1.2,
        "VEHICLE/minFDE": 2.5,
        "VEHICLE/MissRate": 0.3,
        "VEHICLE/mAP": 0.6,
        "PEDESTRIAN/minADE": 0.8,
        "PEDESTRIAN/minFDE": 1.4,
        "PEDESTRIAN/MissRate": 0.1,
        "PEDESTRIAN/mAP": 0.75,
    }
    kinematic = {"speed_violation_rate": 0.02, "offroad_rate": 0.05} if with_violations else {}
    return MetricsReport(
        metric_values=metric_values,
        kinematic_violation_summary=kinematic,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsDashboard:
    """Tests for MetricsDashboard.generate()."""

    def test_returns_output_dir(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        result = dashboard.generate(_make_report())
        assert result == tmp_path

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "output"
        dashboard = MetricsDashboard(out)
        dashboard.generate(_make_report())
        assert out.is_dir()

    def test_generates_motion_csv_with_content(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(_make_report())
        content = (tmp_path / "motion" / "table.csv").read_text()
        # Groups, metric names, and actual values must all render
        assert "VEHICLE" in content
        assert "PEDESTRIAN" in content
        assert "minADE" in content
        assert "1.2" in content  # VEHICLE/minADE value from the fixture
        assert "0.75" in content  # PEDESTRIAN/mAP value from the fixture

    def test_generates_motion_html_with_content(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(_make_report())
        content = (tmp_path / "motion" / "table.html").read_text()
        assert "VEHICLE" in content
        assert "minADE" in content

    def test_violation_table_written_when_present(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(_make_report(with_violations=True))
        assert (tmp_path / "violations" / "table.csv").exists()

    def test_no_violation_dir_when_summary_empty(self, tmp_path: Path) -> None:
        """violations/ subdir should not be created when no violation data."""
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(_make_report(with_violations=False))
        assert not (tmp_path / "violations").exists()

    def test_empty_report_raises(self, tmp_path: Path) -> None:
        """A report with nothing to render fails fast instead of writing nothing."""
        dashboard = MetricsDashboard(tmp_path)
        with pytest.raises(ValueError, match="nothing to render"):
            dashboard.generate(MetricsReport(metric_values={}))
        assert not (tmp_path / "motion").exists()
        assert not (tmp_path / "violations").exists()

    def test_cyclist_type_skipped_when_absent(self, tmp_path: Path) -> None:
        """CYCLIST keys absent → still produces VEHICLE/PEDESTRIAN tables."""
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(_make_report())
        assert (tmp_path / "motion" / "table.csv").exists()

    def test_accepts_string_output_dir(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(str(tmp_path / "str_path"))
        result = dashboard.generate(_make_report())
        assert result.is_dir()


class TestRenderableContract:
    """The dashboard renders SDK reports and fails fast on unrenderable ones."""

    def test_sdk_report_renders_motion_table(self, tmp_path: Path) -> None:
        """Un-prefixed keys (the SDK's own report shape) produce a real table."""
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(MetricsReport(metric_values={"ade": 1.2, "fde": 2.4}))
        csv_path = tmp_path / "motion" / "table.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "ade" in content
        assert "fde" in content

    def test_mixed_prefixed_and_overall_keys_render_both_groups(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(MetricsReport(metric_values={"VEHICLE/minADE": 1.0, "ade": 2.0}))
        content = (tmp_path / "motion" / "table.csv").read_text()
        assert "VEHICLE" in content
        assert "ALL" in content

    def test_unknown_prefix_raises_naming_key(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        with pytest.raises(ValueError, match="TRUCK/minADE"):
            dashboard.generate(MetricsReport(metric_values={"TRUCK/minADE": 1.0}))

    def test_violations_only_report_renders_without_motion(self, tmp_path: Path) -> None:
        dashboard = MetricsDashboard(tmp_path)
        dashboard.generate(
            MetricsReport(
                metric_values={},
                kinematic_violation_summary={"speed_violation_rate": 0.02},
            )
        )
        assert (tmp_path / "violations" / "table.csv").exists()
        assert not (tmp_path / "motion").exists()
