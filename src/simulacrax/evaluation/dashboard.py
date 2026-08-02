"""MetricsDashboard: generates CSV/HTML tables from MetricsReport data.

Uses calibrax's PublicationGenerator to produce per-agent-type breakdown
tables and physics violation summaries, written to a configurable output dir.

Directory layout::

    output_dir/
        motion/
            table.csv
            table.html
        violations/
            table.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

from calibrax.core.models import Metric, Point, Run
from calibrax.exporters import PublicationGenerator

from simulacrax.core.types import MetricsReport


logger = logging.getLogger(__name__)

# Agent type prefixes that appear in MotionMetrics output keys.
_AGENT_TYPES = ("VEHICLE", "PEDESTRIAN", "CYCLIST")

# Group label for un-prefixed metric keys (e.g. the SDK's "ade"/"fde" report).
_OVERALL_GROUP = "ALL"


class MetricsDashboard:
    """Generate publication-ready tables from a MetricsReport.

    Produces per-agent-type breakdown tables (CSV and HTML) under
    ``output_dir/motion/``, and an optional physics violation summary under
    ``output_dir/violations/``.

    Args:
        output_dir: Root directory where generated files are saved.
    """

    def __init__(self, output_dir: Path | str) -> None:
        """Initialise MetricsDashboard."""
        self._output_dir = Path(output_dir)

    def generate(self, report: MetricsReport) -> Path:
        """Generate CSV and HTML tables from a MetricsReport.

        Builds one calibrax ``Run`` with one ``Point`` per metric group —
        each ``VEHICLE``/``PEDESTRIAN``/``CYCLIST`` prefix plus an ``ALL``
        group collecting un-prefixed keys (the shape
        ``ScenarioMiner.evaluate_planner`` reports) — then writes a combined
        motion table (CSV + HTML). If ``report.kinematic_violation_summary``
        is non-empty, a separate violation summary CSV is also written.

        Args:
            report: Aggregated evaluation results from ``EvaluationRunner``
                or ``ScenarioMiner.evaluate_planner``.

        Returns:
            ``output_dir`` path where all files were written.

        Raises:
            ValueError: If the report carries no metric values and no
                violation summary, or a metric key has an unrecognised
                agent-type prefix.
        """
        if not report.metric_values and not report.kinematic_violation_summary:
            msg = (
                "report has no metric_values and no kinematic_violation_summary; nothing to render"
            )
            raise ValueError(msg)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if report.metric_values:
            self._generate_motion_tables(report)
        if report.kinematic_violation_summary:
            self._generate_violation_table(report)
        return self._output_dir

    def _generate_motion_tables(self, report: MetricsReport) -> None:
        """Write combined per-group motion metric tables (CSV + HTML)."""
        groups: dict[str, dict[str, float]] = {}
        unknown: list[str] = []
        for key, val in report.metric_values.items():
            prefix, separator, metric_name = key.partition("/")
            if not separator:
                groups.setdefault(_OVERALL_GROUP, {})[key] = val
            elif prefix in _AGENT_TYPES:
                groups.setdefault(prefix, {})[metric_name] = val
            else:
                unknown.append(key)
        if unknown:
            msg = (
                f"metric keys with unrecognised agent-type prefixes: {sorted(unknown)}; "
                f"expected one of {_AGENT_TYPES} or un-prefixed overall keys"
            )
            raise ValueError(msg)

        points = tuple(
            Point(
                name=group,
                scenario="wod_evaluation",
                tags={"agent_type": group},
                metrics={name: Metric(value=val) for name, val in groups[group].items()},
            )
            for group in (*_AGENT_TYPES, _OVERALL_GROUP)
            if group in groups
        )

        run = Run(points=points)
        motion_dir = self._output_dir / "motion"
        motion_dir.mkdir(parents=True, exist_ok=True)
        generator = PublicationGenerator(motion_dir)
        generator.generate_table(run, output_format="csv", group_by_tag="agent_type")
        generator.generate_table(run, output_format="html", group_by_tag="agent_type")
        logger.info("Generated motion tables in %s.", motion_dir)

    def _generate_violation_table(self, report: MetricsReport) -> None:
        """Write physics violation summary as CSV."""
        point = Point(
            name="kinematic_violations",
            scenario="wod_evaluation",
            tags={"summary": "kinematic"},
            metrics={
                name: Metric(value=val) for name, val in report.kinematic_violation_summary.items()
            },
        )
        run = Run(points=(point,))
        violations_dir = self._output_dir / "violations"
        violations_dir.mkdir(parents=True, exist_ok=True)
        generator = PublicationGenerator(violations_dir)
        generator.generate_table(run, output_format="csv", group_by_tag="summary")
        logger.info("Generated kinematic violation table in %s.", violations_dir)
