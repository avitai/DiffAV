"""Smoke tests for the steering bake-off harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.steering_bakeoff import BakeoffCell, BakeoffConfig, run_steering_bakeoff

from simulacrax.alignment.scenario_steering import SteeringStrategy
from simulacrax.api.config import MinerConfig


pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


def _tiny_config(tmp_path: Path) -> BakeoffConfig:
    return BakeoffConfig(
        strategies=(SteeringStrategy.GUIDANCE,),
        strengths=(0.0, 0.5),
        num_rollouts=2,
        num_candidates=2,
        miner_config=MinerConfig(
            max_agents=4, prediction_horizon=12, context_dim=32, num_diffusion_steps=10
        ),
        output_dir=tmp_path,
    )


class TestSteeringBakeoff:
    def test_grid_produces_one_cell_per_pair(self, tmp_path: Path) -> None:
        cells = run_steering_bakeoff(_tiny_config(tmp_path))
        assert len(cells) == 2
        assert all(isinstance(cell, BakeoffCell) for cell in cells)
        assert [cell.strength for cell in cells] == [0.0, 0.5]

    def test_scores_are_finite_and_in_range(self, tmp_path: Path) -> None:
        cells = run_steering_bakeoff(_tiny_config(tmp_path))
        for cell in cells:
            assert cell.adversariality >= 0.0
            assert 0.0 <= cell.realism <= 1.0
            assert 0.0 <= cell.feasibility <= 1.0

    def test_report_written(self, tmp_path: Path) -> None:
        run_steering_bakeoff(_tiny_config(tmp_path))
        csv_path = tmp_path / "steering" / "table.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "adversariality" in content
        assert "guidance@0.50" in content
