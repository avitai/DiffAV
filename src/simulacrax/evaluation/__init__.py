"""WOD metrics integration and evaluation orchestration.

Public API::

    from simulacrax.evaluation import (
        ade, fde, min_ade, min_fde, miss_rate, mean_average_precision,
        MotionMetrics, MotionMetricsConfig,
        SimAgentMetrics, SimAgentMetricsConfig, SimAgentMetricsResult,
        histogram_log_likelihood, bernoulli_log_likelihood,
        log_likelihood_estimate_timeseries,
        log_likelihood_estimate_scenario_level,
        compute_metametric_features, MetametricFeatures,
        WosacMetametric, WosacMetametricResult,
        EvaluationRunner,
        MetricsDashboard,
    )
"""

from simulacrax.evaluation.clustering import select_representative_modes
from simulacrax.evaluation.dashboard import MetricsDashboard
from simulacrax.evaluation.estimators import (
    bernoulli_log_likelihood,
    histogram_log_likelihood,
    log_likelihood_estimate_scenario_level,
    log_likelihood_estimate_timeseries,
)
from simulacrax.evaluation.map_conditioned_evaluator import (
    MapConditionedEvaluator,
    MapConditionedSampler,
    ValidationEvalConfig,
    ValidationMetrics,
    ValidationScene,
)
from simulacrax.evaluation.metrics import (
    ade,
    fde,
    mean_average_precision,
    min_ade,
    min_fde,
    miss_rate,
    MotionMetrics,
    MotionMetricsConfig,
    offroad_rate,
    SimAgentMetrics,
    SimAgentMetricsConfig,
    SimAgentMetricsResult,
)
from simulacrax.evaluation.runner import EvaluationRunner, TrajectorySampler
from simulacrax.evaluation.wosac_metametric import (
    compute_metametric_features,
    MetametricFeatures,
    WosacMetametric,
    WosacMetametricResult,
)


__all__ = [
    "ade",
    "bernoulli_log_likelihood",
    "compute_metametric_features",
    "fde",
    "histogram_log_likelihood",
    "log_likelihood_estimate_scenario_level",
    "log_likelihood_estimate_timeseries",
    "min_ade",
    "min_fde",
    "miss_rate",
    "mean_average_precision",
    "MetametricFeatures",
    "MotionMetrics",
    "MotionMetricsConfig",
    "SimAgentMetrics",
    "SimAgentMetricsConfig",
    "SimAgentMetricsResult",
    "WosacMetametric",
    "WosacMetametricResult",
    "EvaluationRunner",
    "TrajectorySampler",
    "MetricsDashboard",
    "offroad_rate",
    "MapConditionedEvaluator",
    "MapConditionedSampler",
    "ValidationEvalConfig",
    "ValidationMetrics",
    "ValidationScene",
    "select_representative_modes",
]
