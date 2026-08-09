"""RLHF/DPO preference alignment for adversarial scenario generation.

Provides safety reward functions, preference pair construction, and
DPO fine-tuning for steering the trajectory diffusion model toward
adversarial, physically plausible edge-cases.
"""

from diffav.alignment.dpo_trainer import (
    create_reference_model,
    DPOAlignmentConfig,
    DPOAlignmentMetrics,
    DPOAlignmentTrainer,
)
from diffav.alignment.preferences import (
    PreferenceBatch,
    PreferencePair,
    PreferencePairBuilder,
    PreferencePairConfig,
    RankingStrategy,
)
from diffav.alignment.rewards import (
    BoundaryReward,
    CollisionReward,
    ComfortReward,
    KinematicReward,
    SafetyReward,
    SafetyRewardConfig,
)
from diffav.alignment.scenario_steering import (
    compute_scenario_reward,
    ScenarioSteeringConfig,
    ScenarioSteeringTrainer,
    SteeringStrategy,
)
from diffav.alignment.steering_spine import (
    build_steering_pairs,
    candidate_offroad_fractions,
    CandidatePool,
    sample_and_score,
)
from diffav.alignment.weight_soup import make_weight_soup


__all__ = [
    "BoundaryReward",
    "CollisionReward",
    "ComfortReward",
    "DPOAlignmentConfig",
    "DPOAlignmentMetrics",
    "DPOAlignmentTrainer",
    "KinematicReward",
    "PreferenceBatch",
    "PreferencePair",
    "PreferencePairBuilder",
    "PreferencePairConfig",
    "RankingStrategy",
    "SafetyReward",
    "SafetyRewardConfig",
    "CandidatePool",
    "ScenarioSteeringConfig",
    "ScenarioSteeringTrainer",
    "SteeringStrategy",
    "build_steering_pairs",
    "candidate_offroad_fractions",
    "compute_scenario_reward",
    "create_reference_model",
    "make_weight_soup",
    "sample_and_score",
]
