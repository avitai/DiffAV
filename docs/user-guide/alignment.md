# Preference-Based Alignment

DiffAV steers the trajectory diffusion model toward adversarial,
physically plausible edge-cases through preference-based alignment.
This guide explains the reward composition, preference pair construction,
and the DPO data pipeline that connects generation to fine-tuning.

## Why Preference Alignment?

Standard diffusion training produces realistic trajectories but does not
optimise for safety-critical scenarios. Preference alignment adds a
secondary objective: given two candidate trajectories, the model should
learn to prefer the one that better exercises safety boundaries while
remaining physically plausible.

The pipeline follows three stages:

```
TrajectoryDiffusionModel.sample()
    |  (generate N candidates per scene)
    v
PreferencePairBuilder.score_candidates()
    |  (score each candidate)
    v
PreferencePairBuilder.build_batch()
    |  (rank, pair, filter)
    v
PreferenceBatch.to_dpo_batch()
    |  (format for DPO)
    v
DPOTrainer.train_step()
```

## Safety Reward Composition

The composite `SafetyReward` combines four domain-specific reward functions,
each satisfying the artifex `RewardFunction` protocol:

| Reward | What it measures | Higher = |
|--------|-----------------|----------|
| `CollisionReward` | Pairwise agent distance penalty | Safer (fewer near-misses) |
| `KinematicReward` | Bicycle model residual | More physically plausible |
| `BoundaryReward` | Off-road excursion beyond road edges | Staying on-road |
| `ComfortReward` | Jerk (rate of acceleration change) | Smoother ride |

All rewards return **negative** values for violations, following the RL
convention where higher reward = better. A trajectory with zero collision
penalty, low kinematic residual, and smooth velocity gets a reward near zero.

### Configuration

```python
from diffav.alignment import SafetyReward, SafetyRewardConfig
from diffav.physics.kinematics import BicycleModelConfig

config = SafetyRewardConfig(
    collision_threshold=2.0,   # metres
    collision_weight=1.0,
    kinematic_weight=1.0,
    boundary_weight=1.0,
    comfort_weight=0.5,
    bicycle_config=BicycleModelConfig(wheelbase=2.7),
    dt=0.1,
)
reward = SafetyReward(config)
```

Weights control the relative importance of each safety dimension. Set
a weight to zero to disable that component.

### Diagnostics

Access individual component rewards for debugging:

```python
components = reward.component_rewards
collision_scores = components["collision"](trajectories)
kinematic_scores = components["kinematic"](trajectories)
```

## Ranking Strategies

The `PreferencePairBuilder` supports two strategies for pairing ranked
candidates into chosen/rejected pairs:

### Best vs Worst (default)

Sort candidates by reward descending, then pair the top-k with the
bottom-k. This maximises the reward margin between chosen and rejected,
producing strong training signal:

```
Rank:   1   2   3   4   5   6   7   8
        ^                           ^
        |___________ pair 1 ________|
            ^                   ^
            |____ pair 2 _______|
```

### Adjacent

Sort candidates by reward descending, then pair consecutive positions.
This produces more pairs with smaller margins, useful for fine-grained
preference learning:

```
Rank:   1   2   3   4   5   6   7   8
        ^   ^   ^   ^   ^   ^   ^   ^
        p1      p2      p3      p4
```

### Margin Filtering

Set `min_reward_margin` to discard pairs where chosen and rejected are
too similar. This prevents the DPO trainer from learning noise:

```python
from diffav.alignment import PreferencePairConfig, RankingStrategy

config = PreferencePairConfig(
    ranking_strategy=RankingStrategy.BEST_VS_WORST,
    num_candidates=8,
    top_k=2,
    min_reward_margin=0.1,  # discard pairs with margin < 0.1
)
```

## DPO Data Pipeline

The complete pipeline from generation to DPO training data:

```python
import jax
from diffav.alignment import (
    PreferencePairBuilder,
    PreferencePairConfig,
    SafetyReward,
)

# 1. Set up reward and builder
reward = SafetyReward()
builder = PreferencePairBuilder(
    reward_fn=reward,
    config=PreferencePairConfig(top_k=2, num_candidates=8),
)

# 2. Generate candidates (from diffusion model)
# candidates shape: (8, num_agents, future_steps, 4)
candidates = ...  # model.sample() called 8 times

# 3. Build preference batch
batch = builder.build_batch(candidates, conditions=scene_context)

# 4. Convert to DPO format
dpo_batch = batch.to_dpo_batch()
# {"chosen": Array(2, A, T, 4), "rejected": Array(2, A, T, 4)}
```

For multi-scene training, use `build_batch_from_scenes` to aggregate
pairs across multiple driving scenarios:

```python
batch = builder.build_batch_from_scenes(
    candidates_per_scene=[scene1_candidates, scene2_candidates],
    conditions_per_scene=[scene1_context, scene2_context],
)
```

## DPO Fine-Tuning

The `DPOAlignmentTrainer` consumes preference pairs to fine-tune the
trajectory diffusion model using Direct Preference Optimization (DPO).
It implements the **Diffusion-DPO** approach (Wallace et al. 2023),
estimating trajectory log-probabilities via Monte Carlo averaging of
negative noise prediction error:

$$\log p_\theta(x) \approx -\mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### Diffusion Log-Probabilities

Unlike standard DPO for language models (which uses softmax log-probs
over discrete tokens), diffusion models produce continuous outputs.
The log-probability proxy is computed by:

1. Sampling K random (timestep, noise) pairs
2. Forward-diffusing the trajectory to each timestep
3. Predicting noise with the model (deterministic mode)
4. Averaging the negative MSE across all samples

This is controlled by `num_log_prob_samples` in `DPOAlignmentConfig`.

The policy and reference passes evaluate the **same** (timestep, noise)
draws per trajectory, so the Monte Carlo noise cancels in the log-ratio
— with independent draws the ratio would be noise-dominated at small K.

### Reference Model Handling

DPO computes a log-ratio between the policy model and a frozen reference:

$$\mathcal{L}_\text{DPO} = -\log\sigma\left(\beta \cdot \left[\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}\right]\right)$$

Two modes are supported:

- **Standard DPO** (`reference_free=False`): Creates a frozen copy of
  the policy model via `create_reference_model()`. The reference model's
  parameters are not updated during training.
- **SimPO** (`reference_free=True`): Skips the reference model entirely,
  using only policy log-probs. Saves memory at the cost of less stable
  training.

### Physics Regularisation

To prevent alignment from degrading physical plausibility, the trainer
supports an optional physics regularisation term:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{DPO} + \lambda_\text{phys} \cdot \mathcal{L}_\text{physics}$$

This reuses `DiffAVPhysicsLoss` from the physics-informed training
pipeline, applied to chosen trajectories via `jax.vmap`.

### Training Example

```python
import jax
from flax import nnx
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

from diffav.alignment import (
    DPOAlignmentConfig,
    DPOAlignmentTrainer,
    create_reference_model,
    PreferencePairBuilder,
    SafetyReward,
)
from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)

# 1. Create model and optimizer
model = TrajectoryDiffusionModel(
    TrajectoryDiffusionConfig(hidden_dim=128, num_blocks=2, num_heads=4),
    rngs=nnx.Rngs(params=jax.random.key(0)),
)
tx = create_optimizer(OptimizerConfig(learning_rate=1e-5, gradient_clip=1.0))
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# 2. Create reference model (frozen copy)
reference = create_reference_model(model)

# 3. Configure DPO training
config = DPOAlignmentConfig(
    beta=0.1,
    num_log_prob_samples=4,
    physics_weight=0.0,
)

# 4. Build trainer
trainer = DPOAlignmentTrainer(
    model=model,
    optimizer=optimizer,
    config=config,
    reference_model=reference,
)

# 5. Build preference data and train
# batch = preference_batch.to_dpo_batch()
# metrics = trainer.train_step(batch, jax.random.key(step))
# print(f"DPO loss: {metrics.dpo_loss:.4f}")
# print(f"Reward accuracy: {metrics.reward_accuracy:.2%}")
```

## Scenario Steering

`ScenarioSteeringTrainer` steers generation toward a target scenario type by
training on preference pairs the model itself generated. Each step:

1. Samples `num_candidates` trajectories per scene context from the current
   policy (`sample_and_score`).
2. Scores each candidate with the rule-based scenario reward plus the
   composite safety reward. The two components are rank-normalized per pool
   before combining — the scenario reward is bounded in [-1, 1] while the
   safety composite is unbounded, so raw addition would let safety drown the
   steering target.
3. Gates feasibility on the signed distance to road edges: a candidate whose
   off-road fraction exceeds `feasibility_threshold` can never be "chosen"
   (it remains legitimate "rejected" material).
4. Applies one standard DPO update on the ranked chosen/rejected pairs
   (`build_steering_pairs`).

The reward shapes the training distribution — it never enters the loss as a
term, so the steering gradient is exactly the DPO gradient on
model-generated pairs. `steering_strength` is the selection pressure: pools
span the top/bottom `ceil((1 − α) · num_feasible)` of the ranking, so higher
α widens the reward gap between pairs, and `0.0` pairs randomly (plain DPO,
no steering).

```python
from diffav.alignment import ScenarioSteeringConfig, ScenarioSteeringTrainer

config = ScenarioSteeringConfig(
    target_scenario="forward",
    steering_strength=0.5,
    num_candidates=8,
)
trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=config)
metrics = trainer.steer_step(probe_contexts, jax.random.key(0), num_agents=8)
```

The high-level entry point is `ScenarioMiner.steer()`, which fine-tunes a
clone of the miner's model and generates with the tuned weights.

### Guidance Strategy

With `strategy=SteeringStrategy.GUIDANCE` nothing is trained: sampling
takes reward-gradient ascent steps in x̂₀-space. At each reverse-diffusion
step the clean estimate becomes `x̂₀ + η·ᾱ_t·∇R(x̂₀)` before the posterior
is formed — the `ᾱ_t` annealing applies strong guidance only once x̂₀ is
trustworthy, and non-finite reward gradients are skipped. The reward is
`make_steering_guidance(target, reference_speed=...)` (raw scenario +
safety rewards — guidance needs a continuous gradient, not a pool
ordering), and `steering_strength` is the guidance scale η, calibrated
against the reward's gradient magnitude.

### Weight-Soup Strategy

With `strategy=SteeringStrategy.WEIGHT_SOUP`, `steer()` trains one
full-pressure expert with the ranked-pair mechanism, then interpolates
parameters between the base model and the expert:
`θ = (1 − λ)·θ_base + λ·θ_target` with `λ = steering_strength`
(`make_weight_soup`). Steering strength becomes a post-training dial — one
expert run serves every strength, which is what the steering bake-off
sweeps exploit.

## Related

- [Safety Rewards API](../api/alignment/rewards.md)
- [Preference Builder API](../api/alignment/preferences.md)
- [DPO Trainer API](../api/alignment/dpo_trainer.md)
- [Scenario Steering API](../api/alignment/scenario_steering.md)
- [Physics-Informed Training](physics.md)
- [Bicycle Model Kinematics](../api/physics/kinematics.md)
