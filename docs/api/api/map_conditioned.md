# map_conditioned

`MapConditionedTrajectoryModel` — the map-conditioned diffusion model that
cross-attends to scene/map tokens (the showcase architecture). It is driven
end-to-end by `scripts/train_wod.py` (warmup-cosine schedule, EMA, held-out
validation).

::: simulacrax.api.map_conditioned
