# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
"""
# Title: [Descriptive Example Name]

| Metadata | Value |
|----------|-------|
| **Level** | Beginner / Intermediate / Advanced |
| **Runtime** | ~X min (CPU) / ~Y min (GPU) |
| **Prerequisites** | [List required knowledge] |
| **Format** | Python + Jupyter |

## Overview

[Brief 1-2 paragraph description of what this example demonstrates and why it's useful.]

## Learning Goals

By the end of this example, you will be able to:

1. [First specific, measurable objective - use action verbs: Create, Train, Implement]
2. [Second objective]
3. [Third objective]
"""

# %% [markdown]
"""
## Setup & Prerequisites

### Required Knowledge
- [Prerequisite 1](link) - Brief description
- [Prerequisite 2](link) - Brief description

### Installation

```bash
# Install diffav with development dependencies
uv sync
```

**Estimated Time:** X-Y minutes
"""

# %%
# Imports  # noqa: F401
import jax.numpy as jnp  # noqa: F401
import numpy as np  # noqa: F401
from flax import nnx  # noqa: F401

from diffav.core.types import AgentState, AgentType, SceneContext  # noqa: F401
from diffav.data.wod_source import WODSource, WODSourceConfig  # noqa: F401


# %% [markdown]
"""
## Implementation

### Step 1: [First Step Title]

[Brief explanation of what we're doing and why]
"""

# %%
# Step 1: [implementation code]
print("Step 1 complete")

# %% [markdown]
"""
### Step 2: [Second Step Title]

[Brief explanation]
"""

# %%
# Step 2: [implementation code]
print("Step 2 complete")

# %% [markdown]
"""
## Results & Evaluation

### What We Achieved

[Summary of what was demonstrated]

### Key Metrics

| Metric | Value |
|--------|-------|
| [Metric 1] | [Value] |
| [Metric 2] | [Value] |
"""

# %%
print("Example completed successfully!")

# %% [markdown]
"""
## Next Steps & Resources

### Try These Experiments

1. [Specific, achievable experiment 1]
2. [Experiment 2]

### Related Examples

- [Example Name](../path/to/example.py) - Brief description

### API Reference

- [WODSource](../../docs/api/data/wod_source.md)
- [SceneContext](../../docs/api/core/types.md)
"""


# %%
def main() -> None:
    """Main entry point for command-line execution."""
    print("Running example...")
    print("Done!")


if __name__ == "__main__":
    main()
