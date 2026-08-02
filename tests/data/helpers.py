"""Shared constants for data module tests."""

from __future__ import annotations


NUM_OBJECTS = 4  # ego + 3 agents
NUM_TIMESTEPS = 91  # 10 past + 1 current + 80 future
NUM_RG_POINTS = 20  # roadgraph polyline points
CURRENT_TIME_INDEX = 10
HISTORY_STEPS = 11  # indices 0..10
FUTURE_STEPS = 80  # indices 11..90

OBJ_TYPE_VEHICLE = 1
OBJ_TYPE_PEDESTRIAN = 2
OBJ_TYPE_CYCLIST = 3

NUM_PADDING_OBJECTS = 2  # extra all-invalid rows mimicking TFRecord padding
