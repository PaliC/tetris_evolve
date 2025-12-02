"""
Pytest configuration and fixtures for tetris_evolve tests.
"""

import pytest
import tempfile
from pathlib import Path

from tetris_evolve import (
    Config,
    ExperimentConfig,
    LLMConfig,
    EvolutionConfig,
    BudgetConfig,
    EvaluationConfig,
    config_from_dict,
)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "experiment": {
            "name": "test_experiment",
            "output_dir": "./test_experiments",
        },
        "root_llm": {
            "model": "claude-sonnet-4-20250514",
            "cost_per_input_token": 0.000003,
            "cost_per_output_token": 0.000015,
            "max_iterations": 30,
        },
        "child_llm": {
            "model": "claude-sonnet-4-20250514",
            "cost_per_input_token": 0.000003,
            "cost_per_output_token": 0.000015,
        },
        "evolution": {
            "max_generations": 5,
            "max_children_per_generation": 3,
        },
        "budget": {
            "max_total_cost": 10.0,
        },
        "evaluation": {
            "n_circles": 26,
            "target_sum": 2.635,
            "timeout_seconds": 30,
        },
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """Sample Config object."""
    return config_from_dict(sample_config_dict)


@pytest.fixture
def temp_dir():
    """Temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_valid_packing_code():
    """Valid circle packing code for testing."""
    return '''
import numpy as np

def construct_packing():
    """Simple grid-based packing."""
    n = 26
    centers = []

    # 5x5 grid = 25, plus 1 more
    grid_size = 5
    spacing = 1.0 / (grid_size + 1)

    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx < n:
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                centers.append([x, y])
                idx += 1

    # Add one more circle
    centers.append([0.5, 0.5])

    centers = np.array(centers[:n])

    # Compute radii
    radii = np.ones(n) * (spacing / 2 - 0.01)

    # Limit by boundaries
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1-x, 1-y)
        radii[i] = min(radii[i], max_r)

    # Limit by neighbors
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.99
                radii[i] *= scale
                radii[j] *= scale

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

def run_packing():
    return construct_packing()
'''


@pytest.fixture
def sample_invalid_packing_code():
    """Invalid circle packing code (circles overlap)."""
    return '''
import numpy as np

def construct_packing():
    n = 26
    # All circles at same position - will overlap
    centers = np.ones((n, 2)) * 0.5
    radii = np.ones(n) * 0.1
    return centers, radii, np.sum(radii)

def run_packing():
    return construct_packing()
'''


@pytest.fixture
def sample_broken_code():
    """Code with syntax/runtime error."""
    return '''
def run_packing():
    return undefined_variable
'''


@pytest.fixture
def sample_no_function_code():
    """Code without required function."""
    return '''
x = 5
y = 10
'''
