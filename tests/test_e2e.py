"""
End-to-end tests for mango_evolve with real LLM API calls.

These tests require a valid ANTHROPIC_API_KEY environment variable.
They are skipped by default in CI and only run when explicitly enabled.

To run these tests:
    ANTHROPIC_API_KEY=your_key pytest tests/test_e2e.py -v

Or with a specific budget:
    ANTHROPIC_API_KEY=your_key pytest tests/test_e2e.py -v -k "budget_limited"
"""

import os

import pytest

from mango_evolve import CostTracker, config_from_dict
from mango_evolve.evaluation.circle_packing import CirclePackingEvaluator
from mango_evolve.evolution_api import EvolutionAPI
from mango_evolve.llm.client import LLMClient
from mango_evolve.logger import ExperimentLogger
from mango_evolve.root_llm import RootLLMOrchestrator

# Skip all tests in this module if no API key is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set - skipping E2E tests"
)


@pytest.fixture
def e2e_config_dict(temp_dir):
    """Configuration for E2E tests with conservative budget."""
    return {
        "experiment": {
            "name": "e2e_test",
            "output_dir": str(temp_dir),
        },
        "root_llm": {
            "model": "claude-sonnet-4-20250514",
            "cost_per_million_input_tokens": 3.0,
            "cost_per_million_output_tokens": 15.0,
            "max_iterations": 5,
        },
        "child_llm": {
            "model": "claude-sonnet-4-20250514",
            "cost_per_million_input_tokens": 3.0,
            "cost_per_million_output_tokens": 15.0,
        },
        "evaluation": {
            "evaluator_fn": "mango_evolve.evaluation.circle_packing:CirclePackingEvaluator",
            "evaluator_kwargs": {
                "n_circles": 26,
                "timeout_seconds": 30,
            },
        },
        "evolution": {
            "max_generations": 2,
            "max_children_per_generation": 3,
        },
        "budget": {
            "max_total_cost": 0.50,  # Conservative budget for testing
        },
    }


@pytest.fixture
def e2e_config(e2e_config_dict):
    """Config object for E2E tests."""
    return config_from_dict(e2e_config_dict)


class TestSingleTrialRealLLM:
    """Tests that run a single trial with the real LLM API."""

    def test_single_trial_real_llm(self, e2e_config, temp_dir):
        """
        Test a single child LLM spawn with real API call.

        This test verifies that:
        1. The LLM client can connect to the Anthropic API
        2. The response contains valid Python code
        3. The evaluator can process the generated code
        4. Cost tracking works correctly
        """
        # Set up components
        cost_tracker = CostTracker(e2e_config)
        logger = ExperimentLogger(e2e_config)
        logger.create_experiment_directory()

        evaluator = CirclePackingEvaluator(n_circles=26)

        child_llm = LLMClient(
            model=e2e_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        evolution_api = EvolutionAPI(
            evaluator=evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        # Spawn a single trial with a detailed prompt
        prompt = """Write a Python function to solve the circle packing problem.

Problem: Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of radii.

Constraints:
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

Your code MUST define these functions:

```python
import numpy as np

def construct_packing():
    '''
    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii
    '''
    # Your implementation here
    pass

def run_packing():
    return construct_packing()
```

Please implement a simple but valid grid-based approach that places circles in a regular pattern.
Return the complete code in a single Python code block."""

        result = evolution_api.spawn_child_llm(prompt)

        # Verify the result structure
        assert "trial_id" in result
        assert "code" in result
        assert "metrics" in result
        assert "success" in result

        # The code should have been extracted
        assert len(result["code"]) > 0, "No code was extracted from LLM response"

        # The evaluation should have run
        assert "valid" in result["metrics"], "Evaluation didn't produce 'valid' metric"

        # Cost should have been tracked
        summary = cost_tracker.get_summary()
        assert summary.total_cost > 0, "No cost was recorded"
        assert summary.total_input_tokens > 0
        assert summary.total_output_tokens > 0

        # Print results for manual inspection
        print("\n--- E2E Single Trial Results ---")
        print(f"Trial ID: {result['trial_id']}")
        print(f"Success: {result['success']}")
        print(f"Valid: {result['metrics'].get('valid')}")
        print(f"Score: {result['metrics'].get('score', 0):.4f}")
        print(f"Cost: ${summary.total_cost:.6f}")
        print(f"Tokens: {summary.total_input_tokens} in / {summary.total_output_tokens} out")


class TestBudgetLimitedRun:
    """Tests that run with a strict budget limit."""

    def test_budget_limited_run(self, e2e_config_dict, temp_dir):
        """
        Test a short evolution run with $0.50 budget.

        This test verifies that:
        1. The full orchestration loop works with real LLMs
        2. Budget enforcement stops the run before exceeding the limit
        3. Results are saved correctly
        """
        # Use a very limited budget
        e2e_config_dict["budget"]["max_total_cost"] = 0.50
        e2e_config_dict["root_llm"]["max_iterations"] = 3
        config = config_from_dict(e2e_config_dict)

        orchestrator = RootLLMOrchestrator(config)

        # Run the evolution
        result = orchestrator.run()

        # Basic assertions
        assert result.terminated is True
        assert result.num_iterations >= 1

        # Cost should be within budget (with some margin for the last call)
        if result.cost_summary:
            total_cost = result.cost_summary.get("total_cost", 0)
            # Allow some overage since we check budget before each call
            assert total_cost < 1.0, f"Cost ${total_cost:.4f} exceeded expected limit"

        # Results should be saved
        assert orchestrator.logger.base_dir.exists()
        assert (orchestrator.logger.base_dir / "experiment.json").exists()

        # Print results for manual inspection
        print("\n--- E2E Budget Limited Run Results ---")
        print(f"Termination reason: {result.reason}")
        print(f"Iterations: {result.num_iterations}")
        print(f"Total trials: {result.total_trials}")
        print(f"Successful trials: {result.successful_trials}")
        print(f"Best score: {result.best_score:.4f}")
        if result.cost_summary:
            print(f"Total cost: ${result.cost_summary.get('total_cost', 0):.6f}")
        print(f"Results saved to: {orchestrator.logger.base_dir}")


class TestRealLLMConnection:
    """Basic tests to verify LLM API connectivity."""

    def test_llm_connection(self, e2e_config):
        """Test that we can connect to the Anthropic API."""
        cost_tracker = CostTracker(e2e_config)

        client = LLMClient(
            model=e2e_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        # Simple test message
        response = client.generate(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=50,
            temperature=0.0,
        )

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "hello" in response.content.lower()

        # Cost should be recorded
        summary = cost_tracker.get_summary()
        assert summary.total_cost > 0
