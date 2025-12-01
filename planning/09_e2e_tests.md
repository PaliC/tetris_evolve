# Component 9: End-to-End Tests

## Overview
Full system tests that verify the complete evolution pipeline works correctly.

**File**: `tests/test_e2e.py`
**Dependencies**: Components 1-8 (all complete)

---

## Checklist

### 9.1 Test Setup
- [x] **9.1.1** Create `tests/test_e2e.py`
  - Dependencies: 1-8 all complete
- [x] **9.1.2** Create mock LLM that returns deterministic responses
  - Dependencies: 9.1.1
  - Returns valid code when prompted
  - Tracks call count for verification
- [x] **9.1.3** Create test configuration files
  - Dependencies: 9.1.1
  - Small limits for fast tests (2 generations, $1 budget)
- [x] **9.1.4** Create helper to verify experiment directory structure
  - Dependencies: 9.1.1

### 9.2 Minimal Evolution Test
- [x] **9.2.1** Test: Single generation with 2 children completes
  - Dependencies: 9.1.2, 9.1.3
  - Config: max_generations=1, initial_population=2
  - Verify: 2 trials created, generation completes
- [x] **9.2.2** Test: Experiment directory structure created correctly
  - Dependencies: 9.2.1
  - Verify: experiment.json, generations/, cost_history.json
- [x] **9.2.3** Test: All trial files present
  - Dependencies: 9.2.1
  - Verify: code.py, metrics.json, reasoning.md for each trial

### 9.3 Multi-Generation Evolution
- [x] **9.3.1** Test: 3 generations with selection completes
  - Dependencies: 9.2.1
  - Mock root LLM to spawn, select, advance
  - Verify: 3 generation folders created
- [x] **9.3.2** Test: Parent-child relationships tracked
  - Dependencies: 9.3.1
  - Verify: trial.json has correct parent_id
- [x] **9.3.3** Test: Generation stats calculated
  - Dependencies: 9.3.1
  - Verify: generation_stats.json has best_score, avg_score

### 9.4 Hard Limit Enforcement
- [x] **9.4.1** Test: Stops at max_generations
  - Dependencies: 9.2.1
  - Config: max_generations=2
  - Verify: terminates with "generation_limit" reason
- [x] **9.4.2** Test: Stops at max_cost
  - Dependencies: 9.2.1
  - Config: max_cost=0.01
  - Verify: terminates with "cost_limit" reason
- [x] **9.4.3** Test: Stops at max_time
  - Dependencies: 9.2.1
  - Config: max_time_minutes=0.01
  - Verify: terminates with "time_limit" reason

### 9.5 Root LLM Termination
- [x] **9.5.1** Test: Root LLM can voluntarily terminate
  - Dependencies: 9.2.1
  - Mock root to call terminate_evolution()
  - Verify: terminates with custom reason
- [x] **9.5.2** Test: Root termination saves final state
  - Dependencies: 9.5.1
  - Verify: best trial recorded in experiment.json

### 9.6 Error Recovery
- [x] **9.6.1** Test: Continues when child generates invalid code
  - Dependencies: 9.2.1
  - Mock child to return syntax error code sometimes
  - Verify: evolution continues, error recorded
- [x] **9.6.2** Test: Continues when evaluation fails
  - Dependencies: 9.2.1
  - Mock evaluator to fail sometimes
  - Verify: evolution continues, failure recorded
- [x] **9.6.3** Test: Root LLM error handled gracefully
  - Dependencies: 9.2.1
  - Mock root LLM to return unparseable code
  - Verify: logs error, continues or terminates safely

### 9.7 Resume Functionality
- [x] **9.7.1** Test: Resume continues from last generation
  - Dependencies: 9.3.1
  - Run 2 generations, stop
  - Resume and run 1 more
  - Verify: 3 total generations, state preserved
- [x] **9.7.2** Test: Resume preserves cost history
  - Dependencies: 9.7.1
  - Verify: costs from before resume are included
- [x] **9.7.3** Test: Resume loads population correctly
  - Dependencies: 9.7.1
  - Verify: parent IDs reference original trials

### 9.8 Output Verification
- [x] **9.8.1** Test: Final result contains best code
  - Dependencies: 9.3.1
  - Verify: best_code is valid, evaluatable
- [x] **9.8.2** Test: Cost tracking accurate
  - Dependencies: 9.3.1
  - Compare tracked cost vs expected from mocks
- [x] **9.8.3** Test: All reasoning recorded
  - Dependencies: 9.3.1
  - Verify: root_llm_reasoning.md for each generation

---

## Test Summary

| Test | Description | Components |
|------|-------------|------------|
| `test_single_generation` | 1 gen, 2 children | All |
| `test_directory_structure` | Files created correctly | 3, 7 |
| `test_trial_files_present` | All trial files exist | 3 |
| `test_multi_generation` | 3 generations work | All |
| `test_parent_child_tracking` | Relationships correct | 3, 6 |
| `test_generation_stats` | Stats calculated | 3, 6 |
| `test_max_generations_limit` | Stops at gen limit | 7 |
| `test_max_cost_limit` | Stops at cost limit | 2, 7 |
| `test_max_time_limit` | Stops at time limit | 7 |
| `test_root_terminates` | Root can end early | 6, 7 |
| `test_root_saves_final` | Final state saved | 3, 6 |
| `test_invalid_code_continues` | Handles bad code | 1, 5 |
| `test_eval_fail_continues` | Handles eval fail | 1, 7 |
| `test_root_error_handled` | Handles root errors | 4, 7 |
| `test_resume_continues` | Resume works | 7 |
| `test_resume_costs` | Resume preserves costs | 2, 7 |
| `test_resume_population` | Resume loads pop | 3, 7 |
| `test_final_best_code` | Best code returned | 7 |
| `test_cost_accurate` | Costs match expected | 2 |
| `test_reasoning_recorded` | All reasoning saved | 3, 6 |

---

## Sample Test Code

```python
# tests/test_e2e.py

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from tetris_evolve.evolution.controller import EvolutionController, EvolutionConfig

# Fixtures

@pytest.fixture
def test_config(tmp_path):
    """Minimal config for fast E2E tests."""
    return EvolutionConfig(
        max_generations=3,
        max_cost_usd=1.0,
        max_time_minutes=5,
        max_children_per_generation=5,
        initial_population_size=2,
        games_per_evaluation=2,
        root_model="claude-haiku",
        child_model="claude-haiku",
        root_temperature=0.5,
        child_temperature=0.7,
        output_dir=tmp_path / "experiment"
    )

@pytest.fixture
def mock_llm_responses():
    """Deterministic LLM responses for testing."""
    child_responses = iter([
        # Initial population
        """<reasoning>Simple hard drop</reasoning>
<code>
def choose_action(obs):
    return 5
</code>""",
        """<reasoning>Random moves</reasoning>
<code>
def choose_action(obs):
    import random
    return random.randint(0, 6)
</code>""",
        # Second generation
        """<reasoning>Improved hard drop</reasoning>
<code>
def choose_action(obs):
    return 5 if obs[200] < 10 else 1
</code>""",
    ])

    root_responses = iter([
        # Gen 1: Create population and advance
        """```python
# Create initial population
for i in range(2):
    spawn_child_llm(f"Create player {i}")

# Select best and advance
pop = get_population()
best = sorted(pop, key=lambda x: x.score, reverse=True)[:1]
advance_generation([b.trial_id for b in best], "Selected top performer")
```""",
        # Gen 2: Spawn from parent
        """```python
pop = get_population()
if len(pop) > 0:
    spawn_child_llm("Improve upon parent", parent_id=pop[0].trial_id)

advance_generation([p.trial_id for p in pop[:1]], "Continue evolution")
```""",
        # Gen 3: Terminate
        """```python
history = get_generation_history()
if len(history) >= 3:
    terminate_evolution("Completed 3 generations")
```""",
    ])

    return {"child": child_responses, "root": root_responses}


class MockLLMClient:
    """Mock LLM client for E2E tests."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def send_message(self, messages, **kwargs):
        self.call_count += 1
        try:
            content = next(self.responses)
        except StopIteration:
            content = "<code>def choose_action(obs): return 5</code>"

        return Mock(
            content=content,
            input_tokens=100,
            output_tokens=50,
            model="mock-model",
            stop_reason="end_turn"
        )


def verify_directory_structure(exp_dir: Path) -> dict:
    """Verify experiment directory has expected structure."""
    issues = []

    # Top-level files
    if not (exp_dir / "experiment.json").exists():
        issues.append("Missing experiment.json")
    if not (exp_dir / "cost_history.json").exists():
        issues.append("Missing cost_history.json")
    if not (exp_dir / "generations").exists():
        issues.append("Missing generations directory")

    return {"valid": len(issues) == 0, "issues": issues}


# Tests

class TestMinimalEvolution:
    def test_single_generation_completes(self, test_config, mock_llm_responses):
        """Single generation with 2 children completes successfully."""
        test_config.max_generations = 1

        with patch("tetris_evolve.llm.client.LLMClient") as mock_client:
            mock_client.return_value = MockLLMClient(mock_llm_responses["child"])

            controller = EvolutionController(test_config)
            result = controller.run()

        assert result.generations_completed >= 1
        assert result.termination_reason in ["generation_limit", "root_terminated"]

    def test_directory_structure(self, test_config, mock_llm_responses):
        """Experiment directory created with correct structure."""
        test_config.max_generations = 1

        with patch("tetris_evolve.llm.client.LLMClient"):
            controller = EvolutionController(test_config)
            controller.run()

        verification = verify_directory_structure(test_config.output_dir)
        assert verification["valid"], f"Issues: {verification['issues']}"


class TestMultiGeneration:
    def test_three_generations(self, test_config, mock_llm_responses):
        """Three generations with selection completes."""
        with patch("tetris_evolve.llm.client.LLMClient"):
            controller = EvolutionController(test_config)
            result = controller.run()

        assert result.generations_completed == 3

        # Verify 3 generation folders
        gen_dir = test_config.output_dir / "generations"
        gen_folders = list(gen_dir.glob("gen_*"))
        assert len(gen_folders) == 3


class TestHardLimits:
    def test_max_generations_enforced(self, test_config, mock_llm_responses):
        """Evolution stops at max_generations."""
        test_config.max_generations = 2

        with patch("tetris_evolve.llm.client.LLMClient"):
            controller = EvolutionController(test_config)
            result = controller.run()

        assert result.generations_completed <= 2
        assert "generation" in result.termination_reason.lower() or \
               "terminated" in result.termination_reason.lower()

    def test_max_cost_enforced(self, test_config, mock_llm_responses):
        """Evolution stops when cost limit reached."""
        test_config.max_cost_usd = 0.001  # Very low budget

        with patch("tetris_evolve.llm.client.LLMClient"):
            controller = EvolutionController(test_config)
            result = controller.run()

        assert result.total_cost_usd <= test_config.max_cost_usd + 0.01
        assert "cost" in result.termination_reason.lower()


class TestResume:
    def test_resume_continues(self, test_config, mock_llm_responses):
        """Resume continues from last generation."""
        test_config.max_generations = 2

        with patch("tetris_evolve.llm.client.LLMClient"):
            # Run first 2 generations
            controller = EvolutionController(test_config)
            result1 = controller.run()

            # Update config and resume
            test_config.max_generations = 3
            result2 = controller.resume(test_config.output_dir)

        assert result2.generations_completed == 3


class TestOutputVerification:
    def test_final_result_has_best_code(self, test_config, mock_llm_responses):
        """Final result contains evaluatable best code."""
        with patch("tetris_evolve.llm.client.LLMClient"):
            controller = EvolutionController(test_config)
            result = controller.run()

        assert result.best_code is not None
        assert "choose_action" in result.best_code
        assert result.best_score > 0
```

---

## Mock Strategy

### Child LLM Mock
Returns valid code with `choose_action` function:
- Varies slightly between calls for diversity
- Occasionally returns invalid code to test error handling
- Tracks call count for verification

### Root LLM Mock
Returns Python code that:
1. Calls `spawn_child_llm()` with prompts
2. Uses `get_population()` to see results
3. Calls `advance_generation()` with selection
4. Eventually calls `terminate_evolution()`

### Evaluator Mock (Optional)
For faster tests, mock the evaluator to return predetermined scores instead of running actual Tetris games.

---

## Performance Considerations

- E2E tests are slow by nature
- Use minimal configs: 2 games, 2 children, 2 generations
- Mock LLM calls (no actual API calls)
- Consider mocking evaluator for speed
- Target: Full E2E suite < 60 seconds

---

## Acceptance Criteria

- [x] All 20 E2E tests pass
- [x] Tests complete in < 60 seconds total
- [x] Tests use realistic mock data
- [x] Tests verify actual file output
- [x] Tests cover all termination conditions
- [x] Resume functionality fully tested
- [x] Error handling verified
