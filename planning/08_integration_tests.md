# Component 8: Integration Tests

## Overview
Tests that verify multiple components work together correctly.

**File**: `tests/test_integration.py`
**Dependencies**: Components 1-7 (all implementation complete)

---

## Checklist

### 8.1 Test Setup
- [x] **8.1.1** Create `tests/test_integration.py`
  - Dependencies: 1-7 all complete
- [x] **8.1.2** Create shared fixtures for component mocking
  - Dependencies: 8.1.1
- [x] **8.1.3** Create helper functions for test data generation
  - Dependencies: 8.1.1

### 8.2 Evaluator → Tracker Integration
- [x] **8.2.1** Test: Evaluation results save correctly to experiment tracker
  - Dependencies: 1.6.1, 3.5.2
  - Evaluate code → save trial → verify files
- [x] **8.2.2** Test: Multiple evaluations tracked with unique IDs
  - Dependencies: 8.2.1
- [x] **8.2.3** Test: Failed evaluations recorded with error info
  - Dependencies: 8.2.1

### 8.3 Child LLM → Evaluator Pipeline
- [x] **8.3.1** Test: Child generates code → code evaluates successfully
  - Dependencies: 5.5.2, 1.6.1
  - Mock LLM to return valid code
  - Verify evaluation runs
- [x] **8.3.2** Test: Child generates invalid code → error handled
  - Dependencies: 8.3.1
  - Mock LLM to return syntax error code
  - Verify error captured in result
- [x] **8.3.3** Test: Child retry produces valid code on second attempt
  - Dependencies: 5.6.3, 1.6.1
  - Mock LLM: first call invalid, second valid

### 8.4 Root → Child Pipeline
- [x] **8.4.1** Test: Root spawn_child creates trial and evaluates
  - Dependencies: 6.4.1, 5.5.2, 1.6.1
  - Call spawn_child_llm
  - Verify trial saved with code, metrics, reasoning
- [x] **8.4.2** Test: Root spawn_child with parent uses parent code
  - Dependencies: 8.4.1
  - Create parent trial
  - Spawn child with parent_id
  - Verify parent code in prompt
- [x] **8.4.3** Test: Root spawn_child updates cost tracker
  - Dependencies: 6.4.1, 2.4.2
  - Spawn child
  - Verify cost recorded

### 8.5 Generation Lifecycle
- [x] **8.5.1** Test: Full generation cycle (spawn → evaluate → advance)
  - Dependencies: 6.4.1, 6.8.1, 3.4.3
  - Spawn multiple children
  - Advance generation
  - Verify all files created
- [x] **8.5.2** Test: Generation stats calculated correctly
  - Dependencies: 8.5.1
  - Verify best_score, avg_score, num_trials
- [x] **8.5.3** Test: Selected parents recorded in generation
  - Dependencies: 8.5.1
  - Verify selected_parents.json content

### 8.6 Cost → Controller Integration
- [x] **8.6.1** Test: Controller stops when cost limit reached
  - Dependencies: 7.6.3, 2.5.5
  - Set low budget
  - Run controller
  - Verify stops with cost reason
- [x] **8.6.2** Test: Cost accumulates across generations
  - Dependencies: 8.6.1
  - Run multiple generations
  - Verify total cost sums correctly

### 8.7 Full Pipeline (Mocked LLM)
- [x] **8.7.1** Test: Initial population creates N trials
  - Dependencies: 7.5.2
  - Run with mocked LLM
  - Verify N trials created
- [x] **8.7.2** Test: Controller executes root code correctly
  - Dependencies: 7.7.6
  - Mock root LLM response
  - Verify functions called as expected

---

## Test Summary

| Test | Description | Components |
|------|-------------|------------|
| `test_eval_saves_to_tracker` | Results persist correctly | 1, 3 |
| `test_multiple_evals_unique_ids` | IDs are unique | 1, 3 |
| `test_failed_eval_recorded` | Errors captured | 1, 3 |
| `test_child_generates_evaluates` | Generation → Evaluation | 5, 1 |
| `test_child_invalid_handled` | Invalid code handled | 5, 1 |
| `test_child_retry_succeeds` | Retry produces valid | 5, 1 |
| `test_spawn_child_creates_trial` | Full spawn pipeline | 6, 5, 1, 3 |
| `test_spawn_with_parent` | Parent code included | 6, 5, 3 |
| `test_spawn_updates_cost` | Cost tracked | 6, 2 |
| `test_generation_lifecycle` | Full gen cycle | 6, 3 |
| `test_generation_stats` | Stats calculated | 6, 3 |
| `test_selected_parents_recorded` | Parents saved | 6, 3 |
| `test_cost_limit_stops` | Cost limit works | 7, 2 |
| `test_cost_accumulates` | Costs sum | 7, 2 |
| `test_initial_population` | Creates N trials | 7, 6 |
| `test_root_code_executes` | Root code works | 7, 6 |

---

## Sample Test Code

```python
# tests/test_integration.py

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from tetris_evolve.evaluation.evaluator import ProgramEvaluator
from tetris_evolve.tracking.experiment_tracker import ExperimentTracker, TrialData
from tetris_evolve.tracking.cost_tracker import CostTracker
from tetris_evolve.llm.child_llm import ChildLLMExecutor
from tetris_evolve.llm.root_llm import RootLLMInterface

# Fixtures

@pytest.fixture
def tmp_experiment(tmp_path):
    """Create temporary experiment directory."""
    tracker = ExperimentTracker(tmp_path)
    tracker.create_experiment({"test": True})
    tracker.start_generation(1)
    return tracker

@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns valid code."""
    client = Mock()
    client.send_message.return_value = Mock(
        content="""
<reasoning>Simple hard drop strategy</reasoning>
<code>
def choose_action(obs):
    return 5  # hard drop
</code>
""",
        input_tokens=100,
        output_tokens=50,
        model="test-model",
        stop_reason="end_turn"
    )
    return client

@pytest.fixture
def evaluator():
    """Real evaluator with low game count for speed."""
    return ProgramEvaluator(num_games=2, max_steps=100)

# Tests

class TestEvaluatorTrackerIntegration:
    def test_eval_saves_to_tracker(self, tmp_experiment, evaluator):
        """Evaluation results save correctly to experiment tracker."""
        code = "def choose_action(obs): return 5"
        result = evaluator.evaluate(code, "trial_001")

        trial = TrialData(
            trial_id="trial_001",
            generation=1,
            parent_id=None,
            code=code,
            prompt="Test prompt",
            reasoning="Test reasoning",
            llm_response="Test response",
            metrics=result.__dict__ if result.success else None,
            timestamp=datetime.now()
        )

        path = tmp_experiment.save_trial(trial)

        # Verify files
        assert (path / "code.py").exists()
        assert (path / "metrics.json").exists()

    def test_failed_eval_recorded(self, tmp_experiment, evaluator):
        """Failed evaluations recorded with error info."""
        code = "def choose_action(obs) return 5"  # Syntax error
        result = evaluator.evaluate(code, "trial_002")

        assert result.success is False
        assert result.error is not None


class TestChildEvaluatorPipeline:
    def test_child_generates_evaluates(self, mock_llm_client, evaluator):
        """Child generates code that evaluates successfully."""
        executor = ChildLLMExecutor(mock_llm_client)
        result = executor.generate("Create a simple player")

        assert result.success
        assert "choose_action" in result.code

        eval_result = evaluator.evaluate(result.code, "trial_001")
        assert eval_result.success


class TestRootChildPipeline:
    def test_spawn_child_creates_trial(
        self, tmp_experiment, mock_llm_client, evaluator
    ):
        """Root spawn_child creates trial and evaluates."""
        cost_tracker = CostTracker(100.0)
        executor = ChildLLMExecutor(mock_llm_client)

        interface = RootLLMInterface(
            child_executor=executor,
            evaluator=evaluator,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 5}
        )

        result = interface.spawn_child_llm("Create a player")

        assert "trial_id" in result
        assert "metrics" in result
        assert cost_tracker.get_total_cost() > 0


class TestGenerationLifecycle:
    def test_full_generation_cycle(
        self, tmp_experiment, mock_llm_client, evaluator
    ):
        """Full generation cycle works end-to-end."""
        cost_tracker = CostTracker(100.0)
        executor = ChildLLMExecutor(mock_llm_client)

        interface = RootLLMInterface(
            child_executor=executor,
            evaluator=evaluator,
            experiment_tracker=tmp_experiment,
            cost_tracker=cost_tracker,
            config={"max_generations": 10, "max_children_per_generation": 5}
        )

        # Spawn children
        trial_ids = []
        for i in range(3):
            result = interface.spawn_child_llm(f"Create player {i}")
            trial_ids.append(result["trial_id"])

        # Advance generation
        new_gen = interface.advance_generation(
            selected_trial_ids=trial_ids[:2],
            reasoning="Selected top 2"
        )

        assert new_gen == 2

        # Verify files
        gen_path = tmp_experiment.experiment_dir / "generations" / "gen_001"
        assert (gen_path / "generation_stats.json").exists()
        assert (gen_path / "root_llm_reasoning.md").exists()
```

---

## Acceptance Criteria

- [x] All 16 integration tests pass
- [x] Tests use real components (not all mocked)
- [x] Tests verify cross-component data flow
- [x] Tests complete in reasonable time (<30s total)
- [x] Coverage of critical integration points
