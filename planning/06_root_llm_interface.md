# Component 6: Root LLM Interface

## Overview
REPL functions available to the root LLM for controlling evolution.

**File**: `src/tetris_evolve/llm/root_llm.py`
**Test File**: `tests/test_root_llm.py`
**Dependencies**:
- Component 1: Program Evaluator
- Component 2: Cost Tracker
- Component 3: Experiment Tracker
- Component 5: Child LLM Executor

---

## Checklist

### 6.1 Project Setup
- [ ] **6.1.1** Create `root_llm.py` in `src/tetris_evolve/llm/`
  - Dependencies: 4.1.1 (llm folder exists)
- [ ] **6.1.2** Create `tests/test_root_llm.py` skeleton
  - Dependencies: 6.1.1

### 6.2 Data Classes
- [ ] **6.2.1** Define `PopulationMember` dataclass
  - Dependencies: 6.1.1
  ```python
  @dataclass
  class PopulationMember:
      trial_id: str
      generation: int
      parent_id: Optional[str]
      score: float
      lines_cleared: float
      survival_steps: int
      code_preview: str  # First 500 chars
  ```
- [ ] **6.2.2** Define `GenerationSummary` dataclass
  - Dependencies: 6.1.1
  ```python
  @dataclass
  class GenerationSummary:
      generation: int
      best_score: float
      avg_score: float
      num_trials: int
      best_trial_id: str
  ```
- [ ] **6.2.3** Define `ResourceLimitError` exception
  - Dependencies: 6.1.1
- [ ] **6.2.4** Write tests for dataclass creation
  - Dependencies: 6.2.1, 6.2.2

### 6.3 RootLLMInterface Initialization
- [ ] **6.3.1** Implement `RootLLMInterface.__init__(child_executor, evaluator, exp_tracker, cost_tracker, config)`
  - Dependencies: 6.1.1, 1.7.1, 2.4.1, 3.3.1, 5.5.1
  - Store all component references
  - Initialize state tracking
- [ ] **6.3.2** Write tests for initialization
  - Dependencies: 6.3.1

### 6.4 spawn_child_llm Function
- [ ] **6.4.1** Implement `spawn_child_llm(prompt, parent_id) -> dict`
  - Dependencies: 6.3.1, 5.5.2, 1.6.1
  - Check: children limit not exceeded
  - Get parent code if parent_id provided
  - Call child executor
  - Evaluate generated code
  - Save trial to experiment tracker
  - Record cost
  - Return: `{trial_id, code, metrics, reasoning}`
- [ ] **6.4.2** Write tests for `spawn_child_llm` (mocked dependencies)
  - Dependencies: 6.4.1
- [ ] **6.4.3** Implement child limit enforcement
  - Dependencies: 6.4.1
  - Track children spawned this generation
  - Raise `ResourceLimitError` if exceeded
- [ ] **6.4.4** Write tests for child limit
  - Dependencies: 6.4.3

### 6.5 evaluate_program Function
- [ ] **6.5.1** Implement `evaluate_program(code, num_games) -> dict`
  - Dependencies: 6.3.1, 1.6.1
  - Evaluate code without creating trial
  - Return metrics dict
- [ ] **6.5.2** Write tests for `evaluate_program`
  - Dependencies: 6.5.1

### 6.6 Population Queries
- [ ] **6.6.1** Implement `get_population() -> list[PopulationMember]`
  - Dependencies: 6.2.1, 6.3.1, 3.6.1
  - Return current generation's trials with metrics
- [ ] **6.6.2** Write tests for `get_population`
  - Dependencies: 6.6.1
- [ ] **6.6.3** Implement `get_trial_code(trial_id) -> str`
  - Dependencies: 6.3.1, 3.5.4
  - Return full code for a trial
- [ ] **6.6.4** Write tests for `get_trial_code`
  - Dependencies: 6.6.3

### 6.7 History Queries
- [ ] **6.7.1** Implement `get_generation_history() -> list[GenerationSummary]`
  - Dependencies: 6.2.2, 6.3.1, 3.6.3
  - Return stats from all previous generations
- [ ] **6.7.2** Write tests for `get_generation_history`
  - Dependencies: 6.7.1
- [ ] **6.7.3** Implement `get_improvement_rate() -> float`
  - Dependencies: 6.7.1
  - Calculate avg improvement over last N generations
- [ ] **6.7.4** Write tests for `get_improvement_rate`
  - Dependencies: 6.7.3

### 6.8 Generation Advancement
- [ ] **6.8.1** Implement `advance_generation(selected_trial_ids, reasoning) -> int`
  - Dependencies: 6.3.1, 3.4.3
  - Check: generation limit not exceeded
  - Complete current generation in tracker
  - Start new generation
  - Reset children counter
  - Return new generation number
- [ ] **6.8.2** Write tests for `advance_generation`
  - Dependencies: 6.8.1
- [ ] **6.8.3** Implement generation limit enforcement
  - Dependencies: 6.8.1
  - Raise `ResourceLimitError` if exceeded
- [ ] **6.8.4** Write tests for generation limit
  - Dependencies: 6.8.3

### 6.9 Termination
- [ ] **6.9.1** Implement `terminate_evolution(reason) -> dict`
  - Dependencies: 6.3.1, 3.7.1, 3.6.5
  - Mark evolution as terminated
  - Save final stats
  - Return summary with best trial
- [ ] **6.9.2** Write tests for `terminate_evolution`
  - Dependencies: 6.9.1

### 6.10 Resource Queries
- [ ] **6.10.1** Implement `get_cost_remaining() -> float`
  - Dependencies: 6.3.1, 2.5.3
- [ ] **6.10.2** Write tests for `get_cost_remaining`
  - Dependencies: 6.10.1
- [ ] **6.10.3** Implement `get_limits() -> dict`
  - Dependencies: 6.3.1
  - Return: `{max_generations, max_cost, max_children_per_gen, current_gen, children_this_gen}`
- [ ] **6.10.4** Write tests for `get_limits`
  - Dependencies: 6.10.3

### 6.11 State Properties
- [ ] **6.11.1** Implement `is_terminated` property
  - Dependencies: 6.9.1
- [ ] **6.11.2** Implement `current_generation` property
  - Dependencies: 6.8.1
- [ ] **6.11.3** Write tests for state properties
  - Dependencies: 6.11.1, 6.11.2

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_population_member_creation` | PopulationMember works | 6.2.1 |
| `test_generation_summary_creation` | GenerationSummary works | 6.2.2 |
| `test_init` | Interface initializes correctly | 6.3.1 |
| `test_spawn_child_success` | Spawns and evaluates code | 6.4.1 |
| `test_spawn_child_with_parent` | Uses parent code | 6.4.1 |
| `test_spawn_child_limit_exceeded` | Raises error at limit | 6.4.3 |
| `test_evaluate_program` | Evaluates without saving | 6.5.1 |
| `test_get_population` | Returns current gen trials | 6.6.1 |
| `test_get_trial_code` | Returns full code | 6.6.3 |
| `test_get_generation_history` | Returns all gen stats | 6.7.1 |
| `test_get_improvement_rate` | Calculates correctly | 6.7.3 |
| `test_advance_generation` | Advances to next gen | 6.8.1 |
| `test_advance_generation_limit` | Raises at max gen | 6.8.3 |
| `test_terminate_evolution` | Returns summary | 6.9.1 |
| `test_get_cost_remaining` | Returns correct budget | 6.10.1 |
| `test_get_limits` | Returns all limits | 6.10.3 |
| `test_is_terminated` | Tracks termination state | 6.11.1 |

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/llm/root_llm.py

from dataclasses import dataclass
from typing import Optional
from ..evaluation.evaluator import ProgramEvaluator, EvaluationResult
from ..tracking.cost_tracker import CostTracker
from ..tracking.experiment_tracker import ExperimentTracker, TrialData
from .child_llm import ChildLLMExecutor

@dataclass
class PopulationMember:
    trial_id: str
    generation: int
    parent_id: Optional[str]
    score: float
    lines_cleared: float
    survival_steps: int
    code_preview: str

@dataclass
class GenerationSummary:
    generation: int
    best_score: float
    avg_score: float
    num_trials: int
    best_trial_id: str

class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""
    pass

class RootLLMInterface:
    def __init__(
        self,
        child_executor: ChildLLMExecutor,
        evaluator: ProgramEvaluator,
        experiment_tracker: ExperimentTracker,
        cost_tracker: CostTracker,
        config: dict
    ):
        self.child_executor = child_executor
        self.evaluator = evaluator
        self.exp_tracker = experiment_tracker
        self.cost_tracker = cost_tracker
        self.config = config

        self._current_generation = 1
        self._children_this_gen = 0
        self._terminated = False

        # Limits from config
        self._max_generations = config.get("max_generations", 50)
        self._max_children_per_gen = config.get("max_children_per_generation", 20)

    def spawn_child_llm(
        self,
        prompt: str,
        parent_id: Optional[str] = None
    ) -> dict:
        """Spawn child LLM to generate code."""
        pass

    def evaluate_program(self, code: str, num_games: int = 10) -> dict:
        """Evaluate code without creating trial."""
        pass

    def get_population(self) -> list[PopulationMember]:
        """Get current generation's programs."""
        pass

    def get_trial_code(self, trial_id: str) -> str:
        """Get full code for a trial."""
        pass

    def get_generation_history(self) -> list[GenerationSummary]:
        """Get stats from all previous generations."""
        pass

    def get_improvement_rate(self, window: int = 3) -> float:
        """Calculate improvement rate over last N generations."""
        pass

    def advance_generation(
        self,
        selected_trial_ids: list[str],
        reasoning: str
    ) -> int:
        """Advance to next generation."""
        pass

    def terminate_evolution(self, reason: str) -> dict:
        """End evolution and return summary."""
        pass

    def get_cost_remaining(self) -> float:
        """Get remaining budget."""
        pass

    def get_limits(self) -> dict:
        """Get all resource limits and current state."""
        pass

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    @property
    def current_generation(self) -> int:
        return self._current_generation
```

---

## Function Call Examples (for Root LLM)

```python
# Root LLM would generate code like this:

# Check limits
limits = get_limits()
print(f"Generation {limits['current_gen']}/{limits['max_generations']}")
print(f"Children this gen: {limits['children_this_gen']}/{limits['max_children_per_gen']}")
print(f"Budget remaining: ${get_cost_remaining():.2f}")

# Get current population
population = get_population()
for p in population:
    print(f"{p.trial_id}: score={p.score:.1f}, lines={p.lines_cleared:.1f}")

# Spawn child with custom prompt
result = spawn_child_llm(
    prompt="Focus on reducing holes. The current best creates ~15 holes per game.",
    parent_id="trial_003"
)
print(f"Created {result['trial_id']} with score {result['metrics']['avg_score']}")

# Get history and check improvement
history = get_generation_history()
rate = get_improvement_rate()
print(f"Improvement rate: {rate:.1%}")

if rate < 0.01 and len(history) > 5:
    terminate_evolution("Converged: improvement rate below 1% for 5 generations")
else:
    advance_generation(
        selected_trial_ids=["trial_003", "trial_007"],
        reasoning="Selected top 2 performers for next generation"
    )
```

---

## Acceptance Criteria

- [ ] All 17 tests pass
- [ ] Code coverage > 90%
- [ ] All functions work with mocked dependencies
- [ ] Resource limits properly enforced
- [ ] State tracked correctly across generations
- [ ] Integration with all dependent components
