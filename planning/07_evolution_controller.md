# Component 7: Evolution Controller

## Overview
Main orchestrator that runs the evolution loop and enforces hard limits.

**File**: `src/tetris_evolve/evolution/controller.py`
**Test File**: `tests/test_controller.py`
**Dependencies**: ALL previous components (1-6)

---

## Checklist

### 7.1 Project Setup
- [ ] **7.1.1** Create directory structure `src/tetris_evolve/evolution/`
  - Dependencies: None
- [ ] **7.1.2** Create `__init__.py` files for package
  - Dependencies: 7.1.1
- [ ] **7.1.3** Create `tests/test_controller.py` skeleton
  - Dependencies: 7.1.1

### 7.2 Configuration
- [ ] **7.2.1** Define `EvolutionConfig` dataclass
  - Dependencies: 7.1.2
  ```python
  @dataclass
  class EvolutionConfig:
      max_generations: int
      max_cost_usd: float
      max_time_minutes: int
      max_children_per_generation: int
      initial_population_size: int
      games_per_evaluation: int
      root_model: str
      child_model: str
      root_temperature: float
      child_temperature: float
      output_dir: Path
  ```
- [ ] **7.2.2** Implement `EvolutionConfig.from_yaml(path) -> EvolutionConfig`
  - Dependencies: 7.2.1
  - Load config from YAML file
- [ ] **7.2.3** Write tests for config loading
  - Dependencies: 7.2.2
- [ ] **7.2.4** Implement config validation
  - Dependencies: 7.2.1
  - Check all required fields present
  - Check values are valid (positive, within ranges)
- [ ] **7.2.5** Write tests for config validation
  - Dependencies: 7.2.4

### 7.3 Result Data Classes
- [ ] **7.3.1** Define `EvolutionResult` dataclass
  - Dependencies: 7.1.2
  ```python
  @dataclass
  class EvolutionResult:
      experiment_id: str
      generations_completed: int
      total_cost_usd: float
      total_time_minutes: float
      best_trial_id: str
      best_score: float
      best_code: str
      termination_reason: str
  ```
- [ ] **7.3.2** Write tests for result dataclass
  - Dependencies: 7.3.1

### 7.4 EvolutionController Initialization
- [ ] **7.4.1** Implement `EvolutionController.__init__(config)`
  - Dependencies: 7.2.1
  - Initialize all trackers (cost, experiment)
  - Initialize LLM clients
  - Initialize evaluator
  - Initialize root interface
- [ ] **7.4.2** Write tests for initialization (mocked dependencies)
  - Dependencies: 7.4.1
- [ ] **7.4.3** Implement component wiring
  - Dependencies: 7.4.1, 1-6 all
  - Connect: evaluator → child executor → root interface

### 7.5 Initial Population
- [ ] **7.5.1** Define initial prompts list
  - Dependencies: 7.1.2
  ```python
  INITIAL_PROMPTS = [
      "Create a Tetris player that minimizes holes",
      "Create a Tetris player focused on clearing lines quickly",
      "Create a Tetris player that keeps the board flat",
      "Create a Tetris player using piece lookahead",
      "Create a simple random baseline Tetris player",
  ]
  ```
- [ ] **7.5.2** Implement `_create_initial_population()`
  - Dependencies: 7.5.1, 6.4.1
  - Use root interface to spawn initial children
  - No parent_id for initial population
- [ ] **7.5.3** Write tests for initial population
  - Dependencies: 7.5.2

### 7.6 Hard Limit Checking
- [ ] **7.6.1** Implement `_check_generation_limit() -> Optional[str]`
  - Dependencies: 7.4.1
  - Return termination reason if exceeded, None otherwise
- [ ] **7.6.2** Write tests for generation limit
  - Dependencies: 7.6.1
- [ ] **7.6.3** Implement `_check_cost_limit() -> Optional[str]`
  - Dependencies: 7.4.1, 2.5.1
  - Return termination reason if exceeded
- [ ] **7.6.4** Write tests for cost limit
  - Dependencies: 7.6.3
- [ ] **7.6.5** Implement `_check_time_limit() -> Optional[str]`
  - Dependencies: 7.4.1
  - Track wall-clock time since start
  - Return termination reason if exceeded
- [ ] **7.6.6** Write tests for time limit
  - Dependencies: 7.6.5
- [ ] **7.6.7** Implement `_check_all_limits() -> Optional[str]`
  - Dependencies: 7.6.1, 7.6.3, 7.6.5
  - Check all limits, return first exceeded
- [ ] **7.6.8** Write tests for combined limit checking
  - Dependencies: 7.6.7

### 7.7 Root LLM Turn Execution
- [ ] **7.7.1** Define `ROOT_SYSTEM_PROMPT` constant
  - Dependencies: 7.1.2
  - Full system prompt with available functions
  - Current state placeholders
- [ ] **7.7.2** Implement `_build_root_prompt() -> str`
  - Dependencies: 7.7.1
  - Fill in current state (generation, budget, population)
- [ ] **7.7.3** Write tests for prompt building
  - Dependencies: 7.7.2
- [ ] **7.7.4** Implement `_execute_root_code(code: str)`
  - Dependencies: 7.4.1, 6.3.1
  - Execute Python code in context with root interface functions
  - Catch and handle errors
- [ ] **7.7.5** Write tests for root code execution
  - Dependencies: 7.7.4
- [ ] **7.7.6** Implement `_run_root_llm_turn() -> bool`
  - Dependencies: 7.7.2, 7.7.4, 4.4.1
  - Call root LLM with prompt
  - Execute returned code
  - Record cost
  - Return False if terminated
- [ ] **7.7.7** Write tests for root LLM turn
  - Dependencies: 7.7.6

### 7.8 Main Evolution Loop
- [ ] **7.8.1** Implement `run() -> EvolutionResult`
  - Dependencies: 7.5.2, 7.6.7, 7.7.6
  - Initialize experiment
  - Create initial population
  - Loop: check limits → run root turn → repeat
  - Return final result
- [ ] **7.8.2** Write tests for `run()` with mocked components
  - Dependencies: 7.8.1
- [ ] **7.8.3** Implement logging throughout loop
  - Dependencies: 7.8.1
  - Log: generation start, trial created, generation end
- [ ] **7.8.4** Implement graceful shutdown on interrupt
  - Dependencies: 7.8.1
  - Handle SIGINT/SIGTERM
  - Save current state before exit

### 7.9 Resume Support
- [ ] **7.9.1** Implement `resume(experiment_dir) -> EvolutionResult`
  - Dependencies: 7.8.1, 3.8.1
  - Load existing experiment
  - Continue from last generation
- [ ] **7.9.2** Write tests for resume
  - Dependencies: 7.9.1

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_config_from_yaml` | Loads config correctly | 7.2.2 |
| `test_config_validation_valid` | Valid config passes | 7.2.4 |
| `test_config_validation_invalid` | Invalid config fails | 7.2.4 |
| `test_result_dataclass` | EvolutionResult works | 7.3.1 |
| `test_init` | Controller initializes | 7.4.1 |
| `test_initial_population` | Creates N initial trials | 7.5.2 |
| `test_generation_limit` | Detects at max gen | 7.6.1 |
| `test_cost_limit` | Detects over budget | 7.6.3 |
| `test_time_limit` | Detects time exceeded | 7.6.5 |
| `test_build_root_prompt` | Prompt has current state | 7.7.2 |
| `test_execute_root_code` | Executes safely | 7.7.4 |
| `test_root_turn_spawns_child` | Root can spawn | 7.7.6 |
| `test_root_turn_advances` | Root can advance gen | 7.7.6 |
| `test_root_turn_terminates` | Root can terminate | 7.7.6 |
| `test_run_completes` | Full run returns result | 7.8.1 |
| `test_run_respects_limits` | Stops at limits | 7.8.1 |
| `test_resume` | Can resume experiment | 7.9.1 |

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/evolution/controller.py

import signal
import time
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..evaluation.evaluator import ProgramEvaluator
from ..tracking.cost_tracker import CostTracker
from ..tracking.experiment_tracker import ExperimentTracker
from ..llm.client import LLMClient
from ..llm.child_llm import ChildLLMExecutor
from ..llm.root_llm import RootLLMInterface

@dataclass
class EvolutionConfig:
    max_generations: int
    max_cost_usd: float
    max_time_minutes: int
    max_children_per_generation: int
    initial_population_size: int
    games_per_evaluation: int
    root_model: str
    child_model: str
    root_temperature: float
    child_temperature: float
    output_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "EvolutionConfig":
        pass

    def validate(self) -> None:
        pass

@dataclass
class EvolutionResult:
    experiment_id: str
    generations_completed: int
    total_cost_usd: float
    total_time_minutes: float
    best_trial_id: str
    best_score: float
    best_code: str
    termination_reason: str

INITIAL_PROMPTS = [
    "Create a Tetris player that minimizes holes in the board",
    "Create a Tetris player that clears lines as quickly as possible",
    "Create a Tetris player that keeps the board as flat as possible",
    "Create a Tetris player that uses 1-piece lookahead",
    "Create a simple baseline Tetris player that makes random moves",
]

class EvolutionController:
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.start_time: Optional[datetime] = None

        # Initialize components
        self.cost_tracker = CostTracker(config.max_cost_usd)
        self.exp_tracker = ExperimentTracker(config.output_dir)
        self.evaluator = ProgramEvaluator(config.games_per_evaluation)

        # LLM clients
        self.root_client = LLMClient(config.root_model)
        self.child_client = LLMClient(config.child_model)

        # Child executor
        self.child_executor = ChildLLMExecutor(
            self.child_client,
            temperature=config.child_temperature
        )

        # Root interface
        self.root_interface = RootLLMInterface(
            self.child_executor,
            self.evaluator,
            self.exp_tracker,
            self.cost_tracker,
            config.__dict__
        )

    def run(self) -> EvolutionResult:
        """Main evolution loop."""
        pass

    def resume(self, experiment_dir: Path) -> EvolutionResult:
        """Resume from existing experiment."""
        pass

    def _create_initial_population(self) -> None:
        pass

    def _check_generation_limit(self) -> Optional[str]:
        pass

    def _check_cost_limit(self) -> Optional[str]:
        pass

    def _check_time_limit(self) -> Optional[str]:
        pass

    def _check_all_limits(self) -> Optional[str]:
        pass

    def _build_root_prompt(self) -> str:
        pass

    def _execute_root_code(self, code: str) -> None:
        pass

    def _run_root_llm_turn(self) -> bool:
        pass
```

---

## Root LLM Execution Context

The root LLM code executes with these functions available:

```python
# Functions injected into execution context
context = {
    "spawn_child_llm": self.root_interface.spawn_child_llm,
    "evaluate_program": self.root_interface.evaluate_program,
    "get_population": self.root_interface.get_population,
    "get_trial_code": self.root_interface.get_trial_code,
    "get_generation_history": self.root_interface.get_generation_history,
    "get_improvement_rate": self.root_interface.get_improvement_rate,
    "advance_generation": self.root_interface.advance_generation,
    "terminate_evolution": self.root_interface.terminate_evolution,
    "get_cost_remaining": self.root_interface.get_cost_remaining,
    "get_limits": self.root_interface.get_limits,
    # Standard libraries
    "print": print,
    "len": len,
    "max": max,
    "min": min,
    "sorted": sorted,
    "sum": sum,
}
```

---

## Acceptance Criteria

- [ ] All 17 tests pass
- [ ] Code coverage > 90%
- [ ] Full evolution loop runs with mocked LLMs
- [ ] All hard limits enforced correctly
- [ ] Graceful shutdown on interrupt
- [ ] Resume from interrupted experiment works
