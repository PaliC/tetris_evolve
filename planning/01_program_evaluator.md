# Component 1: Program Evaluator

## Overview
Safely executes evolved code and measures Tetris performance using PufferLib.

**File**: `src/tetris_evolve/evaluation/evaluator.py`
**Test File**: `tests/test_evaluator.py`
**Dependencies**: PufferLib (external)

---

## Checklist

### 1.1 Project Setup
- [x] **1.1.1** Create directory structure `src/tetris_evolve/evaluation/`
  - Dependencies: None
- [x] **1.1.2** Create `__init__.py` files for package
  - Dependencies: 1.1.1
- [x] **1.1.3** Create `tests/test_evaluator.py` skeleton
  - Dependencies: 1.1.1
- [x] **1.1.4** Verify PufferLib Tetris import works
  - Dependencies: None (external dep)

### 1.2 Data Classes
- [x] **1.2.1** Define `GameResult` dataclass
  - Dependencies: 1.1.2
  ```python
  @dataclass
  class GameResult:
      score: int
      lines_cleared: int
      steps: int
      ep_return: float
      terminated_reason: str  # "overflow", "max_ticks", "error"
  ```
- [x] **1.2.2** Define `EvaluationResult` dataclass
  - Dependencies: 1.2.1
  ```python
  @dataclass
  class EvaluationResult:
      trial_id: str
      success: bool
      error: Optional[str]
      games_played: int
      avg_score: Optional[float]
      avg_lines_cleared: Optional[float]
      avg_survival_steps: Optional[int]
      max_score: Optional[int]
      min_score: Optional[int]
      game_results: list[GameResult]
  ```
- [x] **1.2.3** Write tests for dataclass creation
  - Dependencies: 1.2.1, 1.2.2

### 1.3 Code Validation
- [x] **1.3.1** Implement `_validate_syntax(code: str) -> tuple[bool, str]`
  - Dependencies: 1.1.2
  - Uses `ast.parse()` to check syntax
- [x] **1.3.2** Write tests for syntax validation (valid, invalid, edge cases)
  - Dependencies: 1.3.1
- [x] **1.3.3** Implement `_check_safety(code: str) -> tuple[bool, str]`
  - Dependencies: 1.1.2
  - Block: `os`, `subprocess`, `sys`, `eval`, `exec`, `__import__`
  - Block: file operations, network access
- [x] **1.3.4** Write tests for safety checks
  - Dependencies: 1.3.3
- [x] **1.3.5** Implement `_validate_interface(code: str) -> tuple[bool, str]`
  - Dependencies: 1.1.2
  - Check: `choose_action` function exists
  - Check: function signature is correct
- [x] **1.3.6** Write tests for interface validation
  - Dependencies: 1.3.5

### 1.4 Code Execution
- [x] **1.4.1** Implement `_execute_code(code: str) -> Callable`
  - Dependencies: 1.3.1, 1.3.3, 1.3.5
  - Execute in restricted namespace
  - Extract `choose_action` function
- [x] **1.4.2** Write tests for code execution (valid code extracts function)
  - Dependencies: 1.4.1
- [x] **1.4.3** Write tests for execution errors (runtime errors caught)
  - Dependencies: 1.4.1

### 1.5 Single Game Execution
- [x] **1.5.1** Implement `_run_single_game(player_fn, env, max_steps) -> GameResult`
  - Dependencies: 1.2.1, 1.4.1, 1.1.4
  - Create PufferLib Tetris env
  - Run until terminal or max_steps
  - Catch and handle player errors
- [x] **1.5.2** Write tests for single game (completes, returns valid GameResult)
  - Dependencies: 1.5.1
- [x] **1.5.3** Implement timeout handling for slow player functions
  - Dependencies: 1.5.1
  - Use `signal.alarm` or `multiprocessing` with timeout
- [x] **1.5.4** Write tests for timeout (infinite loop terminated)
  - Dependencies: 1.5.3

### 1.6 Multi-Game Evaluation
- [x] **1.6.1** Implement `evaluate(code: str, trial_id: str, num_games: int) -> EvaluationResult`
  - Dependencies: 1.2.2, 1.4.1, 1.5.1
  - Run num_games episodes
  - Aggregate results
  - Handle partial failures gracefully
- [x] **1.6.2** Write tests for multi-game evaluation
  - Dependencies: 1.6.1
- [x] **1.6.3** Implement result aggregation (avg, min, max, std)
  - Dependencies: 1.6.1
- [x] **1.6.4** Write tests for aggregation correctness
  - Dependencies: 1.6.3

### 1.7 ProgramEvaluator Class
- [x] **1.7.1** Implement `ProgramEvaluator.__init__(num_games, max_steps, timeout_seconds)`
  - Dependencies: 1.6.1
- [x] **1.7.2** Implement `ProgramEvaluator.evaluate(code, trial_id)` public method
  - Dependencies: 1.7.1, 1.6.1
- [x] **1.7.3** Write integration test: valid code → metrics
  - Dependencies: 1.7.2
- [x] **1.7.4** Write integration test: syntax error → error result
  - Dependencies: 1.7.2
- [x] **1.7.5** Write integration test: runtime error → error result
  - Dependencies: 1.7.2
- [x] **1.7.6** Write integration test: timeout → error result
  - Dependencies: 1.7.2
- [x] **1.7.7** Write integration test: unsafe code → error result
  - Dependencies: 1.7.2

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_game_result_dataclass` | GameResult creates correctly | 1.2.1 |
| `test_evaluation_result_dataclass` | EvaluationResult creates correctly | 1.2.2 |
| `test_syntax_valid` | Valid code passes syntax check | 1.3.1 |
| `test_syntax_invalid` | Invalid code fails syntax check | 1.3.1 |
| `test_safety_blocks_os` | `import os` is blocked | 1.3.3 |
| `test_safety_blocks_subprocess` | `subprocess` is blocked | 1.3.3 |
| `test_safety_blocks_eval` | `eval()` is blocked | 1.3.3 |
| `test_interface_valid` | Valid `choose_action` passes | 1.3.5 |
| `test_interface_missing_function` | Missing function fails | 1.3.5 |
| `test_execute_extracts_function` | Execution returns callable | 1.4.1 |
| `test_execute_runtime_error` | Runtime errors caught | 1.4.1 |
| `test_single_game_completes` | Game runs to completion | 1.5.1 |
| `test_single_game_timeout` | Infinite loop terminated | 1.5.3 |
| `test_evaluate_valid_code` | Full evaluation returns metrics | 1.6.1 |
| `test_evaluate_aggregation` | Aggregation is correct | 1.6.3 |
| `test_evaluator_syntax_error` | Evaluator handles syntax error | 1.7.4 |
| `test_evaluator_runtime_error` | Evaluator handles runtime error | 1.7.5 |
| `test_evaluator_timeout` | Evaluator handles timeout | 1.7.6 |
| `test_evaluator_unsafe_code` | Evaluator blocks unsafe code | 1.7.7 |

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/evaluation/evaluator.py

import ast
import signal
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

@dataclass
class GameResult:
    score: int
    lines_cleared: int
    steps: int
    ep_return: float
    terminated_reason: str

@dataclass
class EvaluationResult:
    trial_id: str
    success: bool
    error: Optional[str]
    games_played: int
    avg_score: Optional[float]
    avg_lines_cleared: Optional[float]
    avg_survival_steps: Optional[int]
    max_score: Optional[int]
    min_score: Optional[int]
    game_results: list[GameResult]

BLOCKED_IMPORTS = {'os', 'subprocess', 'sys', 'socket', 'urllib', 'requests'}
BLOCKED_BUILTINS = {'eval', 'exec', '__import__', 'open', 'compile'}

class ProgramEvaluator:
    def __init__(
        self,
        num_games: int = 10,
        max_steps: int = 10000,
        timeout_seconds: int = 30
    ):
        self.num_games = num_games
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

    def evaluate(self, code: str, trial_id: str) -> EvaluationResult:
        # 1. Validate syntax
        # 2. Check safety
        # 3. Validate interface
        # 4. Execute code
        # 5. Run games
        # 6. Aggregate results
        pass

    def _validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        pass

    def _check_safety(self, code: str) -> tuple[bool, Optional[str]]:
        pass

    def _validate_interface(self, code: str) -> tuple[bool, Optional[str]]:
        pass

    def _execute_code(self, code: str) -> Callable:
        pass

    def _run_single_game(self, player_fn: Callable) -> GameResult:
        pass
```

---

## Acceptance Criteria

- [x] All 19 tests pass
- [x] Code coverage > 90%
- [x] Can evaluate a simple hard-drop agent
- [x] Properly handles all error cases
- [x] Timeout works for infinite loops
- [x] Safety checks block dangerous code
