# Component 2: Cost Tracker

## Overview
Tracks all LLM API costs with per-call granularity and enforces budget limits.

**File**: `src/tetris_evolve/tracking/cost_tracker.py`
**Test File**: `tests/test_cost_tracker.py`
**Dependencies**: None (standalone)

---

## Checklist

### 2.1 Project Setup
- [x] **2.1.1** Create directory structure `src/tetris_evolve/tracking/`
  - Dependencies: None
- [x] **2.1.2** Create `__init__.py` files for package
  - Dependencies: 2.1.1
- [x] **2.1.3** Create `tests/test_cost_tracker.py` skeleton
  - Dependencies: 2.1.1

### 2.2 Data Classes
- [x] **2.2.1** Define `LLMCall` dataclass
  - Dependencies: 2.1.2
  ```python
  @dataclass
  class LLMCall:
      timestamp: datetime
      model: str
      role: str  # "root" or "child"
      generation: int
      trial_id: Optional[str]
      input_tokens: int
      output_tokens: int
      cost_usd: float
  ```
- [x] **2.2.2** Write tests for `LLMCall` creation
  - Dependencies: 2.2.1

### 2.3 Cost Calculation
- [x] **2.3.1** Define default cost config structure
  - Dependencies: 2.1.2
  ```python
  DEFAULT_COST_CONFIG = {
      "claude-sonnet-4": {"input": 0.003, "output": 0.015},
      "claude-haiku": {"input": 0.00025, "output": 0.00125},
  }
  ```
- [x] **2.3.2** Implement `_calculate_cost(model, input_tokens, output_tokens) -> float`
  - Dependencies: 2.3.1
  - Formula: `(input_tokens * input_rate + output_tokens * output_rate) / 1000`
- [x] **2.3.3** Write tests for cost calculation (various models, token counts)
  - Dependencies: 2.3.2
- [x] **2.3.4** Handle unknown model gracefully (raise or use default)
  - Dependencies: 2.3.2
- [x] **2.3.5** Write tests for unknown model handling
  - Dependencies: 2.3.4

### 2.4 CostTracker Class - Core
- [x] **2.4.1** Implement `CostTracker.__init__(max_cost_usd, cost_config)`
  - Dependencies: 2.3.1
- [x] **2.4.2** Implement `record_call(model, role, generation, input_tokens, output_tokens, trial_id) -> LLMCall`
  - Dependencies: 2.2.1, 2.3.2, 2.4.1
  - Creates `LLMCall` with calculated cost
  - Appends to internal list
  - Returns the created call
- [x] **2.4.3** Write tests for `record_call`
  - Dependencies: 2.4.2

### 2.5 CostTracker Class - Queries
- [x] **2.5.1** Implement `get_total_cost() -> float`
  - Dependencies: 2.4.2
  - Sum of all call costs
- [x] **2.5.2** Write tests for `get_total_cost`
  - Dependencies: 2.5.1
- [x] **2.5.3** Implement `get_remaining_budget() -> float`
  - Dependencies: 2.5.1
  - `max_cost_usd - get_total_cost()`
- [x] **2.5.4** Write tests for `get_remaining_budget`
  - Dependencies: 2.5.3
- [x] **2.5.5** Implement `would_exceed_budget(estimated_cost: float) -> bool`
  - Dependencies: 2.5.1
  - `get_total_cost() + estimated_cost > max_cost_usd`
- [x] **2.5.6** Write tests for `would_exceed_budget`
  - Dependencies: 2.5.5

### 2.6 CostTracker Class - Summaries
- [x] **2.6.1** Implement `get_summary() -> dict`
  - Dependencies: 2.4.2
  - Returns breakdown by role (root/child)
  - Total calls, tokens, cost per role
- [x] **2.6.2** Write tests for `get_summary`
  - Dependencies: 2.6.1
- [x] **2.6.3** Implement `get_per_generation_costs() -> list[dict]`
  - Dependencies: 2.4.2
  - Returns cost breakdown per generation
- [x] **2.6.4** Write tests for `get_per_generation_costs`
  - Dependencies: 2.6.3

### 2.7 Persistence
- [x] **2.7.1** Implement `save(path: Path) -> None`
  - Dependencies: 2.4.2, 2.6.1, 2.6.3
  - Save as JSON with full call history and summaries
- [x] **2.7.2** Write tests for `save` (file created, valid JSON)
  - Dependencies: 2.7.1
- [x] **2.7.3** Implement `load(path: Path, cost_config: dict) -> CostTracker` (classmethod)
  - Dependencies: 2.7.1
  - Load from JSON, reconstruct state
- [x] **2.7.4** Write tests for `load` (round-trip save/load)
  - Dependencies: 2.7.3
- [x] **2.7.5** Handle missing/corrupted file gracefully
  - Dependencies: 2.7.3
- [x] **2.7.6** Write tests for error handling in load
  - Dependencies: 2.7.5

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_llm_call_creation` | LLMCall dataclass works | 2.2.1 |
| `test_cost_calculation_claude_sonnet` | Correct cost for claude-sonnet-4 | 2.3.2 |
| `test_cost_calculation_claude_haiku` | Correct cost for claude-haiku | 2.3.2 |
| `test_unknown_model_raises` | Unknown model raises error | 2.3.4 |
| `test_record_call` | Records call with correct cost | 2.4.2 |
| `test_record_multiple_calls` | Multiple calls tracked | 2.4.2 |
| `test_get_total_cost` | Total cost sums correctly | 2.5.1 |
| `test_get_remaining_budget` | Remaining budget correct | 2.5.3 |
| `test_would_exceed_budget_true` | Detects over budget | 2.5.5 |
| `test_would_exceed_budget_false` | Allows under budget | 2.5.5 |
| `test_get_summary` | Summary has correct structure | 2.6.1 |
| `test_get_per_generation_costs` | Per-gen breakdown correct | 2.6.3 |
| `test_save_creates_file` | Save creates JSON file | 2.7.1 |
| `test_save_valid_json` | Saved file is valid JSON | 2.7.1 |
| `test_load_roundtrip` | Save then load preserves state | 2.7.3 |
| `test_load_missing_file` | Missing file handled | 2.7.5 |

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/tracking/cost_tracker.py

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class LLMCall:
    timestamp: datetime
    model: str
    role: str
    generation: int
    trial_id: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float

DEFAULT_COST_CONFIG = {
    "claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "claude-haiku": {"input": 0.00025, "output": 0.00125},
}

class CostTracker:
    def __init__(self, max_cost_usd: float, cost_config: dict = None):
        self.max_cost_usd = max_cost_usd
        self.cost_config = cost_config or DEFAULT_COST_CONFIG
        self.calls: list[LLMCall] = []

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pass

    def record_call(
        self,
        model: str,
        role: str,
        generation: int,
        input_tokens: int,
        output_tokens: int,
        trial_id: Optional[str] = None
    ) -> LLMCall:
        pass

    def get_total_cost(self) -> float:
        pass

    def get_remaining_budget(self) -> float:
        pass

    def would_exceed_budget(self, estimated_cost: float) -> bool:
        pass

    def get_summary(self) -> dict:
        pass

    def get_per_generation_costs(self) -> list[dict]:
        pass

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path, cost_config: dict = None) -> "CostTracker":
        pass
```

---

## JSON Schema (for save/load)

```json
{
  "max_cost_usd": 100.0,
  "total_cost_usd": 12.45,
  "calls": [
    {
      "timestamp": "2024-01-15T14:30:05Z",
      "model": "claude-sonnet-4",
      "role": "root",
      "generation": 1,
      "trial_id": null,
      "input_tokens": 3500,
      "output_tokens": 1200,
      "cost_usd": 0.0285
    }
  ],
  "summary": {
    "root_llm": {"calls": 15, "input_tokens": 45000, "output_tokens": 12000, "cost_usd": 4.23},
    "child_llm": {"calls": 120, "input_tokens": 180000, "output_tokens": 60000, "cost_usd": 8.22}
  },
  "per_generation": [
    {"generation": 1, "cost_usd": 2.15, "trials": 5}
  ]
}
```

---

## Acceptance Criteria

- [x] All 16 tests pass
- [x] Code coverage > 90%
- [x] Can track costs across multiple generations
- [x] Budget enforcement works correctly
- [x] Save/load round-trip preserves all data
- [x] Unknown models handled gracefully
