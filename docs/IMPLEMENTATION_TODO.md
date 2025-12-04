# Implementation TODO List

This document provides a detailed, ordered list of implementation tasks for the LLM-Evolve project. Each task includes:
- **Description**: What needs to be built
- **Dependencies**: Tasks that must be completed first
- **Files**: Source files to create/modify
- **Tests**: Test files to create
- **Acceptance Criteria**: How to verify the task is complete

**Current Target**: Circle Packing (pack 26 circles in unit square to maximize sum of radii)

---

## Phase 1: Core Infrastructure (No LLM Calls)

### Task 1.1: Project Structure Setup
**Description**: Create the complete project directory structure and boilerplate files.

**Dependencies**: None

**Files to Create**:
```
src/tetris_evolve/
├── __init__.py
├── config.py
├── cost_tracker.py
├── logger.py
├── repl.py
├── evaluator.py
├── llm_client.py
├── evolution_api.py
├── root_llm.py
├── main.py
├── exceptions.py
├── llm/
│   ├── __init__.py
│   ├── client.py
│   └── prompts.py
├── evaluation/
│   ├── __init__.py
│   └── circle_packing.py  ✅ DONE
└── utils/
    ├── __init__.py
    └── code_extraction.py
tests/
├── __init__.py
├── conftest.py
├── test_config.py
├── test_cost_tracker.py
├── test_logger.py
├── test_repl.py
├── test_evaluator.py
├── test_llm_client.py
├── test_evolution_api.py
├── test_root_llm.py
├── test_integration.py
└── test_e2e.py
configs/
└── example_config.yaml
```

**Acceptance Criteria**:
- [x] All directories exist
- [x] All `__init__.py` files exist
- [x] `uv run pytest` runs without import errors

---

### Task 1.2: Exceptions Module
**Description**: Define all custom exceptions used throughout the project.

**Dependencies**: [1.1]

**Files**:
- `src/tetris_evolve/exceptions.py`

**Tests**:
- `tests/test_exceptions.py` (simple import tests)

**Implementation**:
```python
class LLMEvolveError(Exception):
    """Base exception for all LLM-Evolve errors."""
    pass

class BudgetExceededError(LLMEvolveError):
    """Raised when the cost budget is exceeded."""
    pass

class ConfigValidationError(LLMEvolveError):
    """Raised when configuration validation fails."""
    pass

class CodeExtractionError(LLMEvolveError):
    """Raised when code cannot be extracted from LLM response."""
    pass

class EvaluationError(LLMEvolveError):
    """Raised when program evaluation fails."""
    pass
```

**Acceptance Criteria**:
- [x] All exceptions inherit from `LLMEvolveError`
- [x] Exceptions can be imported from `tetris_evolve.exceptions`

---

### Task 1.3: Configuration System
**Description**: Implement YAML configuration loading and validation with dataclasses.

**Dependencies**: [1.1, 1.2]

**Files**:
- `src/tetris_evolve/config.py`
- `configs/example_config.yaml`

**Tests**:
- `tests/test_config.py`

**Test Cases**:
1. `test_load_valid_config`: Load a valid YAML file
2. `test_load_with_defaults`: Load YAML with missing optional fields
3. `test_validation_missing_required`: Raise on missing required fields
4. `test_validation_invalid_types`: Raise on wrong types
5. `test_config_from_dict`: Create config from dictionary
6. `test_config_to_dict`: Serialize config to dictionary

**Configuration Structure**:
```yaml
experiment:
  name: "circle_packing_001"
  output_dir: "./experiments"

root_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_input_token: 0.000003
  cost_per_output_token: 0.000015
  max_iterations: 30

child_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_input_token: 0.000003
  cost_per_output_token: 0.000015

# Evaluation points to an evaluator function/class (pluggable)
evaluation:
  evaluator_fn: "tetris_evolve.evaluation.circle_packing:CirclePackingEvaluator"
  evaluator_kwargs:
    n_circles: 26
    target: 2.635
    timeout_seconds: 30

evolution:
  max_generations: 10
  max_children_per_generation: 10

budget:
  max_total_cost: 20.0
```

**Acceptance Criteria**:
- [x] All tests pass
- [x] Example config loads without errors
- [x] `Config` dataclass has typed fields for all config sections
- [x] `load_evaluator()` function loads evaluator from module path

---

### Task 1.4: Cost Tracker ✅ DONE
**Description**: Implement token usage tracking and budget enforcement.

**Dependencies**: [1.2, 1.3]

**Files**:
- `src/tetris_evolve/cost_tracker.py` ✅ Implemented

**Tests**:
- `tests/test_cost_tracker.py` ✅ Implemented

**Test Cases**:
1. `test_record_usage_basic`: Record a single usage
2. `test_cost_computation_correct`: Verify cost math is correct
3. `test_different_pricing_root_child`: Root and child use different pricing
4. `test_budget_check_within`: Returns True when within budget
5. `test_budget_check_exceeded`: Returns False when exceeded
6. `test_raise_if_over_budget`: Raises BudgetExceededError
7. `test_get_summary`: Summary has all expected fields
8. `test_remaining_budget`: Remaining budget decreases correctly

**Acceptance Criteria**:
- [x] All tests pass
- [x] Cost computation matches expected values for known inputs
- [x] Budget enforcement works correctly

---

### Task 1.5: Experiment Logger ✅ DONE
**Description**: Implement structured logging for experiments, generations, and trials.

**Dependencies**: [1.2, 1.3, 1.4]

**Files**:
- `src/tetris_evolve/logger.py` ✅ Implemented

**Tests**:
- `tests/test_logger.py` ✅ Implemented

**Test Cases**:
1. `test_create_experiment_directory`: Directory structure created correctly
2. `test_log_trial`: Trial JSON written to correct location
3. `test_log_generation`: Generation summary written
4. `test_log_root_turn`: Root LLM turn appended to JSONL
5. `test_save_experiment`: Full experiment state saved
6. `test_load_experiment`: Can reload saved experiment
7. `test_log_cost_tracking`: Cost data saved correctly

**Acceptance Criteria**:
- [x] All tests pass
- [x] Log files are valid JSON/JSONL
- [x] Experiment can be saved and reloaded

---

### Task 1.6: REPL Environment (Basic) ✅ DONE
**Description**: Implement Python code execution environment without Evolution API.

**Dependencies**: [1.1, 1.2]

**Files**:
- `src/tetris_evolve/repl.py` ✅ Implemented

**Tests**:
- `tests/test_repl.py` ✅ Implemented

**Test Cases**:
1. `test_basic_execution`: Execute simple Python code
2. `test_stdout_capture`: Print statements captured
3. `test_stderr_capture`: Errors captured
4. `test_state_persistence`: Variables persist across executions
5. `test_import_capability`: Can import standard libraries
6. `test_error_handling`: Exceptions captured gracefully
7. `test_safe_builtins`: Dangerous builtins blocked
8. `test_custom_functions`: Can define functions in REPL

**Acceptance Criteria**:
- [x] All tests pass
- [x] Code executes in isolated namespace
- [x] stdout/stderr captured correctly
- [x] State persists between calls

---

### Task 1.7: Code Extraction Utilities ✅ DONE
**Description**: Utilities for extracting REPL code blocks from Root LLM responses.

The Root LLM uses ```repl``` blocks to indicate code that should be executed.
This module extracts those blocks for execution in the REPL environment.

**Dependencies**: [1.1, 1.2]

**Files**:
- `src/tetris_evolve/utils/code_extraction.py` ✅ Implemented

**Tests**:
- `tests/test_code_extraction.py`

**Test Cases**:
1. `test_extract_repl_block`: Extract code from ```repl``` block
2. `test_multiple_repl_blocks`: Handle multiple repl code blocks
3. `test_no_code_block`: Return empty list when no repl blocks found
4. `test_ignores_other_languages`: Only extract repl, not python/js/etc
5. `test_extract_reasoning`: Extract text outside code blocks

**Acceptance Criteria**:
- [x] All tests pass
- [x] Extracts ```repl``` blocks correctly
- [x] Ignores non-repl blocks (python, javascript, etc.)

---

### Task 1.8: Circle Packing Evaluator ✅ DONE
**Description**: Implement program evaluation for circle packing.

**Dependencies**: [1.1, 1.2, 1.3]

**Files**:
- `src/tetris_evolve/evaluation/circle_packing.py` ✅ Implemented

**Tests**:
- `tests/test_evaluator.py`

**Test Cases**:
1. `test_evaluate_valid_code`: Valid packing code runs and returns metrics
2. `test_evaluate_no_function`: Detects missing construct_packing/run_packing
3. `test_evaluate_syntax_error`: Handles syntax errors
4. `test_evaluate_runtime_error`: Handles runtime errors
5. `test_validate_packing_overlap`: Detects overlapping circles
6. `test_validate_packing_bounds`: Detects circles outside unit square
7. `test_validate_packing_negative_radii`: Detects negative radii
8. `test_metrics_computed`: All expected metrics returned (sum_radii, target_ratio, valid)
9. `test_timeout_handling`: Code that takes too long is killed

**Code Specification**:
```python
import numpy as np

def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square.

    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii

    Constraints:
        - All circles must be inside [0, 1] x [0, 1]
        - No two circles may overlap
        - Radii must be non-negative
    """
    pass

def run_packing():
    """Entry point called by evaluator."""
    return construct_packing()
```

**Evaluation Metrics**:
```python
{
    "valid": bool,           # True if packing satisfies all constraints
    "sum_radii": float,      # Sum of all radii (0 if invalid)
    "target_ratio": float,   # sum_radii / 2.635 (0 if invalid)
    "combined_score": float, # target_ratio if valid, else 0
    "eval_time": float,      # Seconds to evaluate
    "error": Optional[str],  # Error message if any
}
```

**Acceptance Criteria**:
- [x] All tests pass
- [x] Metrics include sum_radii, target_ratio, valid, combined_score
- [x] Errors handled gracefully without crashing
- [x] Subprocess isolation with timeout

---

## Phase 2: LLM Integration

### Task 2.1: LLM Client ✅ DONE
**Description**: Implement Anthropic API wrapper with cost tracking integration.

**Dependencies**: [1.4]

**Files**:
- `src/tetris_evolve/llm/client.py` ✅ Implemented

**Tests**:
- `tests/test_llm_client.py` ✅ Implemented

**Test Cases**:
1. `test_generate_with_mock`: Generate with mocked API
2. `test_cost_recorded`: Usage recorded in cost tracker
3. `test_budget_check_before_call`: Budget checked before API call
4. `test_budget_exceeded_raises`: BudgetExceededError raised
5. `test_retry_on_transient_error`: Handles rate limits gracefully

**Acceptance Criteria**:
- [x] All tests pass
- [x] Integrates with Anthropic API
- [x] Token usage tracked correctly

---

### Task 2.2: Root LLM System Prompt ✅ DONE
**Description**: Define the system prompt for the Root LLM that documents available functions.

**Note**: There is NO child LLM prompt template. The Root LLM is responsible for crafting
all prompts sent to child LLMs. This gives the Root LLM full control over:
- Problem specification
- Strategy guidance
- Parent code inclusion for mutations
- Adapting prompts based on what works

**Dependencies**: [1.1]

**Files**:
- `src/tetris_evolve/llm/prompts.py` ✅ Implemented

**Tests**:
- `tests/test_prompts.py` ✅ Implemented

**Test Cases**:
1. `test_root_system_prompt_complete`: System prompt documents all API functions
2. `test_root_system_prompt_repl_usage`: Explains how to use ```repl``` blocks

**Acceptance Criteria**:
- [x] Root LLM system prompt documents all Evolution API functions
- [x] Explains REPL usage with ```repl``` blocks

---

### Task 2.3: Evolution API ✅ DONE
**Description**: Core API exposed to Root LLM for evolution control.

**Dependencies**: [1.4, 1.5, 1.7, 1.8, 2.1]

**Files**:
- `src/tetris_evolve/evolution_api.py` ✅ Implemented

**Tests**:
- `tests/test_evolution_api.py` ✅ Implemented

**Test Cases**:
1. `test_spawn_child_llm`: Spawns child and returns result
2. `test_spawn_with_parent`: Parent code included in prompt
3. `test_spawn_handles_extraction_failure`: Graceful failure on bad response
4. `test_evaluate_program`: Direct evaluation works
5. `test_advance_generation`: Generation counter increments
6. `test_terminate_evolution`: Returns final results
7. `test_get_generation_history`: Returns all generations
8. `test_get_best_trials`: Returns sorted trials by sum_radii
9. `test_trial_logging`: Trials logged correctly

**Acceptance Criteria**:
- [x] All tests pass
- [x] All API functions work as documented
- [x] State tracked correctly across calls

---

### Task 2.4: REPL + Evolution API Integration ✅ DONE
**Description**: Inject Evolution API functions into REPL environment.

**Dependencies**: [1.6, 2.3]

**Files**:
- `src/tetris_evolve/repl.py` ✅ Modified (supports api_functions injection)
- `src/tetris_evolve/evolution_api.py` ✅ Provides get_api_functions()

**Tests**:
- `tests/test_evolution_api.py` ✅ Tests API integration

**Test Cases**:
1. `test_spawn_from_repl`: spawn_child_llm callable from REPL
2. `test_evaluate_from_repl`: evaluate_program callable
3. `test_advance_from_repl`: advance_generation callable
4. `test_terminate_from_repl`: terminate_evolution callable
5. `test_get_history_from_repl`: get_generation_history callable
6. `test_complex_workflow`: Multi-step workflow works

**Acceptance Criteria**:
- [x] All tests pass
- [x] All Evolution API functions available in REPL
- [x] State shared correctly

---

## Phase 3: Root LLM Orchestrator

### Task 3.1: Root LLM Orchestrator ✅ DONE
**Description**: Main orchestrator that runs the Root LLM evolution loop.

**Dependencies**: [1.5, 2.1, 2.2, 2.4]

**Files**:
- `src/tetris_evolve/root_llm.py` ✅ Implemented

**Tests**:
- `tests/test_root_llm.py` ✅ Implemented

**Test Cases**:
1. `test_initialization`: All components initialized correctly
2. `test_build_initial_messages`: System prompt constructed
3. `test_extract_code_blocks`: REPL blocks extracted
4. `test_execute_code_in_repl`: Code executed and result added
5. `test_termination_detection`: Detects terminate_evolution call
6. `test_max_iterations_stop`: Stops at max iterations
7. `test_budget_exceeded_stop`: Stops on budget exceeded
8. `test_conversation_history`: History maintained correctly

**Acceptance Criteria**:
- [x] All tests pass
- [x] Full orchestration loop works with mock LLM

---

### Task 3.2: Main Entry Point ✅ DONE
**Description**: CLI entry point for running experiments.

**Dependencies**: [1.3, 3.1]

**Files**:
- `src/tetris_evolve/main.py` ✅ Implemented
- `src/tetris_evolve/__main__.py` ✅ Implemented (module entry point)

**Tests**:
- `tests/test_main.py` ✅ Implemented

**Test Cases**:
1. `test_cli_help`: --help works
2. `test_cli_config_required`: Requires config file
3. `test_cli_loads_config`: Config loaded correctly
4. `test_cli_creates_experiment_dir`: Experiment directory created
5. `test_cli_runs_orchestrator`: Orchestrator starts

**Acceptance Criteria**:
- [x] All tests pass
- [x] Can run: `uv run python -m tetris_evolve --config configs/example.yaml`

---

## Phase 4: Polish and E2E

### Task 4.1: Integration Tests
**Description**: Tests that verify multiple components work together.

**Dependencies**: [2.4, 3.1]

**Files**:
- `tests/test_integration.py`

**Test Cases**:
1. `test_spawn_evaluate_cycle`: Full spawn -> evaluate cycle
2. `test_multi_generation_flow`: Multiple generations work
3. `test_budget_stops_evolution`: Budget cutoff works
4. `test_logging_complete`: All log files created
5. `test_resume_experiment`: Can resume from saved state

**Acceptance Criteria**:
- [x] All tests pass with mock LLMs
- [x] Full pipeline works end-to-end

---

### Task 4.2: E2E Tests (Real LLM)
**Description**: Minimal tests with actual LLM API calls.

**Dependencies**: [4.1]

**Files**:
- `tests/test_e2e.py`

**Test Cases**:
1. `test_single_trial_real_llm`: One trial with real API
2. `test_budget_limited_run`: Short run with $0.50 budget

**Note**: These tests require API key and are skipped in CI by default.

**Acceptance Criteria**:
- [x] Tests pass when run manually with API key
- [x] Tests skip gracefully when no API key

---

### Task 4.3: Example Configuration
**Description**: Create a ready-to-use example configuration.

**Dependencies**: [1.3, 2.2]

**Files**:
- `configs/example_config.yaml`

**Configuration**:
```yaml
experiment:
  name: "circle_packing_v1"
  output_dir: "./experiments"

root_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_input_token: 0.000003
  cost_per_output_token: 0.000015
  max_iterations: 30

child_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_input_token: 0.000003
  cost_per_output_token: 0.000015

evolution:
  max_generations: 10
  max_children_per_generation: 10

budget:
  max_total_cost: 20.0

evaluation:
  n_circles: 26
  target_sum: 2.635
  timeout_seconds: 30
```

**Acceptance Criteria**:
- [x] Config loads without errors
- [x] All required fields present
- [x] Reasonable defaults for testing

---

### Task 4.4: Documentation
**Description**: README and usage documentation.

**Dependencies**: [4.3]

**Files**:
- `README.md`

**Contents**:
1. Project overview
2. Installation instructions
3. Quick start guide
4. Configuration reference
5. API reference (brief)
6. Development guide

**Acceptance Criteria**:
- [x] README explains how to install
- [x] README explains how to run
- [x] Configuration options documented

---

## Dependency Graph

```
Phase 1 (Core):
1.1 Project Structure
 │
 ├─► 1.2 Exceptions
 │    │
 │    ├─► 1.3 Config ─────────────────┬─► 1.4 Cost Tracker
 │    │                               │    │
 │    ├─► 1.6 REPL (Basic)            │    └─► 1.5 Logger
 │    │                               │
 │    ├─► 1.7 Code Extraction         │
 │    │                               │
 │    └─► 1.8 Evaluator ◄─────────────┘  ✅ DONE (CirclePackingEvaluator)

Phase 2 (LLM):
1.4 ─► 2.1 LLM Client
        │
1.1 ─► 2.2 Prompts
        │
        ├─► 2.3 Evolution API ◄── 1.4, 1.5, 1.7, 1.8
        │    │
1.6 ────┴─► 2.4 REPL + API Integration

Phase 3 (Orchestrator):
2.4 ─► 3.1 Root Orchestrator ◄── 1.5, 2.1, 2.2
        │
        └─► 3.2 Main Entry Point ◄── 1.3

Phase 4 (Polish):
3.1 ─► 4.1 Integration Tests
        │
        └─► 4.2 E2E Tests

1.3, 2.2 ─► 4.3 Example Config
             │
             └─► 4.4 Documentation
```

---

## Current Progress

| Task | Status | Notes |
|------|--------|-------|
| 1.1 Project Structure | ✅ DONE | All directories and __init__.py files |
| 1.2 Exceptions | ✅ DONE | `src/tetris_evolve/exceptions.py` |
| 1.3 Configuration | ✅ DONE | `src/tetris_evolve/config.py` - with pluggable evaluator |
| 1.4 Cost Tracker | ✅ DONE | `src/tetris_evolve/cost_tracker.py` |
| 1.5 Logger | ✅ DONE | `src/tetris_evolve/logger.py` |
| 1.6 REPL | ✅ DONE | `src/tetris_evolve/repl.py` |
| 1.7 Code Extraction | ✅ DONE | `src/tetris_evolve/utils/code_extraction.py` - repl blocks only |
| 1.8 CirclePackingEvaluator | ✅ DONE | `src/tetris_evolve/evaluation/circle_packing.py` |
| 2.1 LLM Client | ✅ DONE | `src/tetris_evolve/llm/client.py` |
| 2.2 Root LLM System Prompt | ✅ DONE | `src/tetris_evolve/llm/prompts.py` |
| 2.3 Evolution API | ✅ DONE | `src/tetris_evolve/evolution_api.py` |
| 2.4 REPL + Evolution API | ✅ DONE | API injection via `get_api_functions()` |
| 3.1 Root LLM Orchestrator | ✅ DONE | `src/tetris_evolve/root_llm.py` |
| 3.2 Main Entry Point | ✅ DONE | `src/tetris_evolve/main.py` + `__main__.py` |
| 4.1 Integration Tests | ✅ DONE | `tests/test_integration.py` |
| 4.2 E2E Tests | ✅ DONE | `tests/test_e2e.py` (skipped without API key) |
| 4.3 Example Configuration | ✅ DONE | `configs/example_config.yaml` |
| 4.4 Documentation | ✅ DONE | `README.md` - full documentation |
| PoC: REPL | ✅ Validated | `experiments/poc_repl.py` |
| PoC: Cost Tracker | ✅ Validated | `experiments/poc_cost_tracker.py` |
| PoC: Integration | ✅ Validated | `experiments/poc_circle_packing_integration.py` |

**Phase 1 Complete**: 89 tests passing

**Phase 2 Complete**: LLM integration implemented

**Phase 3 Complete**: Root LLM Orchestrator and CLI implemented (177 tests passing)

**Phase 4 Complete**: Polish and E2E implemented (195 tests passing, 3 skipped)

**PoC Results**: Hexagonal packing achieved 2.08 sum (79% of 2.635 benchmark)

---

## Estimated Effort

| Phase | Tasks | Estimated Tests | Complexity |
|-------|-------|-----------------|------------|
| 1 (Core) | 8 | ~50 | Medium |
| 2 (LLM) | 4 | ~25 | Medium |
| 3 (Orchestrator) | 2 | ~15 | High |
| 4 (Polish) | 4 | ~10 | Low |
| **Total** | **18** | **~100** | |

---

## Notes for Implementation

### Circle Packing Specifics
- **Target benchmark**: AlphaEvolve achieved sum = 2.635 for n=26
- **Evaluation**: Deterministic, fast (~60ms), geometric validation
- **Proven strategies**: Hexagonal (2.08), Grid (1.77), Corner-first (1.65)
- **Common failures**: Greedy approaches can fail validation due to numerical issues

### Testing Strategy
- Use pytest fixtures for common setup
- Mock LLM responses for unit/integration tests
- E2E tests optional, require API key
- CirclePackingEvaluator uses subprocess isolation for safety

### Cost Considerations
- Default budget should be conservative ($5-10 for testing)
- E2E tests should use minimal budget ($0.50)
- Log all costs for analysis

### Error Handling Philosophy
- Never crash the whole experiment on a single trial failure
- Log errors, continue with other trials
- Budget exceeded is the only hard stop

### Observability Priorities
1. Every LLM call logged with full prompt/response
2. Every trial logged with code and metrics
3. Cost tracked in real-time
4. Easy to analyze experiments post-hoc
