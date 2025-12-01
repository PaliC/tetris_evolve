# Component 3: Experiment Tracker

## Overview
Manages the experiment directory structure and persists all trial/generation data to disk.

**File**: `src/tetris_evolve/tracking/experiment_tracker.py`
**Test File**: `tests/test_experiment_tracker.py`
**Dependencies**: None (standalone)

---

## Checklist

### 3.1 Project Setup
- [ ] **3.1.1** Create `experiment_tracker.py` in `src/tetris_evolve/tracking/`
  - Dependencies: None (folder created in 2.1.1)
- [ ] **3.1.2** Create `tests/test_experiment_tracker.py` skeleton
  - Dependencies: 3.1.1

### 3.2 Data Classes
- [ ] **3.2.1** Define `TrialData` dataclass
  - Dependencies: 3.1.1
  ```python
  @dataclass
  class TrialData:
      trial_id: str
      generation: int
      parent_id: Optional[str]
      code: str
      prompt: str
      reasoning: str
      llm_response: str
      metrics: Optional[dict]  # EvaluationResult as dict
      timestamp: datetime
  ```
- [ ] **3.2.2** Define `GenerationStats` dataclass
  - Dependencies: 3.1.1
  ```python
  @dataclass
  class GenerationStats:
      generation: int
      num_trials: int
      best_score: float
      avg_score: float
      best_trial_id: str
      selected_parent_ids: list[str]
  ```
- [ ] **3.2.3** Write tests for dataclass creation
  - Dependencies: 3.2.1, 3.2.2

### 3.3 Experiment Initialization
- [ ] **3.3.1** Implement `ExperimentTracker.__init__(base_dir: Path)`
  - Dependencies: 3.1.1
- [ ] **3.3.2** Implement `create_experiment(config: dict) -> str`
  - Dependencies: 3.3.1
  - Creates timestamped experiment directory
  - Saves frozen config.yaml
  - Returns experiment_id
- [ ] **3.3.3** Write tests for `create_experiment`
  - Dependencies: 3.3.2
  - Check: directory created
  - Check: config.yaml exists and valid
  - Check: experiment_id format correct

### 3.4 Generation Management
- [ ] **3.4.1** Implement `start_generation(generation: int) -> Path`
  - Dependencies: 3.3.2
  - Creates `generations/gen_{N:03d}/` directory
  - Creates `trials/` subdirectory
  - Returns generation path
- [ ] **3.4.2** Write tests for `start_generation`
  - Dependencies: 3.4.1
- [ ] **3.4.3** Implement `complete_generation(generation, selected_ids, root_reasoning, stats)`
  - Dependencies: 3.4.1, 3.2.2
  - Saves `generation_stats.json`
  - Saves `root_llm_reasoning.md`
  - Saves `selected_parents.json`
- [ ] **3.4.4** Write tests for `complete_generation`
  - Dependencies: 3.4.3

### 3.5 Trial Management
- [ ] **3.5.1** Implement `_generate_trial_id() -> str`
  - Dependencies: 3.1.1
  - Format: `trial_{N:03d}` (incrementing)
- [ ] **3.5.2** Implement `save_trial(trial: TrialData) -> Path`
  - Dependencies: 3.2.1, 3.4.1, 3.5.1
  - Creates `trials/trial_{id}/` directory
  - Saves: `code.py`, `prompt.txt`, `reasoning.md`, `llm_response.txt`
  - Saves: `metrics.json` (if present), `parent_id.txt` (if present)
  - Returns trial directory path
- [ ] **3.5.3** Write tests for `save_trial`
  - Dependencies: 3.5.2
  - Check: all files created
  - Check: content matches input
- [ ] **3.5.4** Implement `get_trial(trial_id: str) -> TrialData`
  - Dependencies: 3.5.2
  - Load trial from disk
- [ ] **3.5.5** Write tests for `get_trial`
  - Dependencies: 3.5.4
  - Check: round-trip save/get preserves data

### 3.6 Queries
- [ ] **3.6.1** Implement `get_generation_trials(generation: int) -> list[TrialData]`
  - Dependencies: 3.5.4
  - Load all trials for a generation
- [ ] **3.6.2** Write tests for `get_generation_trials`
  - Dependencies: 3.6.1
- [ ] **3.6.3** Implement `get_all_trials() -> list[TrialData]`
  - Dependencies: 3.6.1
  - Load all trials across all generations
- [ ] **3.6.4** Write tests for `get_all_trials`
  - Dependencies: 3.6.3
- [ ] **3.6.5** Implement `get_best_trial() -> TrialData`
  - Dependencies: 3.6.3
  - Return trial with highest score
- [ ] **3.6.6** Write tests for `get_best_trial`
  - Dependencies: 3.6.5

### 3.7 Experiment Stats
- [ ] **3.7.1** Implement `update_experiment_stats(stats: dict) -> None`
  - Dependencies: 3.3.2
  - Updates `experiment_stats.json` at experiment root
- [ ] **3.7.2** Write tests for `update_experiment_stats`
  - Dependencies: 3.7.1
- [ ] **3.7.3** Implement `get_experiment_stats() -> dict`
  - Dependencies: 3.7.1
  - Load experiment-level stats
- [ ] **3.7.4** Write tests for `get_experiment_stats`
  - Dependencies: 3.7.3

### 3.8 Resume Support
- [ ] **3.8.1** Implement `load_experiment(experiment_dir: Path) -> ExperimentTracker`
  - Dependencies: 3.3.2, 3.6.3
  - Classmethod to resume from existing experiment
  - Reconstructs state from disk
- [ ] **3.8.2** Write tests for `load_experiment`
  - Dependencies: 3.8.1
  - Check: can resume mid-experiment

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_trial_data_creation` | TrialData dataclass works | 3.2.1 |
| `test_generation_stats_creation` | GenerationStats dataclass works | 3.2.2 |
| `test_create_experiment` | Creates directory with config | 3.3.2 |
| `test_experiment_id_format` | ID is timestamped correctly | 3.3.2 |
| `test_start_generation` | Creates gen directory structure | 3.4.1 |
| `test_complete_generation` | Saves all gen files | 3.4.3 |
| `test_save_trial_creates_files` | All trial files created | 3.5.2 |
| `test_save_trial_content` | File contents match input | 3.5.2 |
| `test_get_trial_roundtrip` | Save/get preserves data | 3.5.4 |
| `test_get_generation_trials` | Returns all gen trials | 3.6.1 |
| `test_get_all_trials` | Returns all trials | 3.6.3 |
| `test_get_best_trial` | Returns highest scorer | 3.6.5 |
| `test_update_experiment_stats` | Stats file updated | 3.7.1 |
| `test_get_experiment_stats` | Stats loaded correctly | 3.7.3 |
| `test_load_experiment` | Can resume experiment | 3.8.1 |

---

## Directory Structure Created

```
experiments/
└── exp_2024_01_15_143022/
    ├── config.yaml
    ├── experiment_stats.json
    │
    └── generations/
        ├── gen_001/
        │   ├── generation_stats.json
        │   ├── root_llm_reasoning.md
        │   ├── selected_parents.json
        │   └── trials/
        │       ├── trial_001/
        │       │   ├── code.py
        │       │   ├── prompt.txt
        │       │   ├── reasoning.md
        │       │   ├── llm_response.txt
        │       │   └── metrics.json
        │       └── trial_002/
        │           └── ...
        └── gen_002/
            └── ...
```

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/tracking/experiment_tracker.py

import json
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class TrialData:
    trial_id: str
    generation: int
    parent_id: Optional[str]
    code: str
    prompt: str
    reasoning: str
    llm_response: str
    metrics: Optional[dict]
    timestamp: datetime

@dataclass
class GenerationStats:
    generation: int
    num_trials: int
    best_score: float
    avg_score: float
    best_trial_id: str
    selected_parent_ids: list[str]

class ExperimentTracker:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.experiment_dir: Optional[Path] = None
        self.experiment_id: Optional[str] = None
        self.current_generation: int = 0
        self._trial_counter: int = 0

    def create_experiment(self, config: dict) -> str:
        pass

    def start_generation(self, generation: int) -> Path:
        pass

    def complete_generation(
        self,
        generation: int,
        selected_ids: list[str],
        root_reasoning: str,
        stats: GenerationStats
    ) -> None:
        pass

    def save_trial(self, trial: TrialData) -> Path:
        pass

    def get_trial(self, trial_id: str) -> TrialData:
        pass

    def get_generation_trials(self, generation: int) -> list[TrialData]:
        pass

    def get_all_trials(self) -> list[TrialData]:
        pass

    def get_best_trial(self) -> TrialData:
        pass

    def update_experiment_stats(self, stats: dict) -> None:
        pass

    def get_experiment_stats(self) -> dict:
        pass

    @classmethod
    def load_experiment(cls, experiment_dir: Path) -> "ExperimentTracker":
        pass
```

---

## Acceptance Criteria

- [ ] All 15 tests pass
- [ ] Code coverage > 90%
- [ ] Directory structure matches spec
- [ ] All files are valid (YAML, JSON, Markdown, Python)
- [ ] Can resume experiment from disk
- [ ] Trial IDs are unique and incrementing
