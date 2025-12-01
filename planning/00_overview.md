# Implementation Plan Overview

## Component Dependency Graph

```
                    ┌─────────────────┐
                    │   PufferLib     │ (external)
                    │     Tetris      │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Cost Tracker   │  │Program Evaluator│  │Experiment Tracker│
│      (2)        │  │      (1)        │  │       (3)        │
└────────┬────────┘  └────────┬────────┘  └────────┬─────────┘
         │                    │                    │
         │                    ▼                    │
         │           ┌─────────────────┐           │
         │           │   LLM Client    │           │
         │           │      (4)        │           │
         │           └────────┬────────┘           │
         │                    │                    │
         │                    ▼                    │
         │           ┌─────────────────┐           │
         └──────────▶│Child LLM Exec   │◀──────────┘
                     │      (5)        │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │Root LLM Interface│
                     │      (6)        │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │Evolution Controller│
                     │      (7)        │
                     └────────┬────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
           ┌─────────────────┐  ┌─────────────────┐
           │Integration Tests│  │    E2E Tests    │
           │      (8)        │  │      (9)        │
           └─────────────────┘  └─────────────────┘
```

## Phase Overview

### Phase 1: Core Infrastructure
| # | Component | Dependencies | Est. Tasks |
|---|-----------|--------------|------------|
| 1 | Program Evaluator | PufferLib | 12 |
| 2 | Cost Tracker | None | 8 |
| 3 | Experiment Tracker | None | 11 |

### Phase 2: LLM Integration
| # | Component | Dependencies | Est. Tasks |
|---|-----------|--------------|------------|
| 4 | LLM Client | None | 9 |
| 5 | Child LLM Executor | 1, 4 | 10 |
| 6 | Root LLM Interface | 1, 2, 3, 5 | 14 |

### Phase 3: Integration
| # | Component | Dependencies | Est. Tasks |
|---|-----------|--------------|------------|
| 7 | Evolution Controller | 1-6 | 13 |
| 8 | Integration Tests | 1-7 | 8 |
| 9 | E2E Tests | 1-8 | 7 |

## File Structure

```
planning/
├── 00_overview.md          (this file)
├── 01_program_evaluator.md
├── 02_cost_tracker.md
├── 03_experiment_tracker.md
├── 04_llm_client.md
├── 05_child_llm_executor.md
├── 06_root_llm_interface.md
├── 07_evolution_controller.md
├── 08_integration_tests.md
└── 09_e2e_tests.md
```

## Implementation Order (Critical Path)

```
Week 1: Phase 1 (Parallel)
├── [1] Program Evaluator
├── [2] Cost Tracker       } Can be done in parallel
└── [3] Experiment Tracker

Week 2: Phase 2 (Sequential)
├── [4] LLM Client
├── [5] Child LLM Executor (needs 1, 4)
└── [6] Root LLM Interface (needs 1, 2, 3, 5)

Week 3: Phase 3
├── [7] Evolution Controller (needs all above)
├── [8] Integration Tests
└── [9] E2E Tests
```

## Quick Links

- [Program Evaluator](./01_program_evaluator.md)
- [Cost Tracker](./02_cost_tracker.md)
- [Experiment Tracker](./03_experiment_tracker.md)
- [LLM Client](./04_llm_client.md)
- [Child LLM Executor](./05_child_llm_executor.md)
- [Root LLM Interface](./06_root_llm_interface.md)
- [Evolution Controller](./07_evolution_controller.md)
- [Integration Tests](./08_integration_tests.md)
- [E2E Tests](./09_e2e_tests.md)
