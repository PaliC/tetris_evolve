# LLM-Driven Tetris Evolution

An experimental system that evolves Tetris-playing programs by combining:
- **AlphaEvolve** - Evolutionary code generation with LLMs
- **Recursive LLMs** - Hierarchical LLM decision-making

## Key Idea

A **Root LLM** has complete autonomy over the evolution process:
- Decides which programs survive to the next generation
- Crafts custom prompts for **Child LLMs** to generate/mutate code
- Decides when to terminate (convergence, good enough, etc.)

All within enforced **hard limits** (max generations, max cost, max time).

## Architecture

```
Evolution Controller (enforces hard limits)
         │
         ▼
    Root LLM (complete freedom within limits)
         │
    ┌────┼────┐
    ▼    ▼    ▼
 Child Child Child  (generate/mutate code)
    │    │    │
    ▼    ▼    ▼
   Tetris Evaluator (run games, measure performance)
```

## Safeguards

1. **Max Generations** - Hard cap on evolution iterations
2. **Cost Tracking** - Per-call token costs, budget enforcement
3. **Max Time** - Wall-clock time limit
4. **Reasoning Recording** - Every trial and generation records rationale

## Data Organization

```
experiments/
└── exp_{timestamp}/
    ├── config.yaml
    ├── cost_tracker.json
    └── generations/
        └── gen_{N}/
            ├── root_llm_reasoning.md
            └── trials/
                └── trial_{M}/
                    ├── code.py
                    ├── metrics.json
                    └── reasoning.md
```

## Development Approach

**Test-Driven Development** in 10 chunks:

| Phase | Components |
|-------|------------|
| 1. Core | Tetris Env, Evaluator, Cost Tracker, Experiment Tracker |
| 2. LLM | LLM Client, Child Executor, Root Interface |
| 3. Integration | Evolution Controller, Integration Tests, E2E Tests |

## Getting Started

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Run evolution
python -m tetris_evolve --config config.yaml
```

## Documentation

See [DESIGN.md](DESIGN.md) for the complete design document including:
- Detailed architecture
- All component interfaces
- Test specifications
- Configuration schema
- LLM prompts

## References

- [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) - Evolutionary coding with LLMs
- [Recursive LLMs](https://alexzhang13.github.io/blog/2025/rlm/) - Hierarchical LLM decision-making
