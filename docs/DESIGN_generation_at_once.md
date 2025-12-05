# Design: Generation-at-Once Root LLM Architecture

## Problem Statement

The current architecture uses an iterative feedback loop where:
1. Root LLM is called repeatedly (up to 30 iterations)
2. Root LLM spawns children incrementally based on feedback
3. Root LLM explicitly calls `advance_generation()` when ready

This approach is complex and puts generation management responsibility on the Root LLM.

## Proposed Solution

Simplify to a **generation-driven** architecture where:
1. Root LLM is called **once per generation**
2. Root LLM outputs prompts for **all children in one call** (up to max_children_per_generation)
3. All children are spawned in parallel automatically
4. Generation advances **automatically** after spawning
5. Results from previous generation are fed back for next generation

## Key Changes

### 1. Remove `advance_generation()` from API

- Generation advancement becomes automatic after spawning children
- Removes complexity from Root LLM decision-making
- API functions reduced from 5 to 4:
  - `spawn_children_parallel()` - remains (primary)
  - `spawn_child_llm()` - remains (legacy)
  - `evaluate_program()` - remains
  - `terminate_evolution()` - remains

### 2. Modify Root LLM System Prompt

- Inform Root LLM of max_children_per_generation
- Root LLM should output prompts for entire generation
- Remove documentation for `advance_generation()`
- Update guidelines to reflect generation-at-once behavior

### 3. Change Orchestration Loop

**Current Flow:**
```
for iteration in range(max_iterations):
    response = root_llm.generate()
    execute_repl_blocks()  # May call spawn, advance_generation, terminate
    if terminated: break
```

**New Flow:**
```
for generation in range(max_generations):
    response = root_llm.generate()  # Output prompts for all children
    execute_repl_blocks()           # Spawns children
    auto_advance_generation()       # Automatic
    feed_results_to_root_llm()
    if terminated: break
```

### 4. Automatic Generation Advancement

After `spawn_children_parallel()` or any spawning within a generation:
- If children were spawned in current generation, advance automatically
- No explicit call needed from Root LLM

## API Surface After Change

```python
# Available to Root LLM via REPL:
spawn_children_parallel(children: list[dict]) -> list[dict]  # Primary
spawn_child_llm(prompt: str, parent_id: str = None) -> dict  # Legacy
evaluate_program(code: str) -> dict
terminate_evolution(reason: str, best_program: str = None) -> dict
```

## Expected Root LLM Output Format

Root LLM should output a single REPL block per generation:

```repl
children = [
    {"prompt": "Strategy 1: hexagonal packing..."},
    {"prompt": "Strategy 2: greedy approach..."},
    {"prompt": "Strategy 3: optimization-based..."},
    # ... up to max_children_per_generation prompts
]
results = spawn_children_parallel(children)

# Analyze results for next generation planning
for r in results:
    print(f"{r['trial_id']}: score={r['metrics'].get('sum_radii', 0):.4f}")
```

## Configuration

No new configuration needed. Existing config params:
- `evolution.max_generations`: Maximum generations (unchanged)
- `evolution.max_children_per_generation`: Children per generation (unchanged)

The Root LLM system prompt will be dynamically populated with these values.

## Backward Compatibility

- `spawn_child_llm()` remains available for legacy use
- Tests will be updated to reflect new architecture
- Existing experiments are not affected (config format unchanged)

## Benefits

1. **Simpler mental model**: One Root LLM call = one generation
2. **Reduced iterations**: No iterative feedback within generations
3. **Clearer responsibility**: Root LLM focuses on prompt crafting, not flow control
4. **Predictable execution**: N generations = N Root LLM calls (plus termination)
