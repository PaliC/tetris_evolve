# LLM-Driven Generation Selection Design

## Overview

Replace hardcoded trial selection in `_advance_generation()` with LLM-driven selection. The Root LLM will analyze all trial results and decide which trials to carry forward, providing reasoning for each selection.

## Current Behavior

In `evolution_api.py:_advance_generation()`:
```python
# Auto-select top 3 (or fewer) trials as parents
best_trials = self._get_best_trials(n=min(3, len(gen.trials)))
selected_trial_ids = [t["trial_id"] for t in best_trials]
reasoning = "Auto-selected top performing trials"
```

**Problems:**
1. Selection purely based on `sum_radii` score
2. No consideration of diversity or potential
3. Generic hardcoded reasoning string
4. Root LLM has no say in which approaches to pursue

## Proposed Design

### High-Level Flow

```
1. Root LLM spawns children for generation N
2. System builds feedback with all trial results
3. Feedback asks Root LLM to select promising trials with reasoning
4. Root LLM responds with selections (JSON format)
5. System parses selections and records per-trial reasoning
6. _advance_generation() stores the LLM's selections
7. Generation advances with recorded data
```

### Data Model Changes

**TrialSelection dataclass** (new):
```python
@dataclass
class TrialSelection:
    trial_id: str
    reasoning: str  # Why this trial was selected
    category: str   # "performance" | "diversity" | "potential"
```

**GenerationSummary updates**:
```python
@dataclass
class GenerationSummary:
    # ... existing fields ...
    selected_trial_ids: list[str]
    selection_reasoning: str  # Keep for backwards compat (overall summary)
    trial_selections: list[TrialSelection]  # NEW: per-trial reasoning
```

### API Changes

**_advance_generation() signature**:
```python
def _advance_generation(
    self,
    selections: list[dict[str, str]] | None = None,
) -> int:
    """
    Advance to the next generation.

    Args:
        selections: Optional list of {trial_id, reasoning, category}.
                   If None, falls back to auto-selection (backwards compat).
    """
```

### Root LLM Prompt Changes

Add to system prompt:
```
## Trial Selection

After spawning children, you must select which trials to carry forward to the next generation.

Respond with a ```selection``` block containing JSON:
```selection
{
  "selections": [
    {"trial_id": "trial_0_2", "reasoning": "Highest score of 2.45", "category": "performance"},
    {"trial_id": "trial_0_5", "reasoning": "Novel hexagonal approach worth exploring", "category": "diversity"},
    {"trial_id": "trial_0_1", "reasoning": "Optimization-based approach has room for improvement", "category": "potential"}
  ],
  "summary": "Selected top performer plus two diverse approaches for exploration"
}
```

**Selection Categories:**
- `performance`: Strong metrics, proven results
- `diversity`: Different algorithmic approach worth exploring
- `potential`: Shows promise even if current score is lower

Select 2-5 trials. Balance performance with diversity.
```

### Orchestrator Changes

In `root_llm.py`, after executing REPL blocks:

```python
# After children spawned, request selection from Root LLM
if self.evolution_api.has_children_in_current_generation():
    # Build selection request with full trial data
    selection_request = self._build_selection_request_message()
    self.messages.append({"role": "user", "content": selection_request})

    # Get selection response
    selection_response = self.root_llm.generate(
        messages=self.messages,
        system=system_prompt,
        max_tokens=2048,
        temperature=0.5,  # Lower temp for more deterministic selection
    )

    # Parse selections from response
    selections = self._parse_selection_response(selection_response.content)

    # Advance with LLM selections
    self.evolution_api._advance_generation(selections=selections)
```

### Selection Request Message Format

```python
def _build_selection_request_message(self) -> str:
    """Build message asking Root LLM to select trials."""
    gen = self.evolution_api.current_generation
    trials = self.evolution_api.generations[gen].trials

    lines = [
        f"Generation {gen} spawning complete. {len(trials)} trials evaluated.",
        "",
        "## Trial Results",
        "",
    ]

    # Sort by score but show all
    sorted_trials = sorted(
        trials,
        key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
        reverse=True,
    )

    for i, trial in enumerate(sorted_trials, 1):
        score = trial.metrics.get("sum_radii", 0) if trial.success else 0
        valid = "valid" if trial.success else "INVALID"
        lines.append(f"{i}. **{trial.trial_id}** [{valid}]")
        lines.append(f"   Score: {score:.4f}")
        lines.append(f"   Reasoning: {trial.reasoning[:200]}...")
        if not trial.success:
            lines.append(f"   Error: {trial.error}")
        lines.append("")

    lines.extend([
        "## Your Task",
        "",
        "Select 2-5 trials to carry forward. Consider:",
        "- **Performance**: Which trials have the best scores?",
        "- **Diversity**: Which trials use different approaches worth exploring?",
        "- **Potential**: Which trials might improve with refinement?",
        "",
        "Respond with a ```selection``` block containing JSON.",
    ])

    return "\n".join(lines)
```

### Parsing Selection Response

```python
def _parse_selection_response(self, response: str) -> list[dict[str, str]] | None:
    """Parse selection JSON from LLM response."""
    # Extract ```selection``` block
    pattern = r"```selection\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return None  # Fall back to auto-selection

    try:
        data = json.loads(match.group(1))
        return data.get("selections", [])
    except json.JSONDecodeError:
        return None  # Fall back to auto-selection
```

## Backwards Compatibility

1. `_advance_generation(selections=None)` falls back to auto-selection
2. Tests using mock LLM clients continue to work
3. `selection_reasoning` field remains for overall summary
4. Logging format extends but doesn't break existing structure

## Test Plan

1. **Unit tests for parsing**: Test `_parse_selection_response()` with various inputs
2. **Unit tests for data model**: Test `TrialSelection` serialization
3. **Integration test**: Mock LLM returns selection, verify stored correctly
4. **Fallback test**: Invalid/missing selection falls back to auto-selection
5. **End-to-end test**: Full evolution run with LLM selection enabled

## Implementation Steps

1. Add `TrialSelection` dataclass to `evolution_api.py`
2. Update `GenerationSummary` with `trial_selections` field
3. Modify `_advance_generation()` to accept selections parameter
4. Add `_build_selection_request_message()` to orchestrator
5. Add `_parse_selection_response()` to orchestrator
6. Update orchestrator loop to request and use selections
7. Update system prompt with selection instructions
8. Add extraction helper for ````selection``` blocks
9. Write tests
10. Verify logging captures new data

## Considerations

- **Cost**: Extra LLM call per generation for selection (~500 tokens)
- **Latency**: ~1-2s additional per generation
- **Reliability**: Fallback to auto-selection if parsing fails
- **Transparency**: All reasoning recorded and logged
