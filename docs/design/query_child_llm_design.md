# Design: Query Child LLM Without Side Effects

## Overview

This document proposes a new API function `query_child_llm()` that allows the Root LLM to make auxiliary queries to child LLMs without creating trials, affecting generation state, or running the evaluator.

## Problem Statement

Currently, all child LLM calls go through `spawn_child_llm()` or `spawn_children_parallel()`, which:
1. Create trial IDs and record trials to the database
2. Increment trial counts for the current generation
3. Run the evaluator on the response
4. Extract code from the response (assuming code generation)

However, the Root LLM sometimes needs to use child LLMs for non-code-generation purposes:
- **Analysis**: Ask a child LLM to analyze why certain trials failed
- **Prompt crafting**: Get help writing better mutation prompts
- **Brainstorming**: Generate ideas for new approaches
- **Code review**: Get feedback on a specific trial's code
- **Summarization**: Summarize patterns across trials

## Use Cases

| Use Case | Example Prompt | Expected Output |
|----------|---------------|-----------------|
| Failure analysis | "Analyze these 3 failed trials and identify common patterns..." | Text explaining patterns |
| Prompt improvement | "Improve this mutation prompt to encourage more diverse solutions..." | Better prompt text |
| Brainstorming | "What optimization techniques could help improve circle packing?" | List of ideas |
| Code review | "Review this trial's code and suggest improvements..." | Code review feedback |
| Summarization | "Summarize the evolution strategy so far..." | Strategy summary |

## Proposed API Design

### Option A: Simple Query Function (Recommended)

```python
def query_child_llm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """
    Query a child LLM for auxiliary tasks without creating a trial.

    Use this for analysis, brainstorming, prompt crafting, and other
    non-code-generation tasks. Queries do NOT:
    - Create trials or affect trial counts
    - Run the evaluator
    - Affect generation state

    Queries DO:
    - Count against the cost budget
    - Get logged for observability

    Args:
        prompt: The prompt to send to the child LLM
        model: Alias of the child LLM to use (defaults to default_child_llm_alias)
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum tokens in response (default 4096)

    Returns:
        Dictionary with:
        {
            'response': str,        # Full response text
            'model_alias': str,     # Which model was used
            'input_tokens': int,    # Tokens used
            'output_tokens': int,   # Tokens generated
            'cost': float,          # Cost of this query
        }
    """
```

### Option B: Purpose Parameter

```python
def spawn_child_llm(
    prompt: str,
    parent_id: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    purpose: Literal["trial", "query"] = "trial",  # NEW
) -> dict[str, Any]:
```

**Recommendation**: Option A is cleaner because:
1. Different return types (queries don't have trial_id, metrics, code, etc.)
2. Clearer separation of concerns
3. Easier to document and understand
4. No risk of accidentally creating trials

## Cost Tracking

### Approach: Unified Budget with Labeled Tracking

Queries share the same budget as trials but are tracked separately:

```python
# In CostTracker
llm_type: str  # "root", "child:alias", or "query:alias"  <- NEW

# In CostSummary
query_costs: dict[str, float]   # Per-model query costs
query_calls: dict[str, int]     # Per-model query counts
total_query_cost: float
total_query_calls: int
```

**Rationale**:
- Same budget prevents unbounded query costs
- Separate tracking enables analysis of query overhead
- Easy to add query-specific limits later if needed

## Logging

### Approach: New JSONL File

Create `child_llm_queries.jsonl` alongside `root_llm_log.jsonl`:

```json
{
  "query_id": "query_gen2_0",
  "generation": 2,
  "model_alias": "fast",
  "prompt": "Analyze these failures...",
  "response": "I notice that...",
  "input_tokens": 1234,
  "output_tokens": 567,
  "cost": 0.0023,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Rationale**:
- Separate file keeps trial logs clean
- Easy to analyze query patterns
- Doesn't bloat root LLM conversation log

### Logging Method Addition

```python
class ExperimentLogger:
    def log_query(
        self,
        query_id: str,
        generation: int,
        model_alias: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """Log a child LLM query."""
```

## Observability

### Terminal Output

```
  └─ Query (fast): analyzing trial failures...
     → Response: 234 tokens, $0.0023
```

### Experiment Summary

Add to final summary:
```
  Queries: 12 total ($0.0345)
    - fast: 8 queries
    - strong: 4 queries
```

## Rate Limiting

### Approach: Rely on Cost Budget

No separate rate limiting needed because:
1. Cost budget naturally limits total queries
2. Root LLM is incentivized to be efficient
3. Adding query limits adds complexity without clear benefit

If abuse becomes an issue, we can add:
- `max_queries_per_generation: int` config option
- Per-generation query counter in `EvolutionAPI`

## Configuration Changes

No new config options required. Queries use existing:
- `child_llms` configuration for available models
- `default_child_llm_alias` for default model
- `budget.max_total_cost` for cost limits

## Implementation Plan

### Phase 1: Core Implementation

1. **Add `query_child_llm()` to `EvolutionAPI`** (`evolution_api.py`)
   - Resolve model alias
   - Check budget
   - Make LLM call (reuse `_get_or_create_child_llm_client`)
   - Track costs with `query:alias` type
   - Return response dict

2. **Update `CostTracker`** (`cost_tracker.py`)
   - Support `query:alias` llm_type pattern
   - Add query-specific summary fields

3. **Update `ExperimentLogger`** (`logger.py`)
   - Add `log_query()` method
   - Create `child_llm_queries.jsonl` file

4. **Update API exposure** (`evolution_api.py`)
   - Add `query_child_llm` to `get_api_functions()`

### Phase 2: Documentation & Testing

5. **Update prompts** (`llm/prompts.py`)
   - Document `query_child_llm()` in Root LLM system prompt

6. **Add tests** (`tests/test_evolution_api.py`)
   - Test basic query functionality
   - Test cost tracking
   - Test logging
   - Test error handling

### Files to Modify

| File | Changes |
|------|---------|
| `src/mango_evolve/evolution_api.py` | Add `query_child_llm()`, update `get_api_functions()` |
| `src/mango_evolve/cost_tracker.py` | Support `query:alias` type, add query summary fields |
| `src/mango_evolve/logger.py` | Add `log_query()` method |
| `src/mango_evolve/llm/prompts.py` | Document new function in system prompt |
| `tests/test_evolution_api.py` | Add query tests |

## Test Plan

### Unit Tests

1. **Basic Query**
   ```python
   def test_query_child_llm_returns_response():
       result = api.query_child_llm("What is 2+2?", model="fast")
       assert "response" in result
       assert "cost" in result
       assert len(api.all_trials) == 0  # No trial created
   ```

2. **Cost Tracking**
   ```python
   def test_query_costs_tracked_separately():
       api.query_child_llm("Question", model="fast")
       summary = api.cost_tracker.get_summary()
       assert summary.query_calls["fast"] == 1
       assert summary.total_query_cost > 0
   ```

3. **Budget Enforcement**
   ```python
   def test_query_respects_budget():
       # Set low budget, exhaust with queries
       with pytest.raises(BudgetExceededError):
           for _ in range(100):
               api.query_child_llm("Expensive query")
   ```

4. **Logging**
   ```python
   def test_query_logged_to_file():
       api.query_child_llm("Test query")
       log_path = api.logger.base_dir / "child_llm_queries.jsonl"
       assert log_path.exists()
   ```

### Integration Tests

5. **Query During Evolution**
   - Start evolution, make queries during generation, verify trials unaffected

6. **Query in Calibration Phase**
   - Verify queries work during calibration without consuming calibration budget

## Open Questions

1. **Should queries work during calibration phase?**
   - Recommendation: Yes, but don't count against calibration budget
   - Queries are useful during calibration for analysis

2. **Should we support parallel queries?**
   - Recommendation: Not in v1. If needed, Root LLM can make sequential queries
   - Can add `query_children_parallel()` later if demand exists

3. **Should query responses be cached?**
   - Recommendation: No caching for v1
   - Queries are typically unique/contextual
   - Can add optional caching parameter later

## Alternatives Considered

### Alternative 1: Use Scratchpad for Notes

Instead of queries, Root LLM writes analysis to scratchpad.

**Rejected because**: Root LLM may lack the expertise/perspective of specialized child models.

### Alternative 2: Extend spawn_child_llm with skip_evaluation Flag

```python
spawn_child_llm(prompt, skip_evaluation=True, skip_trial_record=True)
```

**Rejected because**: Too many flags, confusing API, different return types needed.

### Alternative 3: Separate "Advisor" LLM Configuration

Configure advisor LLMs separately from child LLMs.

**Rejected because**: Unnecessary complexity, child LLMs can serve both purposes.

## Success Metrics

1. **Adoption**: Root LLM uses queries in >50% of experiments
2. **Efficiency**: Query overhead <10% of total child LLM cost
3. **Quality**: Experiments using queries achieve higher scores (A/B test)

## Timeline

- Implementation: 2-3 days
- Testing: 1 day
- Documentation: 0.5 days
- Total: ~4 days

## Appendix: Example Usage in Root LLM

```python
# Analyze why recent trials failed
failed_trials = [t for t in get_top_trials(10) if not t['success']]
if len(failed_trials) >= 3:
    analysis = query_child_llm(
        f"""Analyze these failed trials and identify patterns:

        {[t['error'] for t in failed_trials[:5]]}

        What common issues do you see? How can we avoid them?""",
        model="strong"  # Use stronger model for analysis
    )

    # Use insights to update scratchpad
    update_scratchpad(scratchpad + f"\n\n## Failure Analysis\n{analysis['response']}")
```

```python
# Get help improving a mutation prompt
base_prompt = "Mutate this code to improve the score..."
improved = query_child_llm(
    f"""Improve this mutation prompt to encourage more diverse solutions:

    {base_prompt}

    The goal is to maximize the sum of radii in circle packing.""",
    model="fast"
)

# Use the improved prompt
spawn_child_llm(improved['response'], parent_id="trial_2_5")
```
