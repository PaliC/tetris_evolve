# Task: Expose Trials Database as REPL Variable with Rich Trial Objects

## Objective
Make the trials database directly accessible in the REPL as a queryable `trials` variable, and ensure spawn functions return rich TrialView objects that the Root LLM can interact with naturally.

## Context
Currently, the Root LLM must call `get_top_trials()` or `get_trial_code([ids])` to access trial data. This is limiting - the LLM cannot write arbitrary analysis code. AlphaEvolve uses a "Program Database" that the system can query flexibly.

## Requirements

### 1. Create TrialView Class (read-only view of a trial)

Create a new class in `src/mango_evolve/evolution_api.py` (or a new file `src/mango_evolve/repl_proxies.py`):

```python
@dataclass
class TrialView:
    """Read-only view of a trial for REPL access."""
    trial_id: str
    code: str
    score: float  # Convenience: metrics.get("score", 0) if valid else 0
    success: bool
    generation: int
    parent_id: str | None
    reasoning: str
    error: str | None
    model_alias: str | None
    metrics: dict  # Full metrics dict

    @classmethod
    def from_trial_result(cls, trial: TrialResult) -> "TrialView":
        """Create from internal TrialResult."""
        score = trial.metrics.get("score", 0) if trial.success else 0
        return cls(
            trial_id=trial.trial_id,
            code=trial.code,
            score=score,
            success=trial.success,
            generation=trial.generation,
            parent_id=trial.parent_id,
            reasoning=trial.reasoning,
            error=trial.error,
            model_alias=trial.model_alias,
            metrics=trial.metrics,
        )

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"<Trial {self.trial_id} [{status}] score={self.score:.6f}>"
```

### 2. Create TrialsProxy Class (queryable collection)

The interface should be simple: basic collection operations + one powerful `filter()` method.

```python
class TrialsProxy:
    """Read-only queryable view of all trials, injected into REPL."""

    def __init__(self, api: "EvolutionAPI"):
        self._api = api

    def __len__(self) -> int:
        return len(self._api.all_trials)

    def __iter__(self):
        for trial in self._api.all_trials.values():
            yield TrialView.from_trial_result(trial)

    def __getitem__(self, trial_id: str) -> TrialView:
        trial = self._api.all_trials.get(trial_id)
        if trial is None:
            raise KeyError(f"Trial {trial_id} not found")
        return TrialView.from_trial_result(trial)

    def __contains__(self, trial_id: str) -> bool:
        return trial_id in self._api.all_trials

    def filter(
        self,
        *,
        # Direct attribute filters
        success: bool | None = None,
        generation: int | None = None,
        parent_id: str | None = None,
        model_alias: str | None = None,
        # Lineage filters
        ancestor_of: str | None = None,  # Find trials that are ancestors of this trial
        descendant_of: str | None = None,  # Find trials that descend from this trial
        # Custom predicate (lambda)
        predicate: Callable[[TrialView], bool] | None = None,
        # Sorting and limiting
        sort_by: str | None = None,  # "score", "-score" (descending), "generation", etc.
        limit: int | None = None,
    ) -> list[TrialView]:
        """
        Filter trials by various criteria.

        Examples:
            # Top 5 by score
            trials.filter(success=True, sort_by="-score", limit=5)

            # All from generation 2
            trials.filter(generation=2)

            # Custom predicate
            trials.filter(predicate=lambda t: t.score > 2.4 and "grid" in t.reasoning)

            # All descendants of a trial
            trials.filter(descendant_of="trial_0_3")

            # Combined filters
            trials.filter(success=True, generation=1, sort_by="-score", limit=3)
        """
        # Build set of descendant/ancestor trial IDs if needed
        descendant_ids: set[str] | None = None
        if descendant_of is not None:
            descendant_ids = self._get_all_descendant_ids(descendant_of)

        ancestor_ids: set[str] | None = None
        if ancestor_of is not None:
            ancestor_ids = self._get_all_ancestor_ids(ancestor_of)

        results = []
        for trial in self._api.all_trials.values():
            # Apply direct attribute filters
            if success is not None and trial.success != success:
                continue
            if generation is not None and trial.generation != generation:
                continue
            if parent_id is not None and trial.parent_id != parent_id:
                continue
            if model_alias is not None and trial.model_alias != model_alias:
                continue

            # Apply lineage filters
            if descendant_ids is not None and trial.trial_id not in descendant_ids:
                continue
            if ancestor_ids is not None and trial.trial_id not in ancestor_ids:
                continue

            view = TrialView.from_trial_result(trial)

            # Apply custom predicate
            if predicate is not None and not predicate(view):
                continue

            results.append(view)

        # Apply sorting
        if sort_by is not None:
            descending = sort_by.startswith("-")
            field = sort_by.lstrip("-")
            results.sort(key=lambda t: getattr(t, field, 0), reverse=descending)

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return results

    def _get_all_descendant_ids(self, trial_id: str) -> set[str]:
        """Get all trial IDs that descend from the given trial (recursive)."""
        descendants = set()
        for trial in self._api.all_trials.values():
            if trial.parent_id == trial_id:
                descendants.add(trial.trial_id)
                descendants.update(self._get_all_descendant_ids(trial.trial_id))
        return descendants

    def _get_all_ancestor_ids(self, trial_id: str) -> set[str]:
        """Get all trial IDs that are ancestors of the given trial."""
        ancestors = set()
        trial = self._api.all_trials.get(trial_id)
        while trial and trial.parent_id:
            ancestors.add(trial.parent_id)
            trial = self._api.all_trials.get(trial.parent_id)
        return ancestors

    def __repr__(self) -> str:
        total = len(self)
        success = sum(1 for t in self._api.all_trials.values() if t.success)
        return f"<Trials: {total} total, {success} successful>"
```

**Key design decisions:**
- Single `filter()` method handles all query patterns
- `sort_by` with `-` prefix for descending (e.g., `-score`)
- `limit` for top-N queries
- `predicate` for arbitrary lambda filtering
- `descendant_of` / `ancestor_of` for lineage queries
- All parameters are keyword-only (`*`) for clarity

### 3. Inject into REPL

Modify `EvolutionAPI.get_api_functions()` to also return variables to inject:

```python
def get_repl_namespace(self) -> dict[str, Any]:
    """Get functions AND variables to inject into REPL."""
    return {
        # Functions (existing)
        "spawn_child_llm": self.spawn_child_llm,
        "spawn_children_parallel": self.spawn_children_parallel,
        "evaluate_program": self.evaluate_program,
        "terminate_evolution": self.terminate_evolution,
        "get_top_trials": self.get_top_trials,
        "get_trial_code": self.get_trial_code,
        "update_scratchpad": self.update_scratchpad,
        "end_calibration_phase": self.end_calibration_phase,
        "get_calibration_status": self.get_calibration_status,
        # Variables (new)
        "trials": TrialsProxy(self),
    }
```

Update `REPLEnvironment.__init__` and `RootLLMOrchestrator.__init__` to use `get_repl_namespace()` instead of `get_api_functions()`.

### 4. Update spawn_child_llm Return Type

Modify `spawn_child_llm` and `spawn_children_parallel` to return `TrialView` objects:

```python
def spawn_child_llm(...) -> TrialView:
    """..."""
    # ... existing code ...
    self._record_trial(trial)
    return TrialView.from_trial_result(trial)

def spawn_children_parallel(...) -> list[TrialView]:
    """..."""
    # ... existing code ...
    return [TrialView.from_trial_result(self.all_trials[tid]) for tid in trial_ids]
```

**Note**: For backward compatibility, keep the `.to_dict()` method on TrialView so existing code that expects dicts can call it.

### 5. Update System Prompt

Update `src/mango_evolve/llm/prompts.py` to document the new `trials` variable:

```python
# Add to ROOT_SYSTEM_PROMPT:
"""
## REPL Variables

### `trials` - Query all trials
A live view of all trials across all generations.

**Basic access:**
- `trials["trial_0_5"]` - Get specific trial by ID
- `len(trials)` - Total trial count
- `for t in trials: ...` - Iterate all trials
- `"trial_0_5" in trials` - Check if trial exists

**Filtering with `trials.filter()`:**
```python
# Top 5 by score
trials.filter(success=True, sort_by="-score", limit=5)

# All from generation 2
trials.filter(generation=2)

# Custom predicate (lambda)
trials.filter(predicate=lambda t: t.score > 2.4 and "grid" in t.reasoning)

# All descendants of a trial
trials.filter(descendant_of="trial_0_3")

# All ancestors of a trial
trials.filter(ancestor_of="trial_2_5")

# Combined: top 3 successful from gen 1
trials.filter(success=True, generation=1, sort_by="-score", limit=3)

# Filter by model
trials.filter(model_alias="fast", success=True)
```

**Filter parameters:**
- `success`: bool - Filter by success/failure
- `generation`: int - Filter by generation number
- `parent_id`: str - Filter by direct parent
- `model_alias`: str - Filter by child LLM model
- `descendant_of`: str - All trials descending from this trial
- `ancestor_of`: str - All ancestors of this trial
- `predicate`: lambda - Custom filter function
- `sort_by`: str - Sort field ("-score" for descending)
- `limit`: int - Max results to return

**Trial attributes:**
`.trial_id`, `.code`, `.score`, `.success`, `.generation`,
`.parent_id`, `.reasoning`, `.error`, `.model_alias`, `.metrics`
"""
```

### 6. Maintain Observability

**Important**: Do NOT remove or modify the existing file logging. The trial JSON files in `experiments/.../generations/gen_X/trial_X_Y.json` must continue to be written. The TrialsProxy is a read-only view over the in-memory data.

### 7. Tests

Add tests in `tests/test_repl_proxies.py`:
- Test TrialsProxy `__len__`, `__iter__`, `__getitem__`, `__contains__`
- Test `filter()` with each parameter individually
- Test `filter()` with combined parameters
- Test `filter()` with `predicate` lambda
- Test `filter()` with `sort_by` (ascending and descending)
- Test `filter()` with `limit`
- Test `filter()` with `descendant_of` and `ancestor_of`
- Test TrialView properties and `__repr__`
- Test injection into REPL namespace
- Test that spawn_child_llm returns TrialView
- Test backward compatibility with `.to_dict()`

## Files to Modify
- `src/mango_evolve/evolution_api.py` - Add TrialView, TrialsProxy, update spawn returns
- `src/mango_evolve/repl.py` - Update to accept namespace dict
- `src/mango_evolve/root_llm.py` - Update REPL initialization
- `src/mango_evolve/llm/prompts.py` - Document trials variable
- `tests/test_repl_proxies.py` (new) - Tests for proxy classes
- `tests/test_evolution_api.py` - Update spawn tests for new return type

## Acceptance Criteria
1. Root LLM can access `trials` variable in REPL
2. `trials.filter(success=True, min_score=2.4)` works
3. `spawn_child_llm()` returns a TrialView with `.score`, `.code` etc.
4. All existing trial file logging continues unchanged
5. Existing tests pass (may need updates for new return types)
6. New tests cover proxy classes
