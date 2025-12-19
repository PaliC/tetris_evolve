# Context Size and Prompt Caching Analysis

**Date:** December 2024
**Experiment Analyzed:** `circle_packing_v1_20251209_151839`
**Configuration:** 10 generations, 15 children/generation, Claude Sonnet 4.5

---

## Executive Summary

The pineapple_evolve system suffers from **unbounded context growth** that will cause failures at scale. Analysis of the experiment logs reveals that context size grows by approximately **35,000-50,000 tokens per generation**, reaching ~190,000 tokens after just 5 generations. At this rate, a full 10-generation run would require ~400,000+ tokens, exceeding Claude's context limits.

The primary cause is the inclusion of **full source code for every trial** in feedback messages sent to the Root LLM. A single generation's feedback message contains ~75-170KB of text, most of which is redundant code that could be referenced via existing `{{CODE_TRIAL_X_Y}}` tokens.

Current prompt caching implementations provide limited benefit due to the constantly-changing nature of message content.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Context Growth Analysis](#2-context-growth-analysis)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Prompt Caching Evaluation](#4-prompt-caching-evaluation)
5. [Data from Experiment Logs](#5-data-from-experiment-logs)
6. [Recommendations](#6-recommendations)
7. [Remediations](#7-remediations)
   - [7.1 Remove Full Code from Feedback Messages](#71-remediation-1-remove-full-code-from-feedback-messages)
   - [7.2 Implement Message History Pruning](#72-remediation-2-implement-message-history-pruning)
   - [7.3 Fix Prompt Caching Strategy](#73-remediation-3-fix-prompt-caching-strategy)
   - [7.4 Optimize Selection Request Messages](#74-remediation-4-optimize-selection-request-messages)
   - [7.5 Add Context Size Monitoring](#75-remediation-5-add-context-size-monitoring)
   - [7.6 Stateless Generation Architecture](#76-remediation-6-stateless-generation-architecture-advanced)
8. [Remediation Priority Matrix](#8-remediation-priority-matrix)
9. [Validation Plan](#9-validation-plan)

---

## 1. System Architecture Overview

### 1.1 Component Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                     Root LLM Orchestrator                        │
│                      (root_llm.py)                               │
├─────────────────────────────────────────────────────────────────┤
│  self.messages: list[dict]  ← UNBOUNDED GROWTH                  │
│                                                                  │
│  Each generation adds:                                           │
│    1. Assistant response (~10KB)                                 │
│    2. REPL execution result (~20 bytes)                         │
│    3. Selection request (~2-3KB)                                │
│    4. Assistant selection (~3-5KB)                              │
│    5. Feedback message (~75-170KB) ← PRIMARY PROBLEM            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Evolution API                               │
│                    (evolution_api.py)                            │
├─────────────────────────────────────────────────────────────────┤
│  spawn_children_parallel() → Parallel child LLM calls           │
│  _build_generation_feedback_message() ← INCLUDES FULL CODE      │
│  _build_selection_request_message()                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel Workers                              │
│                  (parallel_worker.py)                            │
├─────────────────────────────────────────────────────────────────┤
│  15 parallel processes, each with:                              │
│    - Own Anthropic client                                        │
│    - Cached CHILD_LLM_SYSTEM_PROMPT ← WORKING CORRECTLY         │
│    - Single user message (prompt)                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Message Flow Per Generation

For each generation, the following messages are added to `self.messages`:

| Step | Role | Content | Typical Size |
|------|------|---------|--------------|
| 1 | assistant | REPL code to spawn children | ~10KB |
| 2 | system | "REPL execution result" | ~21 bytes |
| 3 | user | Selection request with trial summaries | ~2-3KB |
| 4 | assistant | Selection response with JSON | ~3-5KB |
| 5 | user | **Feedback with FULL CODE for all trials** | **~75-170KB** |

**The feedback message in step 5 dominates context usage.**

---

## 2. Context Growth Analysis

### 2.1 Measured Context Size by Turn

Analysis of `root_llm_log.jsonl` from the experiment:

```
Turn  0 | Role: assistant  | Size:    9,881 | Cumulative:     9,881 (~  2,470 tokens)
Turn  1 | Role: system     | Size:       21 | Cumulative:     9,902 (~  2,475 tokens)
Turn  2 | Role: user       | Size:    1,847 | Cumulative:    11,749 (~  2,937 tokens)
Turn  3 | Role: assistant  | Size:    3,360 | Cumulative:    15,109 (~  3,777 tokens)
Turn  4 | Role: user       | Size:   85,625 | Cumulative:   100,734 (~ 25,183 tokens) ← Gen 0 feedback
Turn  5 | Role: assistant  | Size:   10,581 | Cumulative:   111,315 (~ 27,828 tokens)
Turn  6 | Role: system     | Size:       21 | Cumulative:   111,336 (~ 27,834 tokens)
Turn  7 | Role: user       | Size:    2,437 | Cumulative:   113,773 (~ 28,443 tokens)
Turn  8 | Role: assistant  | Size:    3,280 | Cumulative:   117,053 (~ 29,263 tokens)
Turn  9 | Role: user       | Size:  138,325 | Cumulative:   255,378 (~ 63,844 tokens) ← Gen 1 feedback
Turn 10 | Role: assistant  | Size:    2,950 | Cumulative:   258,328 (~ 64,582 tokens)
Turn 11 | Role: user       | Size:      109 | Cumulative:   258,437 (~ 64,609 tokens)
Turn 12 | Role: assistant  | Size:   10,847 | Cumulative:   269,284 (~ 67,321 tokens)
Turn 13 | Role: system     | Size:       21 | Cumulative:   269,305 (~ 67,326 tokens)
Turn 14 | Role: user       | Size:    2,277 | Cumulative:   271,582 (~ 67,895 tokens)
Turn 15 | Role: assistant  | Size:    4,738 | Cumulative:   276,320 (~ 69,080 tokens)
Turn 16 | Role: user       | Size:  130,529 | Cumulative:   406,849 (~101,712 tokens) ← Gen 2 feedback
Turn 17 | Role: assistant  | Size:    3,054 | Cumulative:   409,903 (~102,475 tokens)
Turn 18 | Role: user       | Size:      109 | Cumulative:   410,012 (~102,503 tokens)
Turn 19 | Role: assistant  | Size:   11,969 | Cumulative:   421,981 (~105,495 tokens)
Turn 20 | Role: system     | Size:       21 | Cumulative:   422,002 (~105,500 tokens)
Turn 21 | Role: user       | Size:    3,154 | Cumulative:   425,156 (~106,289 tokens)
Turn 22 | Role: assistant  | Size:    4,835 | Cumulative:   429,991 (~107,497 tokens)
Turn 23 | Role: user       | Size:  171,850 | Cumulative:   601,841 (~150,460 tokens) ← Gen 3 feedback
Turn 24 | Role: assistant  | Size:      775 | Cumulative:   602,616 (~150,654 tokens)
Turn 25 | Role: user       | Size:      109 | Cumulative:   602,725 (~150,681 tokens)
Turn 26 | Role: assistant  | Size:   11,376 | Cumulative:   614,101 (~153,525 tokens)
Turn 27 | Role: system     | Size:       21 | Cumulative:   614,122 (~153,530 tokens)
Turn 28 | Role: user       | Size:    2,825 | Cumulative:   616,947 (~154,236 tokens)
Turn 29 | Role: assistant  | Size:    4,230 | Cumulative:   621,177 (~155,294 tokens)
Turn 30 | Role: user       | Size:  141,234 | Cumulative:   762,411 (~190,602 tokens) ← Gen 4 feedback
```

### 2.2 Growth Visualization

```
Context Size (tokens) vs Generation
───────────────────────────────────────────────────────────────────
200K ┤                                                    ████████
     │                                                    ████████
175K ┤                                                    ████████
     │                                           █████████████████
150K ┤                                           █████████████████
     │                                           █████████████████
125K ┤                                           █████████████████
     │                              ██████████████████████████████
100K ┤                              ██████████████████████████████
     │                              ██████████████████████████████
 75K ┤                              ██████████████████████████████
     │                 █████████████████████████████████████████████
 50K ┤                 █████████████████████████████████████████████
     │    ████████████████████████████████████████████████████████
 25K ┤    ████████████████████████████████████████████████████████
     │    ████████████████████████████████████████████████████████
   0 ┼────┴────────────┴──────────────┴───────────────┴───────────
         Gen 0         Gen 1          Gen 2           Gen 3      Gen 4
```

### 2.3 Growth Rate Analysis

| Generation | Cumulative Tokens | Delta | Growth Rate |
|------------|-------------------|-------|-------------|
| 0 | ~25,000 | - | - |
| 1 | ~64,000 | +39,000 | +156% |
| 2 | ~102,000 | +38,000 | +59% |
| 3 | ~150,000 | +48,000 | +47% |
| 4 | ~191,000 | +41,000 | +27% |

**Average growth: ~40,000 tokens per generation**

### 2.4 Projected Context Requirements

| Generations | Estimated Tokens | Feasibility |
|-------------|------------------|-------------|
| 5 | ~230,000 | Possible with Sonnet |
| 6 | ~270,000 | At risk |
| 7 | ~310,000 | Likely to fail |
| 10 | ~430,000 | Will fail |

---

## 3. Root Cause Analysis

### 3.1 Primary Cause: Full Code in Feedback Messages

The `_build_generation_feedback_message()` method in `root_llm.py` (lines 260-289) constructs feedback messages that include the **complete source code** for every trial:

```python
def _build_generation_feedback_message(self) -> str:
    """Build a feedback message with results from the previous generation."""
    prev_gen = self.evolution_api.current_generation - 1
    gen_summary = self.evolution_api.generations[prev_gen]
    trials = gen_summary.trials

    lines = [
        f"Generation {prev_gen} complete. {len(trials)} children spawned.",
        "",
        "Results:",
    ]

    sorted_trials = sorted(
        trials,
        key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
        reverse=True,
    )

    for trial in sorted_trials:
        lines.append(f"  - {trial.trial_id}:")
        lines.append(f"    Code: {trial.code}")      # ← FULL CODE (~3-8KB per trial)
        lines.append(f"    Metrics: {trial.metrics}")
        lines.append(f"    Reasoning: {trial.reasoning}")

    return "\n".join(lines)
```

**Impact calculation:**
- Average code size per trial: ~5KB
- Trials per generation: 15
- Code content per feedback: 15 × 5KB = **~75KB**
- Plus metrics, reasoning, formatting: **~85-170KB total**

### 3.2 Secondary Cause: No Message Pruning

The orchestrator maintains all messages in `self.messages` indefinitely:

```python
class RootLLMOrchestrator:
    def __init__(self, ...):
        # ...
        self.messages: list[dict[str, str]] = []  # Never pruned
```

Every message ever sent is retained, causing linear growth with no upper bound.

### 3.3 Tertiary Cause: Redundant Information

The system already supports `{{CODE_TRIAL_X_Y}}` token substitution (implemented in `utils/prompt_substitution.py`), which allows referencing trial code without including it inline. However, feedback messages don't use this mechanism—they include raw code directly.

This creates redundancy:
1. Code is stored in trial JSON files on disk
2. Code is stored in `all_trials` dictionary in memory
3. Code is included verbatim in feedback messages
4. Code accumulates in conversation history

### 3.4 Why Context Size Matters

Large context causes several problems:

1. **API Limits:** Claude models have maximum context lengths. Exceeding them causes request failures.

2. **Cost:** Input tokens are billed. At $3/million input tokens:
   - Gen 4 context (~190K tokens): ~$0.57 per API call
   - Projected Gen 10 (~430K tokens): ~$1.29 per API call
   - Multiply by 2 calls per generation (spawn + selection)

3. **Latency:** Larger contexts increase processing time.

4. **Quality Degradation:** With very long contexts, models may lose focus on relevant information ("lost in the middle" phenomenon).

---

## 4. Prompt Caching Evaluation

### 4.1 Current Caching Implementation

The codebase implements prompt caching in three locations:

#### 4.1.1 System Prompt Caching (Root LLM)

**Location:** `llm/prompts.py:319-355`

```python
def get_root_system_prompt_parts(...) -> list[dict]:
    return [
        {
            "type": "text",
            "text": ROOT_LLM_SYSTEM_PROMPT_STATIC,  # ~7KB static content
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": dynamic_part,  # ~200 bytes, changes each generation
        },
    ]
```

**Effectiveness:** ✅ **WORKING**
- The static system prompt (~7KB) is cached
- Only the dynamic portion (current generation number) changes
- Provides consistent cache hits across all Root LLM calls

#### 4.1.2 Message History Caching (Root LLM)

**Location:** `root_llm.py:153-202`

```python
def _prepare_messages_with_caching(self, messages: list[dict[str, str]]) -> list[dict]:
    # Find second-to-last user message
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    cache_index = user_indices[-2]

    # Add cache_control to that message
    for i, msg in enumerate(messages):
        if i == cache_index:
            result.append({
                "role": msg["role"],
                "content": [{
                    "type": "text",
                    "text": msg["content"],
                    "cache_control": {"type": "ephemeral"},
                }],
            })
```

**Effectiveness:** ❌ **LIMITED BENEFIT**

The strategy caches at the second-to-last user message, intending to create a stable prefix. However:

1. **Cache breaks every turn:** Each new message shifts which message is "second-to-last"
2. **Massive cache writes:** Each feedback message (~100KB+) is written to cache, then becomes stale
3. **No prefix stability:** Messages before the cache point also change (new assistant responses)

**Anthropic's cache requires exact prefix matching.** If byte 1 of message 1 differs, the entire cache misses.

#### 4.1.3 Child LLM System Prompt Caching

**Location:** `parallel_worker.py:122-130`

```python
if system_prompt:
    kwargs["system"] = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
```

**Effectiveness:** ✅ **WORKING WELL**
- `CHILD_LLM_SYSTEM_PROMPT` is identical across all child LLM calls
- 15 parallel workers benefit from the same cached prompt
- First worker creates cache, remaining 14 read from it (90% discount)

### 4.2 Why Message Caching Doesn't Work

Anthropic's prompt caching works on **exact prefix matching**. The cache key is essentially a hash of all content up to the cache breakpoint.

**Problem illustration:**

```
Call 1: [System] [User1] [Asst1] [User2*] [Asst2] [User3]
                              ↑ cache point

Call 2: [System] [User1] [Asst1] [User2] [Asst2] [User3*] [Asst3] [User4]
                                              ↑ cache point (different!)
```

Even though User1 and User2 are identical, the cache from Call 1 cannot be used in Call 2 because:
1. The cache point has moved
2. New messages exist after the old cache point
3. Anthropic caches the entire prefix, not individual messages

### 4.3 Cache Efficiency Metrics

Based on the cost tracker implementation, the system tracks:
- `cache_creation_input_tokens`: Tokens written to cache (25% price premium)
- `cache_read_input_tokens`: Tokens read from cache (90% discount)

For Root LLM calls, we expect:
- High `cache_creation_input_tokens` (constantly writing new cache entries)
- Low `cache_read_input_tokens` (rarely hitting existing cache)

For Child LLM calls, we expect:
- Moderate `cache_creation_input_tokens` (first worker creates)
- High `cache_read_input_tokens` (14 workers read from cache)

---

## 5. Data from Experiment Logs

### 5.1 Feedback Message Content Analysis

Examining Turn 4 (Generation 0 feedback), the message structure is:

```
Generation 0 complete. 15 children spawned.

Results:
  - trial_0_13:
    Code: import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square.
    ...
    [APPROXIMATELY 5,500 CHARACTERS OF CODE]
    ...
    Metrics: {'valid': False, 'sum_radii': 0.0, 'target_ratio': 0.0, ...}
    Reasoning: [reasoning text]

  - trial_0_14:
    Code: [ANOTHER ~5,000 CHARACTERS]
    ...

  [REPEATED FOR ALL 15 TRIALS]
```

### 5.2 Message Size Breakdown

| Component | Size per Trial | × 15 Trials | Total |
|-----------|---------------|-------------|-------|
| Trial header | ~50 bytes | 750 bytes | 0.75 KB |
| Code | ~5,000 bytes | 75,000 bytes | **75 KB** |
| Metrics | ~200 bytes | 3,000 bytes | 3 KB |
| Reasoning | ~300 bytes | 4,500 bytes | 4.5 KB |
| **Total** | ~5,550 bytes | 83,250 bytes | **~83 KB** |

The code portion represents **~90% of the feedback message size**.

### 5.3 Trial Code Statistics

From `experiments/circle_packing_v1_20251209_151839/generations/gen_0/`:

| Trial | Code Size | Response Size |
|-------|-----------|---------------|
| trial_0_0 | 5,519 bytes | 5,533 bytes |
| trial_0_1 | ~5,000 bytes | ~5,000 bytes |
| trial_0_5 | ~8,000 bytes | ~8,000 bytes |
| trial_0_9 | ~10,000 bytes | ~10,000 bytes |

Average: **~5-6 KB per trial**

---

## 6. Recommendations

### 6.1 High Priority: Eliminate Code from Feedback

**Change:** Modify `_build_generation_feedback_message()` to exclude raw code.

**Before:**
```python
for trial in sorted_trials:
    lines.append(f"    Code: {trial.code}")
```

**After:**
```python
for trial in sorted_trials:
    gen = trial.generation
    trial_num = trial.trial_id.split('_')[-1]
    lines.append(f"    Code: Use {{{{CODE_TRIAL_{gen}_{trial_num}}}}} to reference")
```

**Impact:** Reduces feedback messages from ~85KB to ~5KB (94% reduction).

### 6.2 High Priority: Implement Context Summarization

**Strategy:** After N generations, replace detailed history with summaries.

```python
def _summarize_generation(self, gen_num: int) -> str:
    gen = self.generations[gen_num]
    top_trials = sorted(gen.trials, key=lambda t: t.metrics.get('sum_radii', 0), reverse=True)[:3]
    return f"""Generation {gen_num} Summary:
- Trials: {len(gen.trials)}, Best score: {gen.best_score:.4f}
- Top performers: {', '.join(t.trial_id for t in top_trials)}
- Selected for next gen: {', '.join(gen.selected_trial_ids)}"""
```

### 6.3 Medium Priority: Sliding Window for Messages

**Strategy:** Keep only the last N generations of messages in full detail.

```python
def _prune_old_messages(self, keep_generations: int = 2):
    """Remove detailed messages from older generations."""
    # Keep: system prompt, recent generations, current generation
    # Remove: detailed feedback from gen < (current - keep_generations)
```

### 6.4 Medium Priority: Restructure for Better Caching

**Strategy:** Put static content at the beginning of user messages.

```python
def _build_user_message(self, dynamic_content: str) -> str:
    static_prefix = """You are managing circle packing evolution.
    Available commands: spawn_children_parallel(), evaluate_program(), terminate_evolution()
    Use {{CODE_TRIAL_X_Y}} to reference trial code.
    """
    return static_prefix + "\n\n" + dynamic_content
```

This creates a cacheable prefix that remains constant across turns.

### 6.5 Low Priority: Consider Alternative Architectures

**Options:**
1. **Stateless per-generation:** Each generation is a fresh conversation with summarized history
2. **External memory:** Store detailed history in files, provide summaries to LLM
3. **Hierarchical summarization:** Progressively compress older information

---

## Appendix A: Code Locations

| File | Lines | Purpose |
|------|-------|---------|
| `root_llm.py` | 260-289 | `_build_generation_feedback_message()` - Primary issue |
| `root_llm.py` | 153-202 | `_prepare_messages_with_caching()` - Message caching |
| `root_llm.py` | 119 | `self.messages` - Unbounded list |
| `llm/prompts.py` | 319-355 | `get_root_system_prompt_parts()` - System caching |
| `parallel_worker.py` | 122-130 | Child LLM system prompt caching |
| `evolution_api.py` | 359-510 | `spawn_children_parallel()` - Child spawning |

## Appendix B: Token Estimation

Throughout this document, token counts are estimated using the approximation:
```
tokens ≈ characters / 4
```

This is a rough heuristic. Actual token counts depend on the specific tokenizer and content. For precise measurements, use the Anthropic token counting API.

## Appendix C: Cost Implications

With Claude Sonnet pricing ($3/M input, $15/M output):

| Scenario | Input Tokens | Input Cost | Notes |
|----------|--------------|------------|-------|
| Gen 0 call | 25,000 | $0.075 | Baseline |
| Gen 4 call | 190,000 | $0.57 | 7.6x increase |
| Gen 10 call (projected) | 430,000 | $1.29 | 17x increase |
| Full run (10 gens, 2 calls each) | ~3,000,000 | $9.00 | Just Root LLM |

Cache savings potential:
- If 80% of tokens could be cached: 90% discount on those → ~$7.20 savings
- Current implementation: Minimal cache benefit due to prefix instability

---

## 7. Remediations

This section provides concrete, implementable fixes ordered by impact and complexity.

### 7.1 Remediation 1: Remove Full Code from Feedback Messages

**Priority:** CRITICAL
**Effort:** Low (30 minutes)
**Impact:** ~94% reduction in feedback message size

#### Problem
The `_build_generation_feedback_message()` method includes full source code for every trial, adding ~75KB per generation.

#### Solution
Replace inline code with references to the existing `{{CODE_TRIAL_X_Y}}` token system.

#### Implementation

**File:** `src/pineapple_evolve/root_llm.py`

**Current code (lines 260-289):**
```python
def _build_generation_feedback_message(self) -> str:
    """Build a feedback message with results from the previous generation."""
    prev_gen = self.evolution_api.current_generation - 1
    if prev_gen < 0:
        return ""

    gen_summary = self.evolution_api.generations[prev_gen]
    trials = gen_summary.trials

    lines = [
        f"Generation {prev_gen} complete. {len(trials)} children spawned.",
        "",
        "Results:",
    ]

    sorted_trials = sorted(
        trials,
        key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
        reverse=True,
    )

    for trial in sorted_trials:
        lines.append(f"  - {trial.trial_id}:")
        lines.append(f"    Code: {trial.code}")
        lines.append(f"    Metrics: {trial.metrics}")
        lines.append(f"    Reasoning: {trial.reasoning}")

    return "\n".join(lines)
```

**Remediated code:**
```python
def _build_generation_feedback_message(self) -> str:
    """Build a feedback message with results from the previous generation."""
    prev_gen = self.evolution_api.current_generation - 1
    if prev_gen < 0:
        return ""

    gen_summary = self.evolution_api.generations[prev_gen]
    trials = gen_summary.trials

    lines = [
        f"Generation {prev_gen} complete. {len(trials)} children spawned.",
        "",
        "Results (use {{CODE_TRIAL_X_Y}} tokens to reference code when needed):",
        "",
    ]

    sorted_trials = sorted(
        trials,
        key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
        reverse=True,
    )

    for trial in sorted_trials:
        score = trial.metrics.get("sum_radii", 0) if trial.success else 0
        valid = "valid" if trial.success else "INVALID"

        # Extract generation and trial number for code reference token
        parts = trial.trial_id.split("_")
        gen_num, trial_num = parts[1], parts[2]
        code_ref = f"{{{{CODE_TRIAL_{gen_num}_{trial_num}}}}}"

        lines.append(f"  - **{trial.trial_id}** [{valid}]")
        lines.append(f"    Score: {score:.4f}")
        lines.append(f"    Code reference: {code_ref}")
        if trial.reasoning:
            # Truncate reasoning to first 150 chars
            reasoning_short = trial.reasoning[:150].replace('\n', ' ')
            lines.append(f"    Approach: {reasoning_short}...")
        if not trial.success and trial.error:
            error_short = str(trial.error)[:100]
            lines.append(f"    Error: {error_short}")
        lines.append("")

    return "\n".join(lines)
```

#### Expected Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Feedback message size | ~85 KB | ~5 KB | 94% reduction |
| Context after 5 gens | ~230K tokens | ~50K tokens | 78% reduction |
| Feasible generations | ~6 | 20+ | 3x+ increase |

---

### 7.2 Remediation 2: Implement Message History Pruning

**Priority:** HIGH
**Effort:** Medium (2-3 hours)
**Impact:** Constant context size regardless of generation count

#### Problem
`self.messages` grows unboundedly, accumulating all historical messages.

#### Solution
Implement a sliding window that keeps only recent generations in full detail, replacing older content with summaries.

#### Implementation

**File:** `src/pineapple_evolve/root_llm.py`

**Add new method:**
```python
def _prune_message_history(self, keep_recent_generations: int = 2) -> None:
    """
    Prune old messages to prevent unbounded context growth.

    Keeps:
    - Initial user message (problem setup)
    - Messages from the last N generations
    - Replaces older generation messages with summaries

    Args:
        keep_recent_generations: Number of recent generations to keep in full detail
    """
    if self.evolution_api.current_generation <= keep_recent_generations:
        return  # Not enough history to prune

    cutoff_generation = self.evolution_api.current_generation - keep_recent_generations

    # Build summary of old generations
    old_gen_summaries = []
    for gen_num in range(cutoff_generation):
        gen = self.evolution_api.generations[gen_num]
        summary = (
            f"Gen {gen_num}: {len(gen.trials)} trials, "
            f"best={gen.best_score:.4f} ({gen.best_trial_id}), "
            f"selected={gen.selected_trial_ids}"
        )
        old_gen_summaries.append(summary)

    # Create consolidated history message
    history_summary = {
        "role": "user",
        "content": (
            f"[Historical Summary - Generations 0 to {cutoff_generation - 1}]\n"
            + "\n".join(old_gen_summaries)
            + "\n\n[Detailed history continues from generation {cutoff_generation}]"
        ),
    }

    # Find where recent generations start in message history
    # Each generation adds ~5 messages, so estimate the cutoff point
    messages_per_gen = 5
    keep_from_index = max(1, len(self.messages) - (keep_recent_generations * messages_per_gen))

    # Rebuild messages: initial message + summary + recent messages
    self.messages = (
        [self.messages[0]]  # Keep initial "Begin generation 0" message
        + [history_summary]
        + self.messages[keep_from_index:]
    )
```

**Integrate into the run loop (in `run()` method, after advancing generation):**
```python
# After: self.evolution_api._advance_generation(...)
# Add:
self._prune_message_history(keep_recent_generations=2)
```

#### Expected Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Context growth | O(n) per generation | O(1) constant | Bounded |
| Max context size | Unbounded | ~80K tokens | Predictable |
| Generation limit | ~6-7 | Unlimited | No limit |

---

### 7.3 Remediation 3: Fix Prompt Caching Strategy

**Priority:** MEDIUM
**Effort:** Medium (2 hours)
**Impact:** 50-70% reduction in Root LLM input costs

#### Problem
Current message caching strategy doesn't create stable prefixes, resulting in constant cache misses.

#### Solution
Restructure messages so that a stable, cacheable prefix exists at the beginning of the conversation.

#### Implementation

**File:** `src/pineapple_evolve/root_llm.py`

**Replace `_prepare_messages_with_caching()` with a better strategy:**

```python
def _prepare_messages_with_caching(
    self, messages: list[dict[str, str]]
) -> list[dict]:
    """
    Prepare messages with cache_control for optimal prompt caching.

    Strategy: Cache at the END of a stable prefix that won't change.
    For multi-turn conversations, this means caching after the initial
    setup message, not at a moving target like "second-to-last".

    Since our initial user message is stable, we cache there.
    """
    if len(messages) < 1:
        return messages

    result = []
    for i, msg in enumerate(messages):
        if i == 0:
            # Cache the first user message (stable "Begin generation 0..." prompt)
            result.append({
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            })
        else:
            result.append(msg)

    return result
```

**Alternative: Use a stable context preamble**

Create a substantial stable prefix by combining problem context with the first message:

```python
def build_initial_messages(self) -> list[dict[str, str]]:
    """Build the initial messages for the conversation."""
    # Create a substantial cacheable prefix
    stable_context = """## Evolution Context

You are orchestrating circle packing evolution. Your goal is to find algorithms
that pack 26 circles into a unit square [0,1]x[0,1] maximizing sum of radii.

### Available Tools
- spawn_children_parallel(children): Spawn multiple child LLMs in parallel
- evaluate_program(code): Evaluate code directly
- terminate_evolution(reason): End evolution early

### Code References
Use {{CODE_TRIAL_X_Y}} tokens to reference code from previous trials.
Example: {{CODE_TRIAL_0_3}} references trial 3 from generation 0.

### Your Task
For each generation, spawn diverse children exploring different strategies.
After spawning, you'll select promising trials to carry forward.
"""

    self.messages = [
        {
            "role": "user",
            "content": (
                stable_context +
                f"\n---\n\nBegin generation 0. Spawn up to {self.max_children_per_generation} "
                "children exploring different circle packing strategies."
            ),
        }
    ]
    return self.messages
```

#### Expected Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cache hit rate (Root) | ~5% | ~40-60% | 8-12x improvement |
| Effective input cost | $3/M tokens | ~$1.5/M tokens | 50% reduction |

---

### 7.4 Remediation 4: Optimize Selection Request Messages

**Priority:** LOW
**Effort:** Low (30 minutes)
**Impact:** ~50% reduction in selection request size

#### Problem
`_build_selection_request_message()` includes verbose trial details that could be more concise.

#### Solution
Streamline the selection request format.

#### Implementation

**File:** `src/pineapple_evolve/root_llm.py`

**Current approach builds ~2-3KB messages. Optimize to ~1KB:**

```python
def _build_selection_request_message(self) -> str:
    """Build a concise selection request message."""
    current_gen = self.evolution_api.current_generation
    gen_summary = self.evolution_api.generations[current_gen]
    trials = gen_summary.trials

    sorted_trials = sorted(
        trials,
        key=lambda t: t.metrics.get("sum_radii", 0) if t.success else 0,
        reverse=True,
    )

    lines = [
        f"## Generation {current_gen} Complete - Select Trials",
        "",
        "| Rank | Trial | Score | Status |",
        "|------|-------|-------|--------|",
    ]

    for i, trial in enumerate(sorted_trials, 1):
        score = trial.metrics.get("sum_radii", 0) if trial.success else 0
        status = "✓" if trial.success else "✗"
        lines.append(f"| {i} | {trial.trial_id} | {score:.4f} | {status} |")

    lines.extend([
        "",
        "Select 2-5 trials for next generation. Respond with:",
        "```selection",
        '{"selections": [{"trial_id": "...", "reasoning": "...", "category": "performance|diversity|potential"}], "summary": "..."}',
        "```",
    ])

    return "\n".join(lines)
```

---

### 7.5 Remediation 5: Add Context Size Monitoring

**Priority:** MEDIUM
**Effort:** Low (1 hour)
**Impact:** Early warning system for context issues

#### Problem
No visibility into context size growth during execution.

#### Solution
Add context size tracking and warnings.

#### Implementation

**File:** `src/pineapple_evolve/root_llm.py`

**Add monitoring method:**
```python
def _get_context_size_estimate(self) -> dict:
    """Estimate current context size."""
    total_chars = sum(len(m.get("content", "")) for m in self.messages)
    estimated_tokens = total_chars // 4

    return {
        "total_chars": total_chars,
        "estimated_tokens": estimated_tokens,
        "message_count": len(self.messages),
        "warning": estimated_tokens > 150_000,
        "critical": estimated_tokens > 250_000,
    }

def _check_context_health(self) -> None:
    """Check context size and warn if growing too large."""
    stats = self._get_context_size_estimate()

    if stats["critical"]:
        tqdm.write(
            f"  ⚠️  CRITICAL: Context size ~{stats['estimated_tokens']:,} tokens. "
            "Risk of API failure. Consider pruning history."
        )
    elif stats["warning"]:
        tqdm.write(
            f"  ⚠️  WARNING: Context size ~{stats['estimated_tokens']:,} tokens. "
            "Approaching limits."
        )
```

**Add to run loop:**
```python
# At the start of each generation loop:
self._check_context_health()
```

---

### 7.6 Remediation 6: Stateless Generation Architecture (Advanced)

**Priority:** LOW (architectural change)
**Effort:** High (1-2 days)
**Impact:** Complete elimination of context growth issues

#### Problem
Multi-turn conversation architecture inherently accumulates context.

#### Solution
Refactor to use single-turn calls with explicit state injection.

#### Conceptual Implementation

```python
class StatelessOrchestrator:
    """Each generation is a fresh, single-turn conversation."""

    def _build_generation_prompt(self, generation: int) -> str:
        """Build a complete prompt for one generation."""
        # Historical summary (compressed)
        history = self._build_history_summary(up_to_generation=generation)

        # Current state
        current_state = self._build_current_state(generation)

        # Task instructions
        task = self._build_task_instructions(generation)

        return f"{history}\n\n{current_state}\n\n{task}"

    def run_generation(self, generation: int) -> GenerationResult:
        """Run a single generation as one API call."""
        prompt = self._build_generation_prompt(generation)

        response = self.root_llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,  # Cached
            max_tokens=4096,
        )

        # Parse and execute REPL blocks
        # ... rest of logic
```

**Benefits:**
- Context size is O(1), not O(n)
- Each call is independent and can be retried
- Simpler debugging and testing
- Better prompt caching (system prompt always cached)

**Tradeoffs:**
- Requires explicit state serialization
- Loses conversational continuity
- More engineering effort

---

## 8. Remediation Priority Matrix

| Remediation | Priority | Effort | Impact | Dependencies |
|-------------|----------|--------|--------|--------------|
| 7.1 Remove code from feedback | CRITICAL | Low | 94% size reduction | None |
| 7.2 Message history pruning | HIGH | Medium | Constant context | 7.1 recommended first |
| 7.3 Fix prompt caching | MEDIUM | Medium | 50% cost reduction | None |
| 7.4 Optimize selection requests | LOW | Low | Minor size reduction | None |
| 7.5 Add context monitoring | MEDIUM | Low | Visibility/debugging | None |
| 7.6 Stateless architecture | LOW | High | Complete fix | Major refactor |

### Recommended Implementation Order

1. **Immediate (Day 1):** Implement 7.1 (remove code from feedback) - highest impact, lowest effort
2. **Short-term (Week 1):** Add 7.5 (monitoring) to track improvements
3. **Medium-term (Week 2):** Implement 7.2 (pruning) for guaranteed bounded context
4. **Optimization (Week 3+):** Implement 7.3 (caching) and 7.4 (selection) for cost reduction
5. **Future consideration:** Evaluate 7.6 (stateless) if other remediations insufficient

---

## 9. Validation Plan

After implementing remediations, validate with:

### 9.1 Unit Tests
```python
def test_feedback_message_size():
    """Ensure feedback messages stay under threshold."""
    orchestrator = RootLLMOrchestrator(config)
    # ... setup with mock trials
    feedback = orchestrator._build_generation_feedback_message()
    assert len(feedback) < 10_000, f"Feedback too large: {len(feedback)} chars"

def test_context_growth_bounded():
    """Ensure context doesn't grow unboundedly."""
    orchestrator = RootLLMOrchestrator(config)
    for gen in range(10):
        # ... simulate generation
        stats = orchestrator._get_context_size_estimate()
        assert stats["estimated_tokens"] < 100_000, f"Context too large at gen {gen}"
```

### 9.2 Integration Test
Run a full 10-generation experiment and verify:
- [ ] No API failures due to context size
- [ ] Context size remains under 100K tokens throughout
- [ ] Cache hit rate > 30% for Root LLM calls
- [ ] Total cost reduced by at least 50%

### 9.3 Metrics to Track
- Context size per generation (should be ~constant after remediation)
- Cache creation vs cache read tokens (read should exceed creation)
- Total API cost per generation (should decrease)
- Time per API call (should not increase significantly)
