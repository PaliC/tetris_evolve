# Selection Mechanism Analysis: Cross-Generation Selection Problem

**Document Type**: Technical Handoff / Problem Analysis
**Experiment Analyzed**: `circle_packing_v1_20251216_174624`
**Date**: 2025-12-31

---

## Executive Summary

Analysis of the `circle_packing_v1` experiment reveals a fundamental tension in MangoEvolve's selection mechanism. The system formally restricts selection to the current generation, but the Root LLM naturally attempts to use historical trials as mutation bases—working around the limitation through code token references. This creates a gap between what is formally tracked and what actually happens during evolution.

**Key Finding**: In Generation 4→5, the Root LLM explicitly used historical parents (`trial_3_0`, `trial_3_2`, `trial_3_5`) despite only formally selecting `trial_4_6`. This demonstrates that cross-generation mutation is already occurring, but outside the formal selection tracking system.

---

## 1. Experiment Overview

| Metric | Value |
|--------|-------|
| **Duration** | 2025-12-16, 17:46 - 19:26 (1h 40m) |
| **Generations** | 8 (0-7) |
| **Total Trials** | 120 (15 per generation) |
| **Success Rate** | 62.5% (75/120 trials completed) |
| **Best Score** | 2.6278 (trial_7_6) |
| **Target** | 2.635 |
| **Achievement** | 99.73% of target |
| **Termination** | max_generations_reached |

### Score Progression
```
Gen 0: 2.5569 (baseline)
Gen 1: 2.6058 (+0.0489) ← Major jump
Gen 2: 2.6079 (+0.0021)
Gen 3: 2.6135 (+0.0056)
Gen 4: 2.6185 (+0.0050) ← Only 2/15 trials succeeded
Gen 5: 2.6243 (+0.0058)
Gen 6: 2.6276 (+0.0033)
Gen 7: 2.6278 (+0.0002) ← Stalled
```

---

## 2. The Selection Mechanism

### 2.1 How Selection Currently Works

When a generation completes, the Root LLM is asked to select trials for mutation in the next generation. The `_advance_generation()` function in `evolution_api.py` processes these selections with the following behavior:

1. Root LLM provides a list of `trial_id`s with reasoning
2. The system filters this list to only include trials from the **current generation**
3. Any trial_ids from previous generations are silently dropped
4. The filtered selections are recorded in `experiment.json`

### 2.2 How Mutation Actually Works

Despite the selection filtering, the Root LLM can reference **any historical trial** when spawning children:

1. **Code tokens**: `{{CODE_TRIAL_3_0}}` in a child prompt injects that trial's code
2. **API function**: `get_trial_code(["trial_3_0", "trial_2_5"])` retrieves historical code
3. **Parent tracking**: The `parent_id` field can be set to any historical trial

This creates a disconnect: **selection is restricted, but mutation is not**.

---

## 3. Evidence: The Gen 4→5 Anomaly

### 3.1 What Happened in Generation 4

Generation 4 experienced a catastrophic failure rate:
- **15 trials spawned**
- **2 trials completed** (13% success rate)
- **13 trials timed out** (overly complex algorithms)

Only two trials finished:
- `trial_4_6`: 2.6185 (selected)
- `trial_4_14`: 2.5652 (not selected)

### 3.2 The Root LLM's Response

From the Gen 4 selection reasoning:
> *"Selected trial_4_6 as new best (2.6185, 99.37% of target). However, generation 4 had 73% failure rate due to overly complex algorithms that timed out. **Carrying forward 3 strong performers from generation 3 (trial_3_0, trial_3_5, trial_3_2) to maintain diversity.**"*

The Root LLM explicitly stated its intent to use historical trials, but the formal selection only recorded `trial_4_6`.

### 3.3 What Actually Happened in Generation 5

Analyzing the `parent_id` fields of Gen 5 trials reveals:

| Parent Used | Generation | Score | Formally Selected? |
|-------------|------------|-------|-------------------|
| `trial_4_6` | 4 | 2.6185 | **Yes** |
| `trial_3_0` | 3 | 2.6135 | No (historical) |
| `trial_3_2` | 3 | 2.6008 | No (historical) |
| `trial_3_5` | 3 | 2.6019 | No (historical) |

**4 distinct parents were used, but only 1 was formally selected.**

---

## 4. The Tracking Gap

### 4.1 Selected vs Actually Used

| Gen | Formally Selected | Actually Used as Parents | Discrepancy |
|-----|-------------------|--------------------------|-------------|
| 0 | 5 trials | 4 trials | 1 selected but unused |
| 1 | 5 trials | 5 trials | None |
| 2 | 5 trials | 4 trials | 1 selected but unused |
| 3 | 5 trials | 4 trials | 1 selected but unused |
| 4 | **1 trial** | **4 trials** | **3 historical parents untracked** |
| 5 | 5 trials | 5 trials | None |
| 6 | 4 trials | 3 trials | 1 selected but unused |
| 7 | 3 trials | 0 trials | Last generation |

### 4.2 Information Lost

The `experiment.json` record for Generation 4 shows:
```json
{
  "generation_num": 4,
  "selected_trial_ids": ["trial_4_6"],
  "selection_reasoning": "...Carrying forward 3 strong performers from generation 3..."
}
```

The reasoning text mentions the historical trials, but:
- They are not in `selected_trial_ids`
- There is no structured way to query which historical trials were used
- Post-hoc analysis requires parsing the reasoning text or reconstructing from `parent_id` fields

---

## 5. Parent Utilization Analysis

### 5.1 Most Used Parents (Across All Generations)

| Trial | Gen | Score | Children Spawned | Formally Selected? |
|-------|-----|-------|------------------|-------------------|
| trial_5_2 | 5 | 2.6243 | 8 | Yes (Gen 5) |
| trial_6_6 | 6 | 2.6276 | 7 | Yes (Gen 6) |
| **trial_3_0** | 3 | 2.6135 | **5** | **Only in Gen 3** |
| trial_1_6 | 1 | 2.6058 | 4 | Yes (Gen 1) |
| trial_2_8 | 2 | 2.6079 | 4 | Yes (Gen 2) |
| trial_4_6 | 4 | 2.6185 | 4 | Yes (Gen 4) |

**Key observation**: `trial_3_0` produced 5 children but was only formally selected once (in Gen 3). Its use in Gen 5 is not captured in the selection records.

### 5.2 Lineage of Best Trial

```
trial_2_8 (Gen 2, 2.6079)
    └─→ trial_3_2 (Gen 3, 2.6008)
            └─→ trial_5_5 (Gen 5, 2.6134)
                    └─→ trial_6_6 (Gen 6, 2.6276)
                            └─→ trial_7_6 (Gen 7, 2.6278) ← BEST
```

The winning lineage:
- Traces back to Generation 2
- Includes `trial_3_2`, which was used as a historical parent in Gen 5
- This cross-generation inheritance is not visible in selection records

---

## 6. The Stagnation Problem

### 6.1 Diminishing Returns

| Transition | Improvement | % Change |
|------------|-------------|----------|
| Gen 0→1 | +0.0489 | +1.91% |
| Gen 1→2 | +0.0021 | +0.08% |
| Gen 2→3 | +0.0056 | +0.21% |
| Gen 3→4 | +0.0050 | +0.19% |
| Gen 4→5 | +0.0058 | +0.22% |
| Gen 5→6 | +0.0033 | +0.13% |
| Gen 6→7 | **+0.0002** | **+0.01%** |

By Gen 7, improvements essentially stopped.

### 6.2 Root LLM's Diagnosis

From Gen 7 selection reasoning:
> *"CRITICAL ISSUE: Progress has essentially stalled - improvement dropped from +0.0033 (gen 5→6) to +0.0002 (gen 6→7), an 87% slowdown. Multiple trials converged to the same score (2.6276), suggesting we've hit a local optimum."*

### 6.3 Observable Patterns

1. **Convergence to same score**: Multiple Gen 7 trials achieved exactly 2.6276
2. **Limited diversity**: Same lineages being refined repeatedly
3. **No exploration of alternatives**: Earlier promising approaches were not revisited

---

## 7. The Workaround Pattern

### 7.1 How the Root LLM Bypasses Selection Limits

The Root LLM developed an informal pattern to use historical trials:

1. **In selection reasoning**: Explicitly mentions intent to use historical trials
2. **In child prompts**: Uses `{{CODE_TRIAL_X_Y}}` tokens to inject historical code
3. **Via API calls**: Uses `get_trial_code()` to retrieve historical implementations
4. **In parent_id**: Sets historical trial as the parent for lineage tracking

### 7.2 Costs of This Workaround

| Cost | Description |
|------|-------------|
| **Cognitive overhead** | Root LLM must remember to use workarounds |
| **Tracking gap** | Selection records don't reflect actual parent usage |
| **Extra API calls** | Needs explicit `get_trial_code()` calls |
| **Lost context** | Selection reasoning contains intent, but data is unstructured |
| **Analysis difficulty** | Post-hoc analysis requires reconstructing actual lineages |

### 7.3 Example: Extra Work Required

To use `trial_3_0` as a parent in Gen 5, the Root LLM had to:

```python
# 1. Explicitly fetch the code
code = get_trial_code(["trial_3_0"])

# 2. Or use token in prompt
spawn_child_llm(
    prompt="Improve on this approach: {{CODE_TRIAL_3_0}}",
    parent_id="trial_3_0"  # Manual lineage tracking
)
```

If the selection mechanism allowed historical trials, this would be:
```python
# Simply select trial_3_0 in the selection response
# The system handles code injection automatically
```

---

## 8. The Disconnect Between Selection and Mutation

### 8.1 Two Different Scopes

| Aspect | Selection Scope | Mutation Scope |
|--------|----------------|----------------|
| **Trials visible** | Current generation only | All generations |
| **Validation** | Filters historical trial_ids | Accepts any trial_id |
| **Tracking** | Recorded in `selected_trial_ids` | Recorded in `parent_id` |
| **UI presentation** | Only current gen trials | All-time top 5 in lineage map |

### 8.2 The Fundamental Question

The system currently answers two different questions:

1. **Selection asks**: "Which trials from this generation should inform the next?"
2. **Mutation allows**: "Which trials from all history can be used as a base?"

These questions have different scopes, but the selection UI only presents options for question #1 while the mutation mechanism operates on question #2.

---

## 9. Data Summary

### 9.1 Generation Details

```
Gen 0: 15 trials, 12 successful (80%), best=2.5569
Gen 1: 15 trials, 9 successful (60%), best=2.6058
Gen 2: 15 trials, 11 successful (73%), best=2.6079
Gen 3: 15 trials, 13 successful (87%), best=2.6135
Gen 4: 15 trials, 2 successful (13%), best=2.6185  ← CRITICAL FAILURE
Gen 5: 15 trials, 9 successful (60%), best=2.6243
Gen 6: 15 trials, 10 successful (67%), best=2.6276
Gen 7: 15 trials, 9 successful (60%), best=2.6278
```

### 9.2 All-Time Top 10 Trials

```
1. trial_7_6  (gen 7): 2.627825
2. trial_6_6  (gen 6): 2.627587
3. trial_7_3  (gen 7): 2.627587
4. trial_7_7  (gen 7): 2.627587
5. trial_7_10 (gen 7): 2.627587
6. trial_5_2  (gen 5): 2.624271
7. trial_6_8  (gen 6): 2.624271
8. trial_6_4  (gen 6): 2.621934
9. trial_7_5  (gen 7): 2.621851
10. trial_7_0 (gen 7): 2.620163
```

### 9.3 Historical Parent Usage (Untracked in Selection)

```
Gen 4 → Gen 5:
  Formally Selected: trial_4_6
  Actually Used:     trial_4_6, trial_3_0, trial_3_2, trial_3_5
  Untracked:         trial_3_0, trial_3_2, trial_3_5
```

---

## 10. Files Involved

| File | Relevant Code |
|------|---------------|
| `src/mango_evolve/evolution_api.py:668-774` | `_advance_generation()` - filters selections to current gen |
| `src/mango_evolve/evolution_api.py:82-104` | `TrialSelection` dataclass |
| `src/mango_evolve/root_llm.py:657-708` | `_build_selection_request_message()` - shows only current gen |
| `src/mango_evolve/root_llm.py:596-655` | `_build_generation_feedback_message()` |
| `src/mango_evolve/evolution_api.py:1010-1098` | `_build_lineage_map()` |

---

## 11. Open Questions

1. **Should selection scope match mutation scope?** The Root LLM can already use any historical trial for mutation, so why restrict what it can formally select?

2. **What is the purpose of selection?** If it's tracking intent, the current mechanism misses cross-generation usage. If it's restricting options, it's already bypassed.

3. **How should stagnation be addressed?** When improvements plateau, should the system encourage revisiting earlier approaches?

4. **What tracking is needed for analysis?** The gap between `selected_trial_ids` and actual `parent_id` usage makes post-hoc analysis difficult.

5. **Is the workaround acceptable?** The Root LLM successfully works around limitations, but at the cost of extra complexity and incomplete tracking.

---

*Document generated from experiment `circle_packing_v1_20251216_174624` analysis.*
