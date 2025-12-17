# Context-Size Incident Report – circle_packing_v1_20251209_151839

## Overview
- The run likely failed after generation 4 due to context overflow when calling the root LLM. The conversation log reached ~762k characters (~190k tokens) by turn 30, far beyond typical model limits.
- Root cause: the orchestrator injected every trial’s full code into the feedback message each generation, then resent that ever-growing history without pruning. Prompt caching was ineffective because the cached portion excluded the large feedback blocks.
- Secondary issues: many child trials hit the 300s timeout, inflating logs with unsuccessful runs and wasting budget. The run did not finish cleanly (no `experiment.json` was written).

## Evidence
- Log growth (experiments/circle_packing_v1_20251209_151839/root_llm_log.jsonl):
  - Gen0 feedback user message: 85,625 chars (~21k tokens).
  - Gen1: 138,325 chars (~34.6k tokens).
  - Gen2: 130,529 chars (~32.6k tokens).
  - Gen3: 171,850 chars (~43.0k tokens).
  - Gen4: 141,234 chars (~35.3k tokens).
  - Total conversation by turn 30: ~762k chars (~190k tokens).
- Source of bloat: `src/tetris_evolve/root_llm.py:_build_generation_feedback_message` appends `Code: {trial.code}` for every trial to the feedback user message each generation.
- Caching mismatch: `_prepare_messages_with_caching` caches the second-to-last user message (the small selection request), not the large feedback blobs, so cache hits are minimal while the heavy messages are resent.
- Missing completion artifact: no `experiment.json` in the run directory, suggesting the run halted before `save_experiment` executed (likely due to the context blow-up).
- Timeouts: many trials report `Timeout after 300s`, adding to log size without value.

## Root Cause
- Unbounded inclusion of full trial code in every generation’s feedback.
- No pruning of conversation history before root LLM calls.
- Cache applied to the wrong segment of the conversation (cached small selection prompts instead of large recurring content).
- Long per-trial timeouts keep producing verbose failure records.

## Impact
- Root LLM call after generation 4 likely exceeded context, halting the run.
- Wasted budget and wall-clock on timeouts and oversized prompts.
- No final experiment summary written (missing `experiment.json`).

## Prompt-Caching / Prompt-Shaping Opportunities
- Send compact feedback: scores, validity, parent, short reasoning, top-N only; direct the root LLM to fetch code via `{{CODE_TRIAL_X_Y}}` instead of inlining code.
- Cache the static selection instructions and other stable text; mark them with `cache_control` and keep dynamic trial tables small.
- Prune history before each root call (keep only the latest feedback + selection, or rebuild from persisted summaries).
- Add a token budget guard to auto-summarize when near the limit.

## Remediations
1) Rewrite `_build_generation_feedback_message` to exclude code, cap to top-N trials, and include only compact fields (id, sum_radii, valid, parent_id, brief reasoning).  
2) Prune `self.messages` before each root call (e.g., keep only the latest feedback + selection exchange) or rebuild from disk summaries to avoid resending old generations.  
3) Adjust caching: cache the static selection instructions; do not cache the large dynamic feedback; ensure the cached block is the one that recurs.  
4) Add a lightweight token estimator before `root_llm.generate`; if near a threshold, drop older turns or switch to an ultra-compact summary.  
5) Reduce child timeout or tighten child prompts to avoid 300s stalls and noisy failure logs.  
6) After code changes, rerun a short experiment to confirm the root call proceeds past gen4 and `experiment.json` is written.  
