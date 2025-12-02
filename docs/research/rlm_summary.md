# Recursive Language Models (RLM) Technical Summary

## Overview

Recursive Language Models (RLMs) are a framework that wraps language models to enable recursive self-calls for intermediate computation. The key insight is adopting a **context-centric view**: rather than decomposing problems (like Chain-of-Thought), RLMs decompose and manage context.

## Core Concept

```
Traditional LLM:
┌─────────────────────────────────────────┐
│  Full Context + Query → LLM → Answer    │
│  (context window limited)               │
└─────────────────────────────────────────┘

RLM Approach:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Query → Root LLM (no full context) → REPL Environment          │
│              │                              │                   │
│              │  spawn_child(q', C')         │                   │
│              ▼                              ▼                   │
│         Child LLM ◄───── Context Subset ────┤                   │
│              │                              │                   │
│              └────── Result ───────────────►│                   │
│                                             │                   │
│  Root LLM aggregates results → FINAL(answer)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Context as Memory, Not Input

- The full context is NOT passed to the root LLM
- Context is pre-loaded as a variable in the REPL environment
- The LLM accesses context programmatically through the REPL

### 2. LLM-Driven Decomposition

> "Agents are designed based on human/expert intuition on how to break down a problem... RLMs are designed based on the principle that fundamentally, LMs should decide how to break down a problem"

The LLM itself decides:
- How to partition the context
- Which subsets to examine
- How to aggregate results

### 3. API Preservation

```python
# User-facing API remains identical to standard LLM calls
response = rlm.completion(messages)

# Internally, the RLM manages:
# - REPL environment setup
# - Context variable loading
# - Recursive call orchestration
# - Result aggregation
```

## Architecture

### REPL Environment

The REPL (Read-Eval-Print Loop) serves as the execution context:

```python
class REPLEnvironment:
    """
    Execution environment for RLM.
    - Maintains state across LLM interactions
    - Executes code blocks output by the LLM
    - Manages context as accessible variables
    - Handles sub-LLM spawning
    """

    def __init__(self, context):
        self.variables = {'context': context}
        self.output_buffer = []

    def execute(self, code: str) -> str:
        """Execute Python code, return truncated output"""
        # Run code in sandboxed environment
        # Capture stdout/stderr
        # Return results to LLM
        pass

    def spawn_sub_llm(self, query: str, context_subset) -> str:
        """Spawn a child LLM with subset of context"""
        pass
```

### Root LLM Interaction

The root LLM interacts via code blocks:

```
User Query: "Find the secret number hidden in this 1M line file"

Root LLM Output:
```python
# First, check the structure of the context
print(len(context.split('\n')))  # See how many lines
print(context[:500])  # Peek at beginning
```

REPL Returns: "1000000 lines\nRandom text line 1..."

Root LLM Output:
```python
# Search for patterns that might indicate the secret
import re
matches = re.findall(r'SECRET:\s*(\d+)', context)
print(matches)
```

REPL Returns: "['42']"

Root LLM Output:
FINAL(42)
```

### Sub-LLM Spawning

```python
def spawn_sub_llm(query: str, context_subset: str) -> str:
    """
    Spawn an isolated child LLM instance.

    Args:
        query: New query for the child (q̂)
        context_subset: Transformed/filtered context (Ĉ)

    Returns:
        Child's final answer

    Note: Child has isolated environment, cannot access parent state
    """
    child = SubLLM(model=self.model)
    child_env = REPLEnvironment(context_subset)
    return child.run(query, child_env)
```

### Recursion Depth

Current implementations limit to `depth=1`:
- Root LLM can spawn Child LLMs
- Child LLMs cannot spawn their own children
- Enabling deeper recursion requires replacing `Sub_RLM` with `RLM_REPL`

```
Depth 0: Root LLM (has REPL, can spawn children)
    │
    ├── Depth 1: Child LLM (isolated, cannot spawn)
    ├── Depth 1: Child LLM (isolated, cannot spawn)
    └── Depth 1: Child LLM (isolated, cannot spawn)
```

## Emergent Interaction Patterns

### 1. Peeking
```python
# LLM examines initial segments to understand structure
print(context[:1000])
print(context[-1000:])
```

### 2. Grepping
```python
# Programmatic search instead of semantic retrieval
lines = [l for l in context.split('\n') if 'ERROR' in l]
print(lines[:10])
```

### 3. Partition + Map
```python
# Split context, process chunks with child LLMs
chunks = split_into_chunks(context, size=10000)
results = []
for i, chunk in enumerate(chunks):
    result = spawn_sub_llm(
        query=f"Find errors in chunk {i}",
        context_subset=chunk
    )
    results.append(result)
```

### 4. Summarization
```python
# Compress information for decision-making
summaries = []
for section in context.split('---'):
    summary = spawn_sub_llm(
        query="Summarize the key points",
        context_subset=section
    )
    summaries.append(summary)

combined = '\n'.join(summaries)
```

## Output Mechanisms

### Direct Answer
```python
FINAL(42)  # Return literal value
```

### Variable Reference
```python
result = complex_computation()
FINAL_VAR(result)  # Return value of variable
```

## Comparison with Other Approaches

| Approach | Context Handling | Decomposition | Decision Maker |
|----------|------------------|---------------|----------------|
| Standard LLM | Full context in window | None | N/A |
| RAG | Semantic retrieval | Fixed chunking | Retriever |
| Chain-of-Thought | Full context | Problem steps | Human design |
| Agents | Full context | Tool selection | Human design |
| Tree-of-Thought | Full context | Branching paths | Human design |
| **RLM** | **Variable in REPL** | **LLM-driven** | **LLM itself** |

## Limitations

1. **No Cost Guarantees**: "The system currently lacks strong guarantees about controlling either the total API cost or the total runtime of each call"

2. **Blocking Calls**: Recursive calls are synchronous without prefix caching

3. **Depth Limitation**: Current implementation limited to depth=1

4. **Unoptimized**: "We did not optimize our implementation of RLMs for speed"

## Relevance to Our Project

For our LLM Tetris optimizer, RLM concepts apply to:

| RLM Concept | Our Application |
|-------------|-----------------|
| Root LLM | Orchestrator that decides evolutionary direction |
| REPL Environment | Shared execution context with REPL functions |
| Sub-LLM Spawning | Child LLMs that generate Tetris-playing code |
| Context Management | Experiment/generation/trial history as context |
| LLM-Driven Decisions | Root LLM decides which trials advance |

**Key Adaptation**: Instead of context decomposition, we use the REPL pattern for the Root LLM to:
1. Spawn child LLMs to generate code
2. Evaluate results
3. Decide which trials to advance
4. Terminate when satisfied

The REPL functions (`spawn_child_llm`, `evaluate_program`, `advance_generation`, `terminate_evolution`) are pre-loaded, and the Root LLM orchestrates the evolutionary process through code execution.

## References

- [RLM Blog Post](https://alexzhang13.github.io/blog/2025/rlm/)
- [RLM GitHub Repository](https://github.com/alexzhang13/rlm)
