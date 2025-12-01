# Component 5: Child LLM Executor

## Overview
Generates Tetris-playing code from prompts provided by the root LLM.

**File**: `src/tetris_evolve/llm/child_llm.py`
**Test File**: `tests/test_child_llm.py`
**Dependencies**:
- Component 1: Program Evaluator
- Component 4: LLM Client

---

## Checklist

### 5.1 Project Setup
- [x] **5.1.1** Create `child_llm.py` in `src/tetris_evolve/llm/`
  - Dependencies: 4.1.1 (llm folder exists)
- [x] **5.1.2** Create `tests/test_child_llm.py` skeleton
  - Dependencies: 5.1.1

### 5.2 Data Classes
- [x] **5.2.1** Define `ChildLLMResult` dataclass
  - Dependencies: 5.1.1
  ```python
  @dataclass
  class ChildLLMResult:
      code: str
      reasoning: str
      raw_response: str
      input_tokens: int
      output_tokens: int
      success: bool
      error: Optional[str]
  ```
- [x] **5.2.2** Write tests for `ChildLLMResult` creation
  - Dependencies: 5.2.1

### 5.3 Prompt Templates
- [x] **5.3.1** Define `SYSTEM_PROMPT` constant
  - Dependencies: 5.1.1
  - Tetris environment description
  - Observation format (PufferLib 244-dim)
  - Action space (7 actions)
  - Required interface (`choose_action`)
- [x] **5.3.2** Define `HELPER_CODE` constant
  - Dependencies: 5.1.1
  - `parse_observation()` helper function
  - Piece names constant
- [x] **5.3.3** Implement `_build_user_prompt(task_prompt, parent_code) -> str`
  - Dependencies: 5.3.1
  - Combines task from root with optional parent code
  - Specifies output format (`<reasoning>`, `<code>`)
- [x] **5.3.4** Write tests for prompt building
  - Dependencies: 5.3.3

### 5.4 Response Parsing
- [x] **5.4.1** Implement `_extract_code(response: str) -> str`
  - Dependencies: 5.1.1
  - Extract content between `<code>` and `</code>` tags
  - Handle missing tags gracefully
- [x] **5.4.2** Write tests for `_extract_code`
  - Dependencies: 5.4.1
  - Test: valid tags, missing tags, multiple code blocks
- [x] **5.4.3** Implement `_extract_reasoning(response: str) -> str`
  - Dependencies: 5.1.1
  - Extract content between `<reasoning>` and `</reasoning>` tags
  - Return empty string if not found
- [x] **5.4.4** Write tests for `_extract_reasoning`
  - Dependencies: 5.4.3
- [x] **5.4.5** Implement `_clean_code(code: str) -> str`
  - Dependencies: 5.4.1
  - Remove markdown code fences if present
  - Strip leading/trailing whitespace
  - Normalize indentation
- [x] **5.4.6** Write tests for `_clean_code`
  - Dependencies: 5.4.5

### 5.5 ChildLLMExecutor Class
- [x] **5.5.1** Implement `ChildLLMExecutor.__init__(llm_client, temperature)`
  - Dependencies: 5.1.1, 4.4.1
  - Store LLM client reference
  - Store generation parameters
- [x] **5.5.2** Implement `generate(prompt, parent_code) -> ChildLLMResult`
  - Dependencies: 5.2.1, 5.3.3, 5.4.1, 5.4.3, 5.5.1
  - Build full prompt
  - Call LLM
  - Parse response
  - Return structured result
- [x] **5.5.3** Write tests for `generate` with mocked LLM
  - Dependencies: 5.5.2
- [x] **5.5.4** Implement error handling for LLM failures
  - Dependencies: 5.5.2
  - Handle: API errors, parsing failures, empty responses
- [x] **5.5.5** Write tests for error handling
  - Dependencies: 5.5.4

### 5.6 Code Validation Integration
- [x] **5.6.1** Implement `generate_and_validate(prompt, parent_code, evaluator) -> ChildLLMResult`
  - Dependencies: 5.5.2, 1.3.1 (syntax validation)
  - Generate code
  - Validate syntax
  - Update result with validation status
- [x] **5.6.2** Write tests for `generate_and_validate`
  - Dependencies: 5.6.1
- [x] **5.6.3** Implement retry on invalid code (optional)
  - Dependencies: 5.6.1
  - If syntax invalid, retry with error feedback
  - Max 2 retries
- [x] **5.6.4** Write tests for retry logic
  - Dependencies: 5.6.3

### 5.7 Batch Generation
- [x] **5.7.1** Implement `generate_batch(prompts: list[tuple[str, str]]) -> list[ChildLLMResult]`
  - Dependencies: 5.5.2
  - Generate multiple codes in sequence
  - Could be parallelized later
- [x] **5.7.2** Write tests for `generate_batch`
  - Dependencies: 5.7.1

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_child_result_creation` | ChildLLMResult dataclass works | 5.2.1 |
| `test_build_prompt_no_parent` | Prompt built without parent | 5.3.3 |
| `test_build_prompt_with_parent` | Prompt includes parent code | 5.3.3 |
| `test_extract_code_valid` | Extracts code from tags | 5.4.1 |
| `test_extract_code_no_tags` | Handles missing tags | 5.4.1 |
| `test_extract_reasoning` | Extracts reasoning | 5.4.3 |
| `test_clean_code_markdown` | Removes markdown fences | 5.4.5 |
| `test_generate_success` | Full generation works | 5.5.2 |
| `test_generate_llm_error` | Handles LLM errors | 5.5.4 |
| `test_generate_parse_error` | Handles parse errors | 5.5.4 |
| `test_generate_and_validate_valid` | Valid code passes | 5.6.1 |
| `test_generate_and_validate_invalid` | Invalid code fails | 5.6.1 |
| `test_retry_on_invalid` | Retries generate valid code | 5.6.3 |
| `test_generate_batch` | Batch generation works | 5.7.1 |

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/llm/child_llm.py

import re
from dataclasses import dataclass
from typing import Optional
from .client import LLMClient, LLMResponse

@dataclass
class ChildLLMResult:
    code: str
    reasoning: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    success: bool
    error: Optional[str]

SYSTEM_PROMPT = """You are generating a Tetris-playing program using PufferLib.

## Environment
- Board: 20 rows × 10 columns
- Observation: numpy array of shape (244,)
- Actions: 0=no-op, 1=left, 2=right, 3=rotate, 4=soft_drop, 5=hard_drop, 6=hold

## Observation Format
- [0:200]: Board state (20×10 flattened), 0=empty, 1=placed, 2=current piece
- [200-205]: tick, fall_timer, row, col, rotation, can_hold
- [206-233]: One-hot encoded pieces (current, next, next2, hold)
- [234-243]: Noise bits (ignore)

## Required Interface
Your code must implement:
```python
def choose_action(observation: np.ndarray) -> int:
    # Return action 0-6
```

## Output Format
Provide reasoning in <reasoning> tags, then code in <code> tags.
"""

HELPER_CODE = '''
import numpy as np

def parse_observation(obs: np.ndarray) -> dict:
    """Parse PufferLib observation into readable components."""
    return {
        "board": obs[0:200].reshape(20, 10),
        "tick_progress": obs[200],
        "fall_timer": obs[201],
        "piece_row": int(obs[202] * 20),
        "piece_col": int(obs[203] * 10),
        "rotation": int(obs[204]),
        "can_hold": bool(obs[205]),
        "current_piece": int(np.argmax(obs[206:213])),
        "next_pieces": [int(np.argmax(obs[213+i*7:220+i*7])) for i in range(2)],
        "hold_piece": int(np.argmax(obs[227:234])) if obs[227:234].sum() > 0 else None,
    }

PIECES = ["O", "I", "S", "Z", "T", "L", "J"]
'''

class ChildLLMExecutor:
    def __init__(self, llm_client: LLMClient, temperature: float = 0.8):
        self.llm_client = llm_client
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        parent_code: Optional[str] = None
    ) -> ChildLLMResult:
        """Generate code based on prompt from root LLM."""
        pass

    def generate_and_validate(
        self,
        prompt: str,
        parent_code: Optional[str] = None,
        evaluator = None
    ) -> ChildLLMResult:
        """Generate and validate code syntax."""
        pass

    def generate_batch(
        self,
        prompts: list[tuple[str, Optional[str]]]
    ) -> list[ChildLLMResult]:
        """Generate multiple codes."""
        pass

    def _build_user_prompt(self, task_prompt: str, parent_code: Optional[str]) -> str:
        pass

    def _extract_code(self, response: str) -> str:
        pass

    def _extract_reasoning(self, response: str) -> str:
        pass

    def _clean_code(self, code: str) -> str:
        pass
```

---

## Example LLM Interaction

**User Prompt:**
```
## Task
Focus on reducing holes while maintaining line clear rate.

## Parent Code
```python
def choose_action(obs):
    return 5  # Always hard drop
```

Improve this code to be smarter about piece placement.
```

**Expected Response:**
```
<reasoning>
The parent code just hard drops immediately, which creates many holes.
I'll improve it by:
1. Checking possible landing positions
2. Evaluating each position for holes created
3. Moving to the best position before dropping
</reasoning>

<code>
import numpy as np

def choose_action(observation):
    board = observation[0:200].reshape(20, 10)
    col = int(observation[203] * 10)

    # Simple hole avoidance: prefer columns with fewer holes below
    best_col = col
    min_holes = float('inf')

    for target_col in range(10):
        holes = count_holes_if_drop(board, target_col)
        if holes < min_holes:
            min_holes = holes
            best_col = target_col

    if col < best_col:
        return 2  # move right
    elif col > best_col:
        return 1  # move left
    else:
        return 5  # hard drop

def count_holes_if_drop(board, col):
    # Count holes in column
    holes = 0
    found_block = False
    for row in range(20):
        if board[row, col] > 0:
            found_block = True
        elif found_block:
            holes += 1
    return holes
</code>
```

---

## Acceptance Criteria

- [x] All 14 tests pass
- [x] Code coverage > 90%
- [x] Can generate valid Tetris player code
- [x] Properly extracts code and reasoning from responses
- [x] Handles LLM errors gracefully
- [x] Retry logic works for invalid code
