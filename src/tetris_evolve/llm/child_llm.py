"""Child LLM executor for generating Tetris-playing code."""

import re
from dataclasses import dataclass
from typing import Optional

from .client import LLMClient, LLMResponse, LLMClientError


@dataclass
class ChildResult:
    """Result from child LLM code generation."""

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
- [200]: tick_progress (0-1)
- [201]: fall_timer (0-1)
- [202]: piece_row (0-1, multiply by 20 for actual row)
- [203]: piece_col (0-1, multiply by 10 for actual column)
- [204]: rotation (0-1)
- [205]: can_hold (0 or 1)
- [206:213]: Current piece one-hot (7 values)
- [213:220]: Next piece one-hot
- [220:227]: Next-next piece one-hot
- [227:234]: Hold piece one-hot (zeros if empty)
- [234:244]: Noise bits (ignore)

## Piece Types (index order)
0=O (square), 1=I (line), 2=S, 3=Z, 4=T, 5=L, 6=J

## Required Interface
Your code MUST implement exactly this function:
```python
def choose_action(obs):
    # obs: numpy array of shape (244,)
    # Return: integer action 0-6
    return action
```

## Output Format
First explain your reasoning in <reasoning> tags.
Then provide your complete code in <code> tags.

Example:
<reasoning>
My strategy is to...
</reasoning>

<code>
def choose_action(obs):
    # Your implementation
    return 5
</code>

IMPORTANT:
- Your code must be syntactically valid Python
- You can import numpy and random only
- The choose_action function must return an integer 0-6
- Keep your code simple and focused
"""

HELPER_CODE = '''
import numpy as np

def parse_observation(obs):
    """Parse PufferLib observation into readable components."""
    return {
        "board": obs[0:200].reshape(20, 10),
        "tick_progress": obs[200],
        "fall_timer": obs[201],
        "piece_row": int(obs[202] * 20),
        "piece_col": int(obs[203] * 10),
        "rotation": int(obs[204] * 4),
        "can_hold": bool(obs[205] > 0.5),
        "current_piece": int(np.argmax(obs[206:213])),
        "next_piece": int(np.argmax(obs[213:220])),
        "next_next_piece": int(np.argmax(obs[220:227])),
        "hold_piece": int(np.argmax(obs[227:234])) if obs[227:234].sum() > 0 else None,
    }

PIECES = ["O", "I", "S", "Z", "T", "L", "J"]
'''


class ChildLLMExecutor:
    """Generates Tetris-playing code from prompts."""

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize child LLM executor.

        Args:
            llm_client: LLM client for API calls.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens in response.
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        parent_code: Optional[str] = None,
    ) -> ChildResult:
        """Generate code based on prompt from root LLM.

        Args:
            prompt: Task description from root LLM.
            parent_code: Optional parent code to improve upon.

        Returns:
            ChildResult with generated code and metadata.
        """
        try:
            # Build the full prompt
            user_prompt = self._build_user_prompt(prompt, parent_code)

            # Call LLM
            response = self.llm_client.send_message(
                messages=[{"role": "user", "content": user_prompt}],
                system=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse response
            code = self._extract_code(response.content)
            reasoning = self._extract_reasoning(response.content)

            if not code:
                return ChildResult(
                    code="",
                    reasoning=reasoning,
                    raw_response=response.content,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    success=False,
                    error="No code found in response",
                )

            # Clean the code
            code = self._clean_code(code)

            return ChildResult(
                code=code,
                reasoning=reasoning,
                raw_response=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                success=True,
                error=None,
            )

        except LLMClientError as e:
            return ChildResult(
                code="",
                reasoning="",
                raw_response="",
                input_tokens=0,
                output_tokens=0,
                success=False,
                error=f"LLM error: {e}",
            )
        except Exception as e:
            return ChildResult(
                code="",
                reasoning="",
                raw_response="",
                input_tokens=0,
                output_tokens=0,
                success=False,
                error=f"Unexpected error: {e}",
            )

    def generate_and_validate(
        self,
        prompt: str,
        parent_code: Optional[str] = None,
        evaluator=None,
        max_retries: int = 2,
    ) -> ChildResult:
        """Generate and validate code syntax, with optional retry.

        Args:
            prompt: Task description from root LLM.
            parent_code: Optional parent code to improve upon.
            evaluator: ProgramEvaluator for syntax validation.
            max_retries: Maximum retry attempts for invalid code.

        Returns:
            ChildResult with generated and validated code.
        """
        last_result = None
        last_error = None

        for attempt in range(max_retries + 1):
            # Build prompt with error feedback if this is a retry
            if attempt > 0 and last_error:
                retry_prompt = f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease fix the error and try again."
            else:
                retry_prompt = prompt

            # Generate code
            result = self.generate(retry_prompt, parent_code)
            last_result = result

            if not result.success:
                last_error = result.error
                continue

            # Validate syntax if evaluator provided
            if evaluator is not None:
                valid, error = evaluator._validate_syntax(result.code)
                if not valid:
                    last_error = f"Syntax error: {error}"
                    result = ChildResult(
                        code=result.code,
                        reasoning=result.reasoning,
                        raw_response=result.raw_response,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        success=False,
                        error=last_error,
                    )
                    last_result = result
                    continue

                # Also check safety
                safe, error = evaluator._check_safety(result.code)
                if not safe:
                    last_error = f"Safety violation: {error}"
                    result = ChildResult(
                        code=result.code,
                        reasoning=result.reasoning,
                        raw_response=result.raw_response,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        success=False,
                        error=last_error,
                    )
                    last_result = result
                    continue

            # Success
            return result

        # Return last result (failed after all retries)
        return last_result

    def generate_batch(
        self,
        prompts: list[tuple[str, Optional[str]]],
    ) -> list[ChildResult]:
        """Generate multiple codes in sequence.

        Args:
            prompts: List of (prompt, parent_code) tuples.

        Returns:
            List of ChildResults.
        """
        results = []
        for prompt, parent_code in prompts:
            result = self.generate(prompt, parent_code)
            results.append(result)
        return results

    def _build_user_prompt(
        self,
        task_prompt: str,
        parent_code: Optional[str],
    ) -> str:
        """Build the full user prompt.

        Args:
            task_prompt: Task description from root LLM.
            parent_code: Optional parent code to improve upon.

        Returns:
            Complete user prompt string.
        """
        parts = ["## Task\n", task_prompt, "\n"]

        if parent_code:
            parts.extend([
                "\n## Parent Code (to improve upon)\n",
                "```python\n",
                parent_code,
                "\n```\n",
                "\nImprove this code based on the task above.\n",
            ])
        else:
            parts.append("\nCreate a new Tetris player from scratch.\n")

        parts.extend([
            "\n## Helper Code Available\n",
            "You can use this helper function:\n",
            "```python\n",
            HELPER_CODE,
            "```\n",
        ])

        return "".join(parts)

    def _extract_code(self, response: str) -> str:
        """Extract code from response between <code> tags.

        Args:
            response: Raw LLM response.

        Returns:
            Extracted code, or empty string if not found.
        """
        # Try <code> tags first
        match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try markdown code blocks as fallback
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response between <reasoning> tags.

        Args:
            response: Raw LLM response.

        Returns:
            Extracted reasoning, or empty string if not found.
        """
        match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _clean_code(self, code: str) -> str:
        """Clean extracted code.

        Args:
            code: Raw extracted code.

        Returns:
            Cleaned code.
        """
        # Remove markdown code fences if present
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"```\s*$", "", code)

        # Strip whitespace
        code = code.strip()

        return code
