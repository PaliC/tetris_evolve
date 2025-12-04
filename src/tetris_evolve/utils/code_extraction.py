"""
Code extraction utilities for tetris_evolve.

Extracts REPL code blocks from LLM responses. The Root LLM uses ```repl```
code blocks to indicate Python code that should be executed in the REPL.
"""

import re
from dataclasses import dataclass


@dataclass
class CodeBlock:
    """A code block extracted from text."""

    code: str
    language: str | None
    start_pos: int
    end_pos: int


def _find_matching_fence(text: str, start_pos: int) -> int | None:
    """
    Find the closing ``` fence that matches the opening fence at start_pos.

    Handles nested code blocks by counting fence pairs. An opening fence
    is ``` followed by a language tag (```python\n). A closing fence is
    ``` followed by only whitespace/newline.

    Args:
        text: The full text
        start_pos: Position right after the opening ``` and language tag

    Returns:
        Position of the matching closing ```, or None if not found
    """
    pos = start_pos
    nesting_depth = 1

    while pos < len(text):
        # Find next ``` marker
        next_fence = text.find("```", pos)
        if next_fence == -1:
            return None

        # Check if this fence is at the start of a line
        # (either at position 0 or preceded by a newline)
        is_line_start = next_fence == 0 or text[next_fence - 1] == "\n"

        if is_line_start:
            after_fence = next_fence + 3

            # Determine if this is an opening fence or closing fence
            # Opening fence: ```language\n (must have alphanumeric language tag)
            # Closing fence: ``` followed by end of line or EOF
            is_opening = False

            if after_fence < len(text):
                rest = text[after_fence:]
                if rest and rest[0].isalpha():
                    # Has a language tag - check if it's valid
                    newline_pos = rest.find("\n")
                    if newline_pos != -1:
                        lang_part = rest[:newline_pos].strip()
                        # Valid opening fence has alphanumeric language tag
                        if lang_part and lang_part.isalnum():
                            is_opening = True

            if is_opening:
                nesting_depth += 1
                pos = after_fence
                continue

            # This is a closing fence
            nesting_depth -= 1
            if nesting_depth == 0:
                return next_fence

        pos = next_fence + 3

    return None


def extract_code_blocks(
    text: str,
    language: str = "repl",
) -> list[CodeBlock]:
    """
    Extract code blocks with a specific language tag from markdown-formatted text.

    Handles nested code blocks correctly by finding matching fence pairs.

    Args:
        text: The text to extract code from
        language: The language tag to look for (default: "repl")

    Returns:
        List of CodeBlock objects
    """
    blocks: list[CodeBlock] = []

    # Pattern to find opening fences: ```language at start of line
    # We'll manually find the matching closing fence
    opening_pattern = rf"^```({re.escape(language)})\s*\n"

    pos = 0
    while pos < len(text):
        # Search for opening fence from current position
        match = re.search(opening_pattern, text[pos:], re.MULTILINE | re.IGNORECASE)
        if not match:
            break

        # Calculate absolute positions
        fence_start = pos + match.start()
        content_start = pos + match.end()
        block_language = match.group(1).lower()

        # Find the matching closing fence
        closing_pos = _find_matching_fence(text, content_start)

        if closing_pos is None:
            # No matching close found, skip this opening
            pos = content_start
            continue

        # Extract the code content
        code = text[content_start:closing_pos]

        blocks.append(
            CodeBlock(
                code=code.strip(),
                language=block_language,
                start_pos=fence_start,
                end_pos=closing_pos + 3,  # Include closing ```
            )
        )

        # Continue searching after this block
        pos = closing_pos + 3

    return blocks


def extract_repl_blocks(text: str) -> list[str]:
    """
    Extract all REPL code blocks from text.

    The Root LLM writes Python code in ```repl``` blocks that should
    be executed in the REPL environment.

    Args:
        text: The text to extract code from

    Returns:
        List of code strings (one per ```repl``` block)
    """
    blocks = extract_code_blocks(text, language="repl")
    return [block.code for block in blocks]


def _find_all_top_level_blocks(text: str) -> list[tuple[int, int]]:
    """
    Find all top-level code blocks (any language) with their positions.

    Returns list of (start, end) positions for each block.
    """
    blocks = []
    # Pattern to find any opening fence at start of line
    opening_pattern = r"^```(\w*)\s*\n"

    pos = 0
    while pos < len(text):
        match = re.search(opening_pattern, text[pos:], re.MULTILINE)
        if not match:
            break

        fence_start = pos + match.start()
        content_start = pos + match.end()

        closing_pos = _find_matching_fence(text, content_start)

        if closing_pos is None:
            pos = content_start
            continue

        blocks.append((fence_start, closing_pos + 3))
        pos = closing_pos + 3

    return blocks


def extract_reasoning(text: str) -> str:
    """
    Extract reasoning text (everything outside code blocks).

    Handles nested code blocks correctly.

    Args:
        text: The text to extract reasoning from

    Returns:
        Text with all code blocks removed
    """
    # Find all top-level blocks and remove them
    blocks = _find_all_top_level_blocks(text)

    # Remove blocks from end to start to preserve positions
    result = text
    for start, end in reversed(blocks):
        result = result[:start] + result[end:]

    # Clean up extra whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def extract_python_code(text: str) -> str | None:
    """
    Extract Python code from LLM response.

    Looks for ```python code blocks first, falls back to unlabeled blocks.

    Args:
        text: LLM response text

    Returns:
        Extracted code or None if not found
    """
    # Try python blocks first
    blocks = extract_code_blocks(text, language="python")
    if blocks:
        return blocks[0].code

    # Fallback: try unlabeled code blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    return None
