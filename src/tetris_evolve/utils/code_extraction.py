"""
Code extraction utilities for tetris_evolve.

Extracts Python code from LLM responses with markdown code blocks.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CodeBlock:
    """A code block extracted from text."""

    code: str
    language: Optional[str]
    start_pos: int
    end_pos: int


@dataclass
class ExtractionResult:
    """Result of extracting code from text."""

    code_blocks: List[CodeBlock]
    reasoning: str  # Text outside code blocks


def extract_code_blocks(
    text: str,
    languages: Optional[List[str]] = None,
) -> List[CodeBlock]:
    """
    Extract code blocks from markdown-formatted text.

    Handles both fenced code blocks (```python ... ```) and
    indented code blocks.

    Args:
        text: The text to extract code from
        languages: Optional list of language tags to accept (e.g., ["python", "repl"])
                  If None, accepts all language tags

    Returns:
        List of CodeBlock objects
    """
    blocks: List[CodeBlock] = []

    # Pattern for fenced code blocks with optional language
    # Matches: ```python ... ``` or ```repl ... ``` or ``` ... ```
    fenced_pattern = r"```(\w*)\s*\n(.*?)```"

    for match in re.finditer(fenced_pattern, text, re.DOTALL):
        language = match.group(1).lower() or None
        code = match.group(2)

        # Filter by language if specified
        if languages is not None:
            if language is None or language not in [lang.lower() for lang in languages]:
                continue

        blocks.append(
            CodeBlock(
                code=code.strip(),
                language=language,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    return blocks


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract the first Python code block from text.

    Accepts blocks marked as "python", "py", or unmarked.

    Args:
        text: The text to extract code from

    Returns:
        The code string, or None if no code block found
    """
    blocks = extract_code_blocks(text, languages=["python", "py", ""])

    if blocks:
        return blocks[0].code

    # Also try to find unmarked code blocks
    blocks = extract_code_blocks(text)
    for block in blocks:
        if block.language is None or block.language == "":
            return block.code

    return None


def extract_repl_code(text: str) -> Optional[str]:
    """
    Extract the first REPL code block from text.

    Specifically looks for blocks marked as "repl".

    Args:
        text: The text to extract code from

    Returns:
        The code string, or None if no code block found
    """
    blocks = extract_code_blocks(text, languages=["repl"])

    if blocks:
        return blocks[0].code

    return None


def extract_all_code(
    text: str,
    languages: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """
    Extract all code blocks and the remaining reasoning text.

    Args:
        text: The text to extract from
        languages: Optional list of language tags to accept

    Returns:
        Tuple of (list of code strings, reasoning text)
    """
    blocks = extract_code_blocks(text, languages)

    if not blocks:
        return [], text.strip()

    # Build reasoning by removing code blocks
    reasoning_parts = []
    last_end = 0

    for block in sorted(blocks, key=lambda b: b.start_pos):
        # Get text before this block
        before = text[last_end : block.start_pos].strip()
        if before:
            reasoning_parts.append(before)
        last_end = block.end_pos

    # Get text after last block
    after = text[last_end:].strip()
    if after:
        reasoning_parts.append(after)

    reasoning = "\n\n".join(reasoning_parts)
    code_strings = [block.code for block in blocks]

    return code_strings, reasoning


def extract_with_reasoning(text: str) -> ExtractionResult:
    """
    Extract all code blocks and reasoning from text.

    Accepts Python, REPL, and unmarked code blocks.

    Args:
        text: The text to extract from

    Returns:
        ExtractionResult with code blocks and reasoning
    """
    blocks = extract_code_blocks(text)

    # Build reasoning by removing code blocks
    reasoning_parts = []
    last_end = 0

    sorted_blocks = sorted(blocks, key=lambda b: b.start_pos)
    for block in sorted_blocks:
        before = text[last_end : block.start_pos].strip()
        if before:
            reasoning_parts.append(before)
        last_end = block.end_pos

    after = text[last_end:].strip()
    if after:
        reasoning_parts.append(after)

    reasoning = "\n\n".join(reasoning_parts)

    return ExtractionResult(
        code_blocks=blocks,
        reasoning=reasoning,
    )


def has_function(code: str, function_name: str) -> bool:
    """
    Check if code defines a specific function.

    Args:
        code: Python code to check
        function_name: Name of the function to look for

    Returns:
        True if function is defined, False otherwise
    """
    pattern = rf"^\s*def\s+{re.escape(function_name)}\s*\("
    return bool(re.search(pattern, code, re.MULTILINE))


def has_required_functions(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if code defines required circle packing functions.

    Either run_packing() or construct_packing() must be defined.

    Args:
        code: Python code to check

    Returns:
        Tuple of (has_required, error_message)
    """
    has_run = has_function(code, "run_packing")
    has_construct = has_function(code, "construct_packing")

    if has_run or has_construct:
        return True, None

    return False, "Code must define run_packing() or construct_packing()"
