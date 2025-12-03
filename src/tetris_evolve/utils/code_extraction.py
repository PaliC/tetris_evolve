"""
Code extraction utilities for tetris_evolve.

Extracts REPL code blocks from LLM responses. The Root LLM uses ```repl```
code blocks to indicate Python code that should be executed in the REPL.
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


def extract_code_blocks(
    text: str,
    language: str = "repl",
) -> List[CodeBlock]:
    """
    Extract code blocks with a specific language tag from markdown-formatted text.

    Args:
        text: The text to extract code from
        language: The language tag to look for (default: "repl")

    Returns:
        List of CodeBlock objects
    """
    blocks: List[CodeBlock] = []

    # Pattern for fenced code blocks: ```language ... ```
    fenced_pattern = r"```(\w*)\s*\n(.*?)```"

    for match in re.finditer(fenced_pattern, text, re.DOTALL):
        block_language = match.group(1).lower() or None
        code = match.group(2)

        # Only match the specified language
        if block_language != language.lower():
            continue

        blocks.append(
            CodeBlock(
                code=code.strip(),
                language=block_language,
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )

    return blocks


def extract_repl_blocks(text: str) -> List[str]:
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


def extract_reasoning(text: str) -> str:
    """
    Extract reasoning text (everything outside code blocks).

    Args:
        text: The text to extract reasoning from

    Returns:
        Text with all code blocks removed
    """
    # Remove all fenced code blocks
    pattern = r"```\w*\s*\n.*?```"
    reasoning = re.sub(pattern, "", text, flags=re.DOTALL)

    # Clean up extra whitespace
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)

    return reasoning.strip()
