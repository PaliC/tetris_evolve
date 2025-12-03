"""
Tests for tetris_evolve.utils.code_extraction module.
"""

import pytest

from tetris_evolve.utils import (
    extract_code_blocks,
    extract_repl_blocks,
    extract_reasoning,
    CodeBlock,
)


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function."""

    def test_extract_repl_block(self):
        """Extract code from ```repl``` block."""
        text = """
Here is some code:

```repl
x = spawn_child_llm("test")
print(x)
```

That's all.
"""
        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 1
        assert "spawn_child_llm" in blocks[0].code
        assert blocks[0].language == "repl"

    def test_multiple_repl_blocks(self):
        """Handle multiple repl code blocks."""
        text = """
```repl
x = 1
```

Some text

```repl
y = 2
```
"""
        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 2
        assert "x = 1" in blocks[0].code
        assert "y = 2" in blocks[1].code

    def test_no_code_block(self):
        """Return empty list when no repl blocks found."""
        text = "This is just plain text with no code."

        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 0

    def test_malformed_block(self):
        """Handle edge cases gracefully."""
        # Unclosed block
        text = """
```repl
x = 5
"""
        blocks = extract_code_blocks(text, language="repl")
        # Should not match unclosed blocks
        assert len(blocks) == 0

    def test_ignores_other_languages(self):
        """Only extract blocks with matching language tag."""
        text = """
```python
python_code()
```

```repl
repl_code()
```

```javascript
javascript_code();
```
"""
        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 1
        assert "repl_code" in blocks[0].code

    def test_custom_language(self):
        """Can extract blocks with custom language tag."""
        text = """
```python
x = 5
```
"""
        blocks = extract_code_blocks(text, language="python")

        assert len(blocks) == 1
        assert "x = 5" in blocks[0].code


class TestExtractReplBlocks:
    """Tests for extract_repl_blocks function."""

    def test_extract_single_repl(self):
        """Extract single REPL block."""
        text = """
```repl
spawn_child_llm("test")
```
"""
        codes = extract_repl_blocks(text)

        assert len(codes) == 1
        assert "spawn_child_llm" in codes[0]

    def test_extract_multiple_repl(self):
        """Extract multiple REPL blocks."""
        text = """
First step:
```repl
result1 = spawn_child_llm("grid")
```

Second step:
```repl
result2 = spawn_child_llm("hex")
```
"""
        codes = extract_repl_blocks(text)

        assert len(codes) == 2
        assert "result1" in codes[0]
        assert "result2" in codes[1]

    def test_no_repl_block(self):
        """Return empty list when no REPL block."""
        text = """
```python
x = 5
```
"""
        codes = extract_repl_blocks(text)

        assert len(codes) == 0

    def test_ignores_python_blocks(self):
        """REPL extraction ignores python blocks."""
        text = """
Here's some Python code for reference:
```python
def example():
    pass
```

Now execute this:
```repl
spawn_child_llm("test")
```
"""
        codes = extract_repl_blocks(text)

        assert len(codes) == 1
        assert "spawn_child_llm" in codes[0]
        assert "def example" not in codes[0]


class TestExtractReasoning:
    """Tests for extract_reasoning function."""

    def test_extract_reasoning_simple(self):
        """Extract text outside code blocks."""
        text = """
First, I'll explain my approach.

```repl
spawn_child_llm("test")
```

This implements the solution.
"""
        reasoning = extract_reasoning(text)

        assert "First, I'll explain" in reasoning
        assert "This implements" in reasoning
        assert "spawn_child_llm" not in reasoning

    def test_extract_reasoning_multiple_blocks(self):
        """Handle multiple code blocks."""
        text = """
Step 1:
```repl
x = 1
```

Step 2:
```python
y = 2
```

Done.
"""
        reasoning = extract_reasoning(text)

        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Done" in reasoning
        assert "x = 1" not in reasoning
        assert "y = 2" not in reasoning

    def test_extract_reasoning_no_code(self):
        """Handle text with no code blocks."""
        text = "Just plain text with no code."

        reasoning = extract_reasoning(text)

        assert reasoning == "Just plain text with no code."
