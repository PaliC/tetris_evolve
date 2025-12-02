"""
Tests for tetris_evolve.utils.code_extraction module.
"""

import pytest

from tetris_evolve.utils import (
    extract_code_blocks,
    extract_python_code,
    extract_repl_code,
    extract_all_code,
    extract_with_reasoning,
    has_function,
    has_required_functions,
    CodeBlock,
)


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks function."""

    def test_extract_python_block(self):
        """Extract code from ```python``` block."""
        text = """
Here is some code:

```python
def hello():
    print("Hello")
```

That's all.
"""
        blocks = extract_code_blocks(text, languages=["python"])

        assert len(blocks) == 1
        assert "def hello():" in blocks[0].code
        assert blocks[0].language == "python"

    def test_extract_repl_block(self):
        """Extract code from ```repl``` block."""
        text = """
```repl
x = spawn_child_llm("test")
print(x)
```
"""
        blocks = extract_code_blocks(text, languages=["repl"])

        assert len(blocks) == 1
        assert "spawn_child_llm" in blocks[0].code
        assert blocks[0].language == "repl"

    def test_multiple_blocks(self):
        """Handle multiple code blocks."""
        text = """
```python
def func1():
    pass
```

Some text

```python
def func2():
    pass
```
"""
        blocks = extract_code_blocks(text, languages=["python"])

        assert len(blocks) == 2
        assert "func1" in blocks[0].code
        assert "func2" in blocks[1].code

    def test_no_code_block(self):
        """Return empty list when no code found."""
        text = "This is just plain text with no code."

        blocks = extract_code_blocks(text, languages=["python"])

        assert len(blocks) == 0

    def test_malformed_block(self):
        """Handle edge cases gracefully."""
        # Unclosed block
        text = """
```python
def hello():
    pass
"""
        blocks = extract_code_blocks(text, languages=["python"])
        # Should not match unclosed blocks
        assert len(blocks) == 0

    def test_filter_by_language(self):
        """Filter blocks by language tag."""
        text = """
```python
python_code()
```

```javascript
javascript_code();
```
"""
        blocks = extract_code_blocks(text, languages=["python"])

        assert len(blocks) == 1
        assert "python_code" in blocks[0].code

    def test_no_language_filter(self):
        """Accept all blocks when no filter."""
        text = """
```python
python_code()
```

```javascript
javascript_code();
```

```
unmarked_code()
```
"""
        blocks = extract_code_blocks(text)

        assert len(blocks) == 3


class TestExtractPythonCode:
    """Tests for extract_python_code function."""

    def test_extract_python(self):
        """Extract Python code."""
        text = """
```python
x = 5
```
"""
        code = extract_python_code(text)

        assert code == "x = 5"

    def test_extract_py_alias(self):
        """Accept 'py' language tag."""
        text = """
```py
x = 5
```
"""
        code = extract_python_code(text)

        assert code == "x = 5"

    def test_no_python_block(self):
        """Return None when no Python block."""
        text = "Just text, no code."

        code = extract_python_code(text)

        assert code is None

    def test_unmarked_block_fallback(self):
        """Fall back to unmarked blocks."""
        text = """
```
x = 5
```
"""
        code = extract_python_code(text)

        assert code == "x = 5"


class TestExtractReplCode:
    """Tests for extract_repl_code function."""

    def test_extract_repl(self):
        """Extract REPL code."""
        text = """
```repl
spawn_child_llm("test")
```
"""
        code = extract_repl_code(text)

        assert "spawn_child_llm" in code

    def test_no_repl_block(self):
        """Return None when no REPL block."""
        text = """
```python
x = 5
```
"""
        code = extract_repl_code(text)

        assert code is None


class TestExtractAllCode:
    """Tests for extract_all_code function."""

    def test_extract_reasoning(self):
        """Extract text outside code blocks."""
        text = """
First, I'll explain my approach.

```python
def solve():
    pass
```

This implements the solution.
"""
        codes, reasoning = extract_all_code(text, languages=["python"])

        assert len(codes) == 1
        assert "First, I'll explain" in reasoning
        assert "This implements" in reasoning

    def test_multiple_blocks_and_reasoning(self):
        """Handle multiple blocks with reasoning."""
        text = """
Step 1:
```python
x = 1
```

Step 2:
```python
y = 2
```

Done.
"""
        codes, reasoning = extract_all_code(text, languages=["python"])

        assert len(codes) == 2
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Done" in reasoning


class TestExtractWithReasoning:
    """Tests for extract_with_reasoning function."""

    def test_returns_extraction_result(self):
        """Returns ExtractionResult dataclass."""
        text = """
Here's my solution:

```python
def solve():
    return 42
```

It returns 42.
"""
        result = extract_with_reasoning(text)

        assert len(result.code_blocks) == 1
        assert "Here's my solution" in result.reasoning
        assert "It returns 42" in result.reasoning


class TestHasFunction:
    """Tests for has_function function."""

    def test_function_exists(self):
        """Detect existing function."""
        code = """
def run_packing():
    return None
"""
        assert has_function(code, "run_packing") is True

    def test_function_missing(self):
        """Detect missing function."""
        code = """
def other_function():
    pass
"""
        assert has_function(code, "run_packing") is False

    def test_indented_function(self):
        """Handle indented function definitions."""
        code = """
class Foo:
    def run_packing(self):
        pass
"""
        assert has_function(code, "run_packing") is True


class TestHasRequiredFunctions:
    """Tests for has_required_functions function."""

    def test_has_run_packing(self):
        """Accept code with run_packing."""
        code = "def run_packing():\n    pass"

        has_required, error = has_required_functions(code)

        assert has_required is True
        assert error is None

    def test_has_construct_packing(self):
        """Accept code with construct_packing."""
        code = "def construct_packing():\n    pass"

        has_required, error = has_required_functions(code)

        assert has_required is True
        assert error is None

    def test_missing_required(self):
        """Reject code without required functions."""
        code = "def other_function():\n    pass"

        has_required, error = has_required_functions(code)

        assert has_required is False
        assert error is not None
        assert "run_packing" in error or "construct_packing" in error
