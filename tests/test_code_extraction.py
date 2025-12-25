"""
Tests for mango_evolve.utils.code_extraction module.
"""

from mango_evolve.utils import (
    extract_code_blocks,
    extract_python_code,
    extract_reasoning,
    extract_repl_blocks,
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


class TestNestedCodeBlocks:
    """Tests for handling nested code blocks (code blocks inside code blocks)."""

    def test_repl_block_with_nested_python(self):
        """Extract REPL block containing nested ```python``` example."""
        text = '''Here's my approach:

```repl
# Strategy 1: Hexagonal lattice packing
prompt1 = """
Write a circle packing algorithm.

Example structure:
```python
import numpy as np

def construct_packing():
    return centers, radii, np.sum(radii)
```

Focus on maximizing sum of radii.
"""

result1 = spawn_child_llm(prompt1)
print(f"Trial {result1['trial_id']}")
```

Now try another approach.
'''
        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 1
        # The extracted code should contain the nested ```python block
        assert "```python" in blocks[0].code
        assert "spawn_child_llm" in blocks[0].code
        assert "import numpy" in blocks[0].code

    def test_repl_block_with_multiple_nested_blocks(self):
        """Handle REPL block with multiple nested code blocks."""
        text = '''Explanation:

```repl
prompt = """
Here are two approaches:

Approach 1:
```python
def method1():
    pass
```

Approach 2:
```python
def method2():
    pass
```

Choose the best one.
"""
result = spawn_child_llm(prompt)
```

Done.
'''
        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 1
        assert "method1" in blocks[0].code
        assert "method2" in blocks[0].code
        assert "spawn_child_llm" in blocks[0].code

    def test_multiple_repl_blocks_with_nested(self):
        """Handle multiple REPL blocks each with nested code."""
        text = '''First:

```repl
prompt1 = """
```python
def foo():
    pass
```
"""
result1 = spawn_child_llm(prompt1)
```

Second:

```repl
prompt2 = """
```python
def bar():
    pass
```
"""
result2 = spawn_child_llm(prompt2)
```

Done.
'''
        blocks = extract_code_blocks(text, language="repl")

        assert len(blocks) == 2
        assert "foo" in blocks[0].code
        assert "result1" in blocks[0].code
        assert "bar" in blocks[1].code
        assert "result2" in blocks[1].code

    def test_extract_reasoning_with_nested_blocks(self):
        """Reasoning extraction should remove entire blocks including nested ones."""
        text = '''First, I'll explain my approach.

```repl
prompt = """
Example:
```python
x = 5
```
"""
spawn_child_llm(prompt)
```

This implements the solution.
'''
        reasoning = extract_reasoning(text)

        assert "First, I'll explain" in reasoning
        assert "This implements" in reasoning
        # All code including nested should be removed
        assert "spawn_child_llm" not in reasoning
        assert "x = 5" not in reasoning
        assert "```" not in reasoning

    def test_unlabeled_nested_block_closes_outer(self):
        """Unlabeled nested blocks (no language tag) close the outer block.

        This is expected behavior - only labeled code blocks (```python, etc.)
        are treated as nested. Unlabeled ``` markers close blocks.
        """
        text = '''Start:

```repl
prompt = """
Example:
```
code_here
```
"""
result = func(prompt)
```

End.
'''
        blocks = extract_code_blocks(text, language="repl")

        # The first unlabeled ``` closes the repl block early
        # This is a known limitation - use labeled blocks for nesting
        assert len(blocks) == 1
        # Only content up to the first ``` is captured
        assert "prompt" in blocks[0].code
        assert "Example:" in blocks[0].code


class TestExtractPythonCode:
    """Tests for extract_python_code function."""

    def test_extract_python_block(self):
        """Extract code from ```python block."""
        text = """
Here's my solution:

```python
def construct_packing():
    return centers, radii, sum_radii
```

This implements the algorithm.
"""
        code = extract_python_code(text)

        assert code is not None
        assert "def construct_packing" in code

    def test_extract_unlabeled_block(self):
        """Fall back to unlabeled code block if no python block."""
        text = """
Here's some code:

```
x = 5
y = 10
```
"""
        code = extract_python_code(text)

        assert code is not None
        assert "x = 5" in code

    def test_prefers_python_over_unlabeled(self):
        """Prefer ```python block over unlabeled."""
        text = """
```
unlabeled code
```

```python
python_code()
```
"""
        code = extract_python_code(text)

        assert code is not None
        assert "python_code" in code
        assert "unlabeled" not in code

    def test_no_code_block_returns_none(self):
        """Return None when no code blocks found."""
        text = "Just plain text with no code."

        code = extract_python_code(text)

        assert code is None

    def test_ignores_repl_blocks(self):
        """Only extract python or unlabeled blocks, not repl."""
        text = """
```repl
spawn_child_llm("test")
```
"""
        code = extract_python_code(text)

        assert code is None

    def test_ignores_other_languages(self):
        """Ignore non-python language tags."""
        text = """
```javascript
console.log("hello");
```
"""
        code = extract_python_code(text)

        assert code is None
