"""
Tests for Child LLM executor.

These tests define the expected behavior of the Child RLLM execution
system that generates code mutations.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from tetris_evolve.child_llm import (
    ChildLLMExecutor,
    CodeGenerator,
    CodeValidator,
    GenerationResult,
    ValidationError,
)
from tetris_evolve.rlm import SharedREPL
from tetris_evolve.environment import TetrisConfig


class TestCodeValidator:
    """Tests for CodeValidator."""

    @pytest.fixture
    def validator(self):
        return CodeValidator()

    def test_validate_valid_code(self, validator):
        """Should accept valid Python code."""
        code = '''
class TetrisPlayer:
    def select_action(self, obs):
        return 5
    def reset(self):
        pass
'''
        result = validator.validate(code)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_syntax_error(self, validator):
        """Should detect syntax errors."""
        code = "def broken( {"

        result = validator.validate(code)

        assert result.is_valid is False
        assert any("syntax" in e.lower() for e in result.errors)

    def test_validate_missing_class(self, validator):
        """Should detect missing player class."""
        code = '''
def some_function():
    return 5
'''
        result = validator.validate(code, require_class="TetrisPlayer")

        assert result.is_valid is False
        assert any("TetrisPlayer" in e for e in result.errors)

    def test_validate_missing_method(self, validator):
        """Should detect missing required method."""
        code = '''
class TetrisPlayer:
    def reset(self):
        pass
'''
        result = validator.validate(code, require_method="select_action")

        assert result.is_valid is False
        assert any("select_action" in e for e in result.errors)

    def test_validate_dangerous_code(self, validator):
        """Should detect potentially dangerous code."""
        code = '''
import os
os.system("rm -rf /")

class TetrisPlayer:
    def select_action(self, obs):
        return 5
'''
        result = validator.validate(code, check_safety=True)

        assert result.is_valid is False
        assert any("dangerous" in e.lower() or "os.system" in e for e in result.errors)

    def test_validate_import_subprocess(self, validator):
        """Should flag subprocess imports."""
        code = '''
import subprocess
subprocess.run(["ls"])

class TetrisPlayer:
    def select_action(self, obs):
        return 5
'''
        result = validator.validate(code, check_safety=True)

        assert result.is_valid is False


class TestCodeGenerator:
    """Tests for CodeGenerator."""

    @pytest.fixture
    def generator(self):
        return CodeGenerator(env_config=TetrisConfig())

    def test_extract_code_from_response(self, generator):
        """Should extract code from LLM response."""
        response = '''
Here's the improved code:

```python
class TetrisPlayer:
    def select_action(self, obs):
        return 5
    def reset(self):
        pass
```

This should work better.
'''
        code = generator.extract_code(response)

        assert "class TetrisPlayer" in code
        assert "def select_action" in code

    def test_extract_code_no_markers(self, generator):
        """Should handle response without code markers."""
        response = '''class TetrisPlayer:
    def select_action(self, obs):
        return 5
    def reset(self):
        pass'''

        code = generator.extract_code(response)

        assert "class TetrisPlayer" in code

    def test_extract_code_multiple_blocks(self, generator):
        """Should extract the relevant code block."""
        response = '''
First, here's a helper:

```python
def helper():
    pass
```

And here's the player:

```python
class TetrisPlayer:
    def select_action(self, obs):
        return 5
    def reset(self):
        pass
```
'''
        code = generator.extract_code(response)

        # Should prefer the block with TetrisPlayer
        assert "class TetrisPlayer" in code

    def test_build_prompt_includes_context(self, generator):
        """Should build prompt with environment context."""
        prompt = generator.build_prompt(
            base_prompt="Improve this player",
            parent_code="class OldPlayer: pass",
        )

        assert "Improve this player" in prompt
        assert "class OldPlayer" in prompt
        # Should include environment description
        assert "observation" in prompt.lower() or "action" in prompt.lower()


class TestChildLLMExecutor:
    """Tests for ChildLLMExecutor."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.generate = Mock(return_value='''
```python
class TetrisPlayer:
    def select_action(self, obs):
        return 5
    def reset(self):
        pass
```
''')
        return client

    @pytest.fixture
    def executor(self, mock_llm_client):
        return ChildLLMExecutor(
            llm_client=mock_llm_client,
            env_config=TetrisConfig(),
        )

    def test_execute_returns_result(self, executor):
        """Should return GenerationResult."""
        result = executor.execute("Generate a player")

        assert isinstance(result, GenerationResult)

    def test_execute_generates_valid_code(self, executor):
        """Should generate valid code."""
        result = executor.execute("Generate a player")

        assert result.success is True
        assert result.code is not None
        assert "class TetrisPlayer" in result.code

    def test_execute_with_validation(self, executor):
        """Should validate generated code."""
        result = executor.execute("Generate a player", validate=True)

        assert result.validation_passed is True

    def test_execute_handles_invalid_response(self, executor, mock_llm_client):
        """Should handle invalid LLM response."""
        mock_llm_client.generate.return_value = "Not valid Python code {"

        result = executor.execute("Generate a player", validate=True)

        assert result.success is False
        assert result.validation_passed is False

    def test_execute_with_repl_context(self, executor):
        """Should use shared REPL context."""
        repl = SharedREPL()
        repl.set_variable("parent_score", 1000)

        result = executor.execute(
            "Generate a player (parent score: {parent_score})",
            repl_context=repl,
        )

        assert result is not None

    def test_execute_tracks_metadata(self, executor):
        """Should track execution metadata."""
        result = executor.execute("Generate a player")

        assert result.rllm_id is not None
        assert result.prompt is not None
        assert result.execution_time >= 0


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = GenerationResult(
            success=True,
            code="class Player: pass",
            rllm_id="rllm_001",
            prompt="test prompt",
            validation_passed=True,
            execution_time=1.5,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["rllm_id"] == "rllm_001"
        assert d["execution_time"] == 1.5

    def test_from_error(self):
        """Should create error result."""
        result = GenerationResult.from_error(
            error="Something went wrong",
            rllm_id="rllm_001",
            prompt="test",
        )

        assert result.success is False
        assert "Something went wrong" in result.error


class TestChildLLMIntegration:
    """Integration tests for Child LLM with REPL."""

    @pytest.fixture
    def repl(self):
        return SharedREPL()

    def test_generated_code_runs_in_repl(self, repl):
        """Generated code should be executable in REPL."""
        code = '''
class TetrisPlayer:
    def __init__(self):
        self.count = 0
    def select_action(self, obs):
        self.count += 1
        return 5
    def reset(self):
        self.count = 0
'''
        repl.execute(code)

        # Should be able to use the class
        result = repl.execute("TetrisPlayer().select_action(None)")
        assert result == 5

    def test_generated_code_uses_repl_context(self, repl):
        """Generated code should access REPL context."""
        repl.set_variable("WEIGHTS", [1.0, 2.0, 3.0])

        code = '''
class TetrisPlayer:
    def select_action(self, obs):
        return int(sum(WEIGHTS))
    def reset(self):
        pass
'''
        repl.execute(code)

        result = repl.execute("TetrisPlayer().select_action(None)")
        assert result == 6
