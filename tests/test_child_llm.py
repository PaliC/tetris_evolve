"""Tests for Child LLM Executor (Component 5)."""

import pytest
from unittest.mock import Mock, patch

from tetris_evolve.llm.child_llm import (
    ChildLLMExecutor,
    ChildResult,
    SYSTEM_PROMPT,
    HELPER_CODE,
)
from tetris_evolve.llm.client import LLMClientError


class TestChildResult:
    """Tests for ChildResult dataclass."""

    def test_child_result_creation(self):
        """ChildResult dataclass creates correctly."""
        result = ChildResult(
            code="def choose_action(obs): return 5",
            reasoning="Simple hard drop strategy",
            raw_response="<reasoning>Simple</reasoning><code>...</code>",
            input_tokens=100,
            output_tokens=50,
            success=True,
            error=None,
        )

        assert "choose_action" in result.code
        assert result.success is True
        assert result.error is None

    def test_child_result_failure(self):
        """ChildResult captures failures."""
        result = ChildResult(
            code="",
            reasoning="",
            raw_response="",
            input_tokens=0,
            output_tokens=0,
            success=False,
            error="LLM API error",
        )

        assert result.success is False
        assert result.error == "LLM API error"


class TestPromptBuilding:
    """Tests for prompt building."""

    @pytest.fixture
    def executor(self):
        """Create executor with mock client."""
        mock_client = Mock()
        return ChildLLMExecutor(mock_client)

    def test_build_prompt_no_parent(self, executor):
        """Prompt built correctly without parent code."""
        prompt = executor._build_user_prompt(
            "Create a simple player",
            parent_code=None,
        )

        assert "## Task" in prompt
        assert "Create a simple player" in prompt
        assert "from scratch" in prompt
        assert "## Parent Code" not in prompt

    def test_build_prompt_with_parent(self, executor):
        """Prompt includes parent code when provided."""
        prompt = executor._build_user_prompt(
            "Improve line clearing",
            parent_code="def choose_action(obs): return 5",
        )

        assert "## Task" in prompt
        assert "Improve line clearing" in prompt
        assert "## Parent Code" in prompt
        assert "def choose_action(obs): return 5" in prompt
        assert "Improve this code" in prompt

    def test_build_prompt_includes_helper(self, executor):
        """Prompt includes helper code reference."""
        prompt = executor._build_user_prompt("Create a player", None)

        assert "## Helper Code" in prompt
        assert "parse_observation" in prompt


class TestResponseParsing:
    """Tests for response parsing."""

    @pytest.fixture
    def executor(self):
        """Create executor with mock client."""
        mock_client = Mock()
        return ChildLLMExecutor(mock_client)

    def test_extract_code_valid(self, executor):
        """Extracts code from valid tags."""
        response = """
<reasoning>
My strategy is to hard drop immediately.
</reasoning>

<code>
def choose_action(obs):
    return 5
</code>
"""
        code = executor._extract_code(response)

        assert "def choose_action(obs):" in code
        assert "return 5" in code

    def test_extract_code_no_tags(self, executor):
        """Returns empty string when no tags found."""
        response = "Here is some text without code tags."

        code = executor._extract_code(response)

        assert code == ""

    def test_extract_code_markdown_fallback(self, executor):
        """Falls back to markdown code blocks."""
        response = """
Some explanation.

```python
def choose_action(obs):
    return 3
```
"""
        code = executor._extract_code(response)

        assert "def choose_action(obs):" in code
        assert "return 3" in code

    def test_extract_reasoning(self, executor):
        """Extracts reasoning from tags."""
        response = """
<reasoning>
I'm using a simple strategy because:
1. It's fast
2. It works well
</reasoning>

<code>
def choose_action(obs): return 5
</code>
"""
        reasoning = executor._extract_reasoning(response)

        assert "simple strategy" in reasoning
        assert "It's fast" in reasoning

    def test_extract_reasoning_not_found(self, executor):
        """Returns empty string when no reasoning tags."""
        response = "<code>def choose_action(obs): return 5</code>"

        reasoning = executor._extract_reasoning(response)

        assert reasoning == ""

    def test_clean_code_markdown(self, executor):
        """Removes markdown fences."""
        code = "```python\ndef choose_action(obs):\n    return 5\n```"

        cleaned = executor._clean_code(code)

        assert "```" not in cleaned
        assert "def choose_action(obs):" in cleaned

    def test_clean_code_strips_whitespace(self, executor):
        """Strips leading/trailing whitespace."""
        code = "\n\n  def choose_action(obs):\n    return 5\n\n"

        cleaned = executor._clean_code(code)

        assert cleaned.startswith("def")
        assert cleaned.endswith("5")


class TestGenerate:
    """Tests for generate method."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = Mock()
        return client

    def test_generate_success(self, mock_client):
        """Full generation works correctly."""
        mock_client.send_message.return_value = Mock(
            content="""<reasoning>Simple hard drop</reasoning>
<code>
def choose_action(obs):
    return 5
</code>""",
            input_tokens=100,
            output_tokens=50,
        )

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate("Create a simple player")

        assert result.success is True
        assert "choose_action" in result.code
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_generate_with_parent(self, mock_client):
        """Generation with parent code works."""
        mock_client.send_message.return_value = Mock(
            content="<code>def choose_action(obs): return 3</code>",
            input_tokens=150,
            output_tokens=30,
        )

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate(
            "Add rotation",
            parent_code="def choose_action(obs): return 5",
        )

        assert result.success is True
        # Verify parent was included in prompt
        call_args = mock_client.send_message.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "Parent Code" in user_content

    def test_generate_llm_error(self, mock_client):
        """Handles LLM errors gracefully."""
        mock_client.send_message.side_effect = LLMClientError("API error")

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate("Create a player")

        assert result.success is False
        assert "LLM error" in result.error

    def test_generate_parse_error(self, mock_client):
        """Handles missing code in response."""
        mock_client.send_message.return_value = Mock(
            content="I can't generate code for that.",
            input_tokens=50,
            output_tokens=10,
        )

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate("Create something")

        assert result.success is False
        assert "No code found" in result.error

    def test_generate_uses_system_prompt(self, mock_client):
        """Uses correct system prompt."""
        mock_client.send_message.return_value = Mock(
            content="<code>def choose_action(obs): return 0</code>",
            input_tokens=100,
            output_tokens=20,
        )

        executor = ChildLLMExecutor(mock_client)
        executor.generate("Test")

        call_args = mock_client.send_message.call_args
        assert call_args.kwargs["system"] == SYSTEM_PROMPT


class TestGenerateAndValidate:
    """Tests for generate_and_validate method."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        return Mock()

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        evaluator = Mock()
        evaluator._validate_syntax.return_value = (True, None)
        evaluator._check_safety.return_value = (True, None)
        return evaluator

    def test_generate_and_validate_valid(self, mock_client, mock_evaluator):
        """Valid code passes validation."""
        mock_client.send_message.return_value = Mock(
            content="<code>def choose_action(obs): return 5</code>",
            input_tokens=100,
            output_tokens=20,
        )

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate_and_validate(
            "Create player",
            evaluator=mock_evaluator,
        )

        assert result.success is True
        mock_evaluator._validate_syntax.assert_called_once()

    def test_generate_and_validate_invalid(self, mock_client, mock_evaluator):
        """Invalid code fails validation."""
        mock_client.send_message.return_value = Mock(
            content="<code>def choose_action(obs) return 5</code>",
            input_tokens=100,
            output_tokens=20,
        )
        mock_evaluator._validate_syntax.return_value = (False, "Missing colon")

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate_and_validate(
            "Create player",
            evaluator=mock_evaluator,
            max_retries=0,
        )

        assert result.success is False
        assert "Syntax error" in result.error

    def test_retry_on_invalid(self, mock_client, mock_evaluator):
        """Retries on invalid code and eventually succeeds."""
        # First call returns invalid, second returns valid
        mock_client.send_message.side_effect = [
            Mock(
                content="<code>def choose_action(obs) return 5</code>",
                input_tokens=100,
                output_tokens=20,
            ),
            Mock(
                content="<code>def choose_action(obs): return 5</code>",
                input_tokens=120,
                output_tokens=25,
            ),
        ]
        mock_evaluator._validate_syntax.side_effect = [
            (False, "Missing colon"),
            (True, None),
        ]

        executor = ChildLLMExecutor(mock_client)
        result = executor.generate_and_validate(
            "Create player",
            evaluator=mock_evaluator,
            max_retries=2,
        )

        assert result.success is True
        assert mock_client.send_message.call_count == 2

    def test_retry_includes_error_feedback(self, mock_client, mock_evaluator):
        """Retry prompt includes error feedback."""
        mock_client.send_message.side_effect = [
            Mock(content="<code>bad code</code>", input_tokens=100, output_tokens=20),
            Mock(content="<code>def choose_action(obs): return 5</code>", input_tokens=120, output_tokens=25),
        ]
        mock_evaluator._validate_syntax.side_effect = [
            (False, "Syntax error on line 1"),
            (True, None),
        ]

        executor = ChildLLMExecutor(mock_client)
        executor.generate_and_validate(
            "Create player",
            evaluator=mock_evaluator,
            max_retries=2,
        )

        # Check second call includes error feedback
        second_call = mock_client.send_message.call_args_list[1]
        user_content = second_call.kwargs["messages"][0]["content"]
        assert "PREVIOUS ATTEMPT FAILED" in user_content


class TestBatchGeneration:
    """Tests for batch generation."""

    def test_generate_batch(self):
        """Batch generation processes all prompts."""
        mock_client = Mock()
        mock_client.send_message.return_value = Mock(
            content="<code>def choose_action(obs): return 5</code>",
            input_tokens=100,
            output_tokens=20,
        )

        executor = ChildLLMExecutor(mock_client)
        results = executor.generate_batch([
            ("Create player 1", None),
            ("Create player 2", None),
            ("Improve player", "def choose_action(obs): return 0"),
        ])

        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_client.send_message.call_count == 3


class TestSystemPromptContent:
    """Tests for system prompt content."""

    def test_system_prompt_has_observation_format(self):
        """System prompt describes observation format."""
        assert "244" in SYSTEM_PROMPT
        assert "Board state" in SYSTEM_PROMPT or "[0:200]" in SYSTEM_PROMPT

    def test_system_prompt_has_actions(self):
        """System prompt describes action space."""
        assert "0=no-op" in SYSTEM_PROMPT
        assert "hard_drop" in SYSTEM_PROMPT

    def test_system_prompt_has_interface(self):
        """System prompt specifies required interface."""
        assert "choose_action" in SYSTEM_PROMPT

    def test_helper_code_has_parser(self):
        """Helper code includes observation parser."""
        assert "parse_observation" in HELPER_CODE
        assert "board" in HELPER_CODE
