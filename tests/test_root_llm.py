"""
Tests for the Root LLM Orchestrator.
"""

import pytest

from mango_evolve import (
    CostTracker,
    MockLLMClient,
    config_from_dict,
)
from mango_evolve.root_llm import OrchestratorResult, RootLLMOrchestrator


@pytest.fixture
def mock_config(sample_config_dict, temp_dir):
    """Config with temp directory for output."""
    sample_config_dict["experiment"]["output_dir"] = str(temp_dir)
    sample_config_dict["root_llm"]["max_iterations"] = 5
    sample_config_dict["budget"]["max_total_cost"] = 10.0
    return config_from_dict(sample_config_dict)


@pytest.fixture
def mock_root_llm(mock_config):
    """Mock root LLM client."""
    cost_tracker = CostTracker(mock_config)
    return MockLLMClient(
        model=mock_config.root_llm.model,
        cost_tracker=cost_tracker,
        llm_type="root",
    )


class TestOrchestratorInitialization:
    """Tests for orchestrator initialization."""

    def test_initialization(self, mock_config, temp_dir):
        """Test that all components are initialized correctly."""
        orchestrator = RootLLMOrchestrator(mock_config)

        assert orchestrator.config == mock_config
        # max_generations accessed via evolution_api (single source of truth)
        assert (
            orchestrator.evolution_api.max_generations
            == mock_config.evolution.max_generations
        )
        assert (
            orchestrator.evolution_api.max_children_per_generation
            == mock_config.evolution.max_children_per_generation
        )
        assert orchestrator.cost_tracker is not None
        assert orchestrator.logger is not None
        assert orchestrator.root_llm is not None
        assert len(orchestrator.child_llm_configs) > 0  # Has child LLM configs
        assert orchestrator.evaluator is not None
        assert orchestrator.evolution_api is not None
        assert orchestrator.repl is not None
        assert len(orchestrator.messages) == 0


class TestBuildInitialMessages:
    """Tests for building initial messages."""

    def test_build_initial_messages(self, mock_config):
        """Test that initial messages are constructed correctly."""
        orchestrator = RootLLMOrchestrator(mock_config)
        messages = orchestrator.build_initial_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "generation 0" in messages[0]["content"].lower()
        assert "spawn" in messages[0]["content"].lower()


class TestCodeBlockExtraction:
    """Tests for extracting Python code blocks."""

    def test_extract_code_blocks_single(self, mock_config):
        """Test extracting a single Python block."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = """Let me try a simple approach.

```python
x = 1 + 2
print(x)
```

That should work.
"""
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "x = 1 + 2" in blocks[0]

    def test_extract_code_blocks_multiple(self, mock_config):
        """Test extracting multiple Python blocks."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = """First block:

```python
a = 1
```

Second block:

```python
b = 2
```
"""
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 2
        assert "a = 1" in blocks[0]
        assert "b = 2" in blocks[1]

    def test_extract_code_blocks_multiple_python(self, mock_config):
        """Test that multiple python blocks are extracted in order."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = """This is Python:

```python
def foo():
    pass
```

More Python:

```python
x = 1
```
"""
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 2
        assert "def foo" in blocks[0]
        assert "x = 1" in blocks[1]

    def test_extract_code_blocks_none(self, mock_config):
        """Test when no Python blocks are present."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = "Just some text with no code blocks."
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 0


class TestREPLExecution:
    """Tests for executing code in the REPL."""

    def test_execute_code_in_repl_success(self, mock_config):
        """Test successful code execution."""
        orchestrator = RootLLMOrchestrator(mock_config)

        result = orchestrator.execute_code_in_repl("print('hello')")
        assert "hello" in result

    def test_execute_code_in_repl_with_return(self, mock_config):
        """Test code execution with return value."""
        orchestrator = RootLLMOrchestrator(mock_config)

        result = orchestrator.execute_code_in_repl("1 + 2")
        assert "3" in result

    def test_execute_code_in_repl_error(self, mock_config):
        """Test code execution with error."""
        orchestrator = RootLLMOrchestrator(mock_config)

        result = orchestrator.execute_code_in_repl("undefined_variable")
        assert "error" in result.lower() or "Error" in result

    def test_execute_code_in_repl_no_output(self, mock_config):
        """Test code execution with no output."""
        orchestrator = RootLLMOrchestrator(mock_config)

        result = orchestrator.execute_code_in_repl("x = 5")
        assert "executed successfully" in result.lower() or "no output" in result.lower()


class TestTerminationDetection:
    """Tests for detecting termination."""

    def test_termination_detection_not_terminated(self, mock_config):
        """Test that non-terminated state is detected."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = "Let me try something else."
        assert orchestrator.check_termination(response) is False

    def test_termination_detection_after_terminate_call(self, mock_config):
        """Test detection after terminate_evolution is called."""
        orchestrator = RootLLMOrchestrator(mock_config)

        # Execute terminate_evolution in REPL
        orchestrator.repl.execute("terminate_evolution('done testing')")

        # Now check termination
        assert orchestrator.check_termination("") is True


class TestBudgetExceededStop:
    """Tests for stopping on budget exceeded."""

    def test_budget_exceeded_stop(self, mock_config, temp_dir):
        """Test that orchestrator stops when budget is exceeded."""
        # Set very low budget
        mock_config.budget.max_total_cost = 0.0001

        orchestrator = RootLLMOrchestrator(mock_config)

        # Set up mock root LLM
        cost_tracker = orchestrator.cost_tracker
        mock_root = MockLLMClient(
            model=mock_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_root.set_responses(
            [
                "Starting...\n\n```python\nprint('hello')\n```",
            ]
        )
        orchestrator.root_llm = mock_root

        # Pre-spend most of the budget
        cost_tracker.record_usage(
            input_tokens=10000,
            output_tokens=10000,
            llm_type="root",
            call_id="test",
        )

        result = orchestrator.run()

        assert result.terminated is True
        assert "budget_exceeded" in result.reason


class TestConversationHistory:
    """Tests for conversation history maintenance."""

    def test_conversation_history(self, mock_config, temp_dir):
        """Test that conversation history is maintained correctly."""
        orchestrator = RootLLMOrchestrator(mock_config)

        # Build initial messages
        orchestrator.build_initial_messages()
        assert len(orchestrator.messages) == 1
        assert orchestrator.messages[0]["role"] == "user"

        # Simulate one iteration with mock LLM
        cost_tracker = orchestrator.cost_tracker
        mock_root = MockLLMClient(
            model=mock_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_root.set_responses(
            [
                "I'll start.\n\n```python\nx = 1\nprint(x)\n```",
                "Done.\n\n```python\nterminate_evolution('test complete')\n```",
            ]
        )
        orchestrator.root_llm = mock_root

        _result = orchestrator.run()

        # Check that messages were accumulated
        # Initial user + assistant response + user (execution results) + assistant + terminate
        assert len(orchestrator.messages) >= 3
        assert orchestrator.messages[0]["role"] == "user"
        assert orchestrator.messages[1]["role"] == "assistant"


class TestFullOrchestration:
    """Integration tests for full orchestration loop."""

    def test_orchestrator_result_fields(self, mock_config, temp_dir):
        """Test that OrchestratorResult has all expected fields."""
        result = OrchestratorResult(
            terminated=True,
            reason="test",
            num_generations=5,
            best_program="code",
            best_score=2.0,
            total_trials=10,
            successful_trials=8,
            cost_summary={"total_cost": 1.0},
        )

        assert result.terminated is True
        assert result.reason == "test"
        assert result.num_generations == 5
        assert result.best_program == "code"
        assert result.best_score == 2.0
        assert result.total_trials == 10
        assert result.successful_trials == 8
        assert result.cost_summary == {"total_cost": 1.0}
