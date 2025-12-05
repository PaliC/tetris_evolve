"""
Tests for the Root LLM Orchestrator.
"""


import pytest

from tetris_evolve import (
    CostTracker,
    MockLLMClient,
    config_from_dict,
)
from tetris_evolve.root_llm import OrchestratorResult, RootLLMOrchestrator


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


@pytest.fixture
def mock_child_llm(mock_config):
    """Mock child LLM client."""
    cost_tracker = CostTracker(mock_config)
    return MockLLMClient(
        model=mock_config.child_llm.model,
        cost_tracker=cost_tracker,
        llm_type="child",
    )


class TestOrchestratorInitialization:
    """Tests for orchestrator initialization."""

    def test_initialization(self, mock_config, temp_dir):
        """Test that all components are initialized correctly."""
        orchestrator = RootLLMOrchestrator(mock_config)

        assert orchestrator.config == mock_config
        assert orchestrator.max_generations == mock_config.evolution.max_generations
        assert orchestrator.max_children_per_generation == mock_config.evolution.max_children_per_generation
        assert orchestrator.cost_tracker is not None
        assert orchestrator.logger is not None
        assert orchestrator.root_llm is not None
        assert orchestrator.child_llm is not None
        assert orchestrator.evaluator is not None
        assert orchestrator.evolution_api is not None
        assert orchestrator.repl is not None
        assert len(orchestrator.messages) == 0

    def test_initialization_with_mock_llms(self, mock_config, mock_root_llm, mock_child_llm):
        """Test initialization with provided mock LLM clients."""
        orchestrator = RootLLMOrchestrator(
            mock_config,
            root_llm=mock_root_llm,
            child_llm=mock_child_llm,
        )

        assert orchestrator.root_llm is mock_root_llm
        assert orchestrator.child_llm is mock_child_llm


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
    """Tests for extracting REPL code blocks."""

    def test_extract_code_blocks_single(self, mock_config):
        """Test extracting a single REPL block."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = '''Let me try a simple approach.

```repl
x = 1 + 2
print(x)
```

That should work.
'''
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "x = 1 + 2" in blocks[0]

    def test_extract_code_blocks_multiple(self, mock_config):
        """Test extracting multiple REPL blocks."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = '''First block:

```repl
a = 1
```

Second block:

```repl
b = 2
```
'''
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 2
        assert "a = 1" in blocks[0]
        assert "b = 2" in blocks[1]

    def test_extract_code_blocks_ignores_python(self, mock_config):
        """Test that Python blocks are ignored (only repl blocks extracted)."""
        orchestrator = RootLLMOrchestrator(mock_config)

        response = '''This is Python:

```python
def foo():
    pass
```

This is REPL:

```repl
x = 1
```
'''
        blocks = orchestrator.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "x = 1" in blocks[0]

    def test_extract_code_blocks_none(self, mock_config):
        """Test when no REPL blocks are present."""
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


class TestMaxGenerationsStop:
    """Tests for stopping at max generations."""

    def test_max_generations_stop(self, mock_config, temp_dir, sample_valid_packing_code):
        """Test that orchestrator stops at max generations."""
        # Set very low max generations
        mock_config.evolution.max_generations = 2

        orchestrator = RootLLMOrchestrator(mock_config)

        # Set up mock responses that spawn children (to trigger generation advance)
        cost_tracker = orchestrator.cost_tracker
        mock_root = MockLLMClient(
            model=mock_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )
        mock_child = MockLLMClient(
            model=mock_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        # Child returns valid packing code
        mock_child.set_responses([
            f"```python\n{sample_valid_packing_code}\n```",
            f"```python\n{sample_valid_packing_code}\n```",
        ])

        # Mock root responses: spawn, selection, spawn, selection
        mock_root.set_responses([
            '''Generation 0:
```repl
result = spawn_child_llm("Test gen 0")
print(f"Gen 0 result: {result['success']}")
```
''',
            '''```selection
{"selections": [{"trial_id": "trial_0_0", "reasoning": "Best", "category": "performance"}], "summary": "Selected best"}
```
''',
            '''Generation 1:
```repl
result = spawn_child_llm("Test gen 1")
print(f"Gen 1 result: {result['success']}")
```
''',
            '''```selection
{"selections": [{"trial_id": "trial_1_0", "reasoning": "Best", "category": "performance"}], "summary": "Selected best"}
```
''',
        ])
        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        orchestrator.evolution_api.child_llm = mock_child
        orchestrator.max_generations = 2

        result = orchestrator.run()

        assert result.terminated is True
        assert "max_generations" in result.reason
        assert result.num_generations == 2


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
        mock_root.set_responses([
            "Starting...\n\n```repl\nprint('hello')\n```",
        ])
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
        mock_root.set_responses([
            "I'll start.\n\n```repl\nx = 1\nprint(x)\n```",
            "Done.\n\n```repl\nterminate_evolution('test complete')\n```",
        ])
        orchestrator.root_llm = mock_root

        _result = orchestrator.run()

        # Check that messages were accumulated
        # Initial user + assistant response + user (execution results) + assistant + terminate
        assert len(orchestrator.messages) >= 3
        assert orchestrator.messages[0]["role"] == "user"
        assert orchestrator.messages[1]["role"] == "assistant"


class TestFullOrchestration:
    """Integration tests for full orchestration loop."""

    def test_full_loop_with_termination(self, mock_config, temp_dir, sample_valid_packing_code):
        """Test a full orchestration loop that terminates properly."""
        orchestrator = RootLLMOrchestrator(mock_config)
        cost_tracker = orchestrator.cost_tracker

        # Create mock responses with a child spawn and termination
        mock_root = MockLLMClient(
            model=mock_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )

        mock_child = MockLLMClient(
            model=mock_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )

        # Child returns valid packing code
        mock_child.set_responses([
            f"Here's my solution:\n\n```python\n{sample_valid_packing_code}\n```"
        ])

        # Root spawns a child, provides selection, then terminates in next generation
        mock_root.set_responses([
            '''Let me spawn a child to try a solution.

```repl
prompt = "Write a circle packing algorithm."
result = spawn_child_llm(prompt)
print(f"Result: valid={result['metrics'].get('valid')}, sum={result['metrics'].get('sum_radii', 0):.4f}")
best_code = result['code']
```
''',
            '''```selection
{"selections": [{"trial_id": "trial_0_0", "reasoning": "Only trial", "category": "performance"}], "summary": "Selected only trial"}
```
''',
            '''Good result! Let me terminate.

```repl
terminate_evolution("Found a valid solution", best_program=best_code)
```
''',
        ])

        orchestrator.root_llm = mock_root
        orchestrator.child_llm = mock_child
        # Also update the evolution API's child LLM
        orchestrator.evolution_api.child_llm = mock_child

        result = orchestrator.run()

        assert result.terminated is True
        assert result.total_trials == 1
        assert result.successful_trials == 1
        assert result.best_score > 0
        assert result.cost_summary is not None

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
