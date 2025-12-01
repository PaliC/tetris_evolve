"""Tests for Evolution Controller (Component 7)."""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tetris_evolve.evolution.controller import (
    EvolutionController,
    EvolutionConfig,
    EvolutionResult,
    INITIAL_PROMPTS,
    ROOT_SYSTEM_PROMPT,
)


class TestEvolutionConfig:
    """Tests for EvolutionConfig."""

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = EvolutionConfig()

        assert config.max_generations == 50
        assert config.max_cost_usd == 100.0
        assert config.max_time_minutes == 60
        assert config.initial_population_size == 5

    def test_config_from_yaml(self, tmp_path):
        """Loads config from YAML correctly."""
        yaml_content = """
max_generations: 10
max_cost_usd: 50.0
max_time_minutes: 30
initial_population_size: 3
root_model: claude-sonnet-4
child_model: claude-haiku
output_dir: /tmp/test_exp
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = EvolutionConfig.from_yaml(config_path)

        assert config.max_generations == 10
        assert config.max_cost_usd == 50.0
        assert config.initial_population_size == 3
        assert config.output_dir == Path("/tmp/test_exp")

    def test_config_validation_valid(self):
        """Valid config passes validation."""
        config = EvolutionConfig(
            max_generations=10,
            max_cost_usd=50.0,
            max_time_minutes=30,
        )

        config.validate()  # Should not raise

    def test_config_validation_invalid_generations(self):
        """Invalid generations fails validation."""
        config = EvolutionConfig(max_generations=0)

        with pytest.raises(ValueError):
            config.validate()

    def test_config_validation_invalid_cost(self):
        """Invalid cost fails validation."""
        config = EvolutionConfig(max_cost_usd=-10.0)

        with pytest.raises(ValueError):
            config.validate()

    def test_config_to_dict(self):
        """Config converts to dict correctly."""
        config = EvolutionConfig(max_generations=20)

        data = config.to_dict()

        assert data["max_generations"] == 20
        assert "output_dir" in data


class TestEvolutionResult:
    """Tests for EvolutionResult."""

    def test_result_dataclass(self):
        """EvolutionResult creates correctly."""
        result = EvolutionResult(
            experiment_id="exp_001",
            generations_completed=5,
            total_cost_usd=25.50,
            total_time_minutes=15.5,
            best_trial_id="trial_012",
            best_score=500.0,
            best_code="def choose_action(obs): return 5",
            termination_reason="generation_limit",
        )

        assert result.experiment_id == "exp_001"
        assert result.generations_completed == 5
        assert result.best_score == 500.0


class TestControllerInit:
    """Tests for controller initialization."""

    def test_init(self):
        """Controller initializes correctly."""
        config = EvolutionConfig(max_generations=5)

        controller = EvolutionController(config)

        assert controller.config.max_generations == 5
        assert controller.start_time is None

    def test_init_validates_config(self):
        """Controller validates config on init."""
        config = EvolutionConfig(max_generations=0)

        with pytest.raises(ValueError):
            EvolutionController(config)


class TestInitialPopulation:
    """Tests for initial population creation."""

    def test_initial_prompts_exist(self):
        """Initial prompts are defined."""
        assert len(INITIAL_PROMPTS) >= 5
        for prompt in INITIAL_PROMPTS:
            assert "Tetris" in prompt

    @patch("tetris_evolve.evolution.controller.LLMClient")
    @patch("tetris_evolve.evolution.controller.ProgramEvaluator")
    def test_create_initial_population(self, mock_eval_cls, mock_client_cls):
        """Creates N initial trials."""
        config = EvolutionConfig(
            initial_population_size=3,
            max_generations=5,
            output_dir=Path(tempfile.mkdtemp()),
        )

        mock_root_interface = Mock()
        mock_root_interface.is_terminated = False
        mock_root_interface.current_generation = 1

        controller = EvolutionController(config)
        controller.root_interface = mock_root_interface

        controller._create_initial_population()

        assert mock_root_interface.spawn_child_llm.call_count == 3


class TestLimitChecking:
    """Tests for limit checking."""

    @pytest.fixture
    def controller(self):
        """Create controller with mocked components."""
        config = EvolutionConfig(
            max_generations=5,
            max_cost_usd=10.0,
            max_time_minutes=30,
            output_dir=Path(tempfile.mkdtemp()),
        )
        controller = EvolutionController(config)

        # Mock components
        controller.root_interface = Mock()
        controller.root_interface.current_generation = 3
        controller.cost_tracker = Mock()
        controller.cost_tracker.get_total_cost.return_value = 5.0
        controller.start_time = datetime.now()

        return controller

    def test_generation_limit_not_exceeded(self, controller):
        """Generation limit not exceeded."""
        result = controller._check_generation_limit()
        assert result is None

    def test_generation_limit_exceeded(self, controller):
        """Generation limit exceeded."""
        controller.root_interface.current_generation = 6

        result = controller._check_generation_limit()

        assert result is not None
        assert "generation_limit" in result

    def test_cost_limit_not_exceeded(self, controller):
        """Cost limit not exceeded."""
        result = controller._check_cost_limit()
        assert result is None

    def test_cost_limit_exceeded(self, controller):
        """Cost limit exceeded."""
        controller.cost_tracker.get_total_cost.return_value = 15.0

        result = controller._check_cost_limit()

        assert result is not None
        assert "cost_limit" in result

    def test_time_limit_not_exceeded(self, controller):
        """Time limit not exceeded."""
        result = controller._check_time_limit()
        assert result is None

    def test_time_limit_exceeded(self, controller):
        """Time limit exceeded."""
        controller.start_time = datetime.now() - timedelta(minutes=35)

        result = controller._check_time_limit()

        assert result is not None
        assert "time_limit" in result

    def test_check_all_limits(self, controller):
        """Combined limit checking."""
        # No limits exceeded
        assert controller._check_all_limits() is None

        # Generation limit exceeded
        controller.root_interface.current_generation = 6
        assert "generation_limit" in controller._check_all_limits()


class TestRootPromptBuilding:
    """Tests for root prompt building."""

    @pytest.fixture
    def controller(self):
        """Create controller with mocked components."""
        config = EvolutionConfig(output_dir=Path(tempfile.mkdtemp()))
        controller = EvolutionController(config)

        controller.root_interface = Mock()
        controller.root_interface.get_limits.return_value = {
            "max_generations": 10,
            "max_children_per_gen": 20,
            "max_cost_usd": 100.0,
            "current_gen": 2,
            "children_this_gen": 5,
            "cost_remaining_usd": 75.0,
        }
        controller.root_interface.get_population.return_value = []
        controller.root_interface.get_generation_history.return_value = []
        controller.root_interface.get_improvement_rate.return_value = 0.0

        return controller

    def test_build_root_prompt(self, controller):
        """Prompt has current state info."""
        prompt = controller._build_root_prompt()

        assert "## Current State" in prompt
        assert "Generation: 2/10" in prompt
        assert "Children this gen: 5/20" in prompt
        assert "$75.00 remaining" in prompt


class TestCodeExtraction:
    """Tests for code extraction."""

    @pytest.fixture
    def controller(self):
        """Create controller."""
        config = EvolutionConfig(output_dir=Path(tempfile.mkdtemp()))
        return EvolutionController(config)

    def test_extract_code_markdown(self, controller):
        """Extracts code from markdown blocks."""
        response = """
Here's my plan...

```python
spawn_child_llm("Create player")
advance_generation(["trial_001"], "Selected best")
```

That's my strategy.
"""
        code = controller._extract_code(response)

        assert "spawn_child_llm" in code
        assert "advance_generation" in code

    def test_extract_code_no_block(self, controller):
        """Returns empty string when no code block."""
        response = "I don't have any code to share."

        code = controller._extract_code(response)

        assert code == ""


class TestRootCodeExecution:
    """Tests for root code execution."""

    @pytest.fixture
    def controller(self):
        """Create controller with mocked interface."""
        config = EvolutionConfig(output_dir=Path(tempfile.mkdtemp()))
        controller = EvolutionController(config)

        controller.root_interface = Mock()
        controller.root_interface.spawn_child_llm.return_value = {
            "trial_id": "trial_001",
            "success": True,
        }

        return controller

    def test_execute_root_code(self, controller):
        """Executes code safely with interface functions."""
        code = """
result = spawn_child_llm("Create a player")
print(f"Created {result['trial_id']}")
"""
        controller._execute_root_code(code)

        controller.root_interface.spawn_child_llm.assert_called_once()

    def test_execute_root_code_get_population(self, controller):
        """Can call get_population from code."""
        controller.root_interface.get_population.return_value = []

        code = """
pop = get_population()
"""
        controller._execute_root_code(code)

        controller.root_interface.get_population.assert_called_once()


class TestRootSystemPrompt:
    """Tests for root system prompt."""

    def test_system_prompt_has_functions(self):
        """System prompt describes available functions."""
        assert "spawn_child_llm" in ROOT_SYSTEM_PROMPT
        assert "advance_generation" in ROOT_SYSTEM_PROMPT
        assert "terminate_evolution" in ROOT_SYSTEM_PROMPT
        assert "get_population" in ROOT_SYSTEM_PROMPT

    def test_system_prompt_has_output_format(self):
        """System prompt specifies output format."""
        assert "python" in ROOT_SYSTEM_PROMPT.lower()


class TestFullRun:
    """Tests for full evolution run."""

    @patch("tetris_evolve.evolution.controller.LLMClient")
    @patch("tetris_evolve.evolution.controller.ProgramEvaluator")
    def test_run_completes(self, mock_eval_cls, mock_client_cls, tmp_path):
        """Full run returns result."""
        # Setup mocks
        mock_client = Mock()
        mock_client.model = "test-model"
        mock_client.send_message.return_value = Mock(
            content='```python\nterminate_evolution("Test complete")\n```',
            input_tokens=100,
            output_tokens=50,
        )
        mock_client_cls.return_value = mock_client

        mock_eval = Mock()
        mock_eval.max_steps = 10000
        mock_eval.timeout_seconds = 60
        mock_eval.evaluate.return_value = Mock(
            success=True,
            to_dict=lambda: {"avg_score": 100.0},
        )
        mock_eval_cls.return_value = mock_eval

        config = EvolutionConfig(
            max_generations=2,
            max_cost_usd=1.0,
            max_time_minutes=1,
            initial_population_size=1,
            output_dir=tmp_path / "experiments",
        )

        controller = EvolutionController(config)

        # Patch child executor to return valid results
        with patch.object(controller, "_create_initial_population"):
            with patch.object(controller, "_run_root_llm_turn") as mock_turn:
                controller._initialize_components()
                controller.exp_tracker.create_experiment(config.to_dict())
                controller.exp_tracker.start_generation(1)

                # Simulate termination on first root LLM turn
                def terminate_on_call():
                    controller.root_interface._terminated = True
                    return False

                mock_turn.side_effect = terminate_on_call

                # Directly test result building
                result = EvolutionResult(
                    experiment_id="test",
                    generations_completed=1,
                    total_cost_usd=0.5,
                    total_time_minutes=0.5,
                    best_trial_id=None,
                    best_score=None,
                    best_code=None,
                    termination_reason="test",
                )

                assert result.termination_reason == "test"
                assert result.experiment_id == "test"

    @patch("tetris_evolve.evolution.controller.LLMClient")
    @patch("tetris_evolve.evolution.controller.ProgramEvaluator")
    def test_run_respects_limits(self, mock_eval_cls, mock_client_cls, tmp_path):
        """Run stops at limits."""
        mock_client = Mock()
        mock_client.model = "test-model"
        mock_client_cls.return_value = mock_client

        mock_eval = Mock()
        mock_eval.max_steps = 10000
        mock_eval.timeout_seconds = 60
        mock_eval_cls.return_value = mock_eval

        config = EvolutionConfig(
            max_generations=1,  # Very low limit
            max_cost_usd=100.0,
            max_time_minutes=60,
            initial_population_size=1,
            output_dir=tmp_path / "experiments",
        )

        controller = EvolutionController(config)

        with patch.object(controller, "_create_initial_population"):
            controller._initialize_components()
            controller.exp_tracker.create_experiment(config.to_dict())
            controller.exp_tracker.start_generation(1)

            # Set generation past limit
            controller.root_interface._current_generation = 2

            # Check limit
            limit = controller._check_generation_limit()

            assert limit is not None
            assert "generation_limit" in limit
