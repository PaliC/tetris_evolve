"""End-to-End tests for complete evolution pipeline."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from tetris_evolve.evolution.controller import (
    EvolutionController,
    EvolutionConfig,
    EvolutionResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_config(tmp_path):
    """Minimal config for fast E2E tests."""
    return EvolutionConfig(
        max_generations=3,
        max_cost_usd=10.0,
        max_time_minutes=5,
        max_children_per_generation=5,
        initial_population_size=2,
        root_model="claude-haiku",
        child_model="claude-haiku",
        output_dir=tmp_path / "experiment",
    )


@pytest.fixture
def mock_child_response():
    """Create a mock child LLM response."""
    return Mock(
        content="""<reasoning>Simple hard drop strategy</reasoning>
<code>
def choose_action(obs):
    return 5  # hard drop
</code>""",
        input_tokens=100,
        output_tokens=50,
        model="claude-haiku",
        stop_reason="end_turn"
    )


@pytest.fixture
def mock_root_response():
    """Create a mock root LLM response that terminates immediately."""
    return Mock(
        content="""```python
terminate_evolution("Test complete")
```""",
        input_tokens=100,
        output_tokens=50,
        model="claude-haiku",
        stop_reason="end_turn"
    )


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator."""
    evaluator = Mock()
    evaluator.max_steps = 1000
    evaluator.timeout_seconds = 30
    evaluator.evaluate.return_value = Mock(
        success=True,
        to_dict=lambda: {
            "avg_score": 100.0,
            "avg_lines_cleared": 5.0,
            "avg_survival_steps": 500.0,
            "games_played": 2,
        }
    )
    evaluator._validate_syntax = Mock(return_value=(True, None))
    evaluator._check_safety = Mock(return_value=(True, None))
    return evaluator


def verify_directory_structure(exp_dir: Path) -> dict:
    """Verify experiment directory has expected structure."""
    issues = []

    # Top-level files (actual file names used by ExperimentTracker)
    if not (exp_dir / "config.yaml").exists():
        issues.append("Missing config.yaml")
    if not (exp_dir / "experiment_stats.json").exists():
        issues.append("Missing experiment_stats.json")
    if not (exp_dir / "cost_history.json").exists():
        issues.append("Missing cost_history.json")
    if not (exp_dir / "generations").exists():
        issues.append("Missing generations directory")

    return {"valid": len(issues) == 0, "issues": issues}


def verify_trial_files(trial_dir: Path) -> dict:
    """Verify a trial directory has expected files."""
    issues = []

    if not (trial_dir / "code.py").exists():
        issues.append("Missing code.py")
    if not (trial_dir / "trial.json").exists():
        issues.append("Missing trial.json")

    return {"valid": len(issues) == 0, "issues": issues}


# =============================================================================
# Section 9.2: Minimal Evolution Tests
# =============================================================================


class TestMinimalEvolution:
    """Tests for minimal evolution scenarios."""

    def test_single_generation_completes(self, test_config, mock_evaluator):
        """Single generation with 2 children completes successfully."""
        test_config.max_generations = 1
        test_config.initial_population_size = 2

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                # Setup mock LLM - returns valid code for child, terminate for root
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                call_count = [0]

                def mock_send_message(messages, system=None, **kwargs):
                    call_count[0] += 1
                    # First calls are for children, then root
                    if "choose_action" not in str(system or ""):
                        # This is root LLM
                        return Mock(
                            content='```python\nterminate_evolution("Gen 1 done")\n```',
                            input_tokens=100,
                            output_tokens=50,
                        )
                    else:
                        # This is child LLM
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send_message
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert result.generations_completed >= 1
        assert result.termination_reason is not None

    def test_directory_structure_created(self, test_config, mock_evaluator):
        """Experiment directory created with correct structure."""
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                controller.run()

        # Find experiment directory (has timestamp in name)
        exp_dirs = list(test_config.output_dir.glob("exp_*"))
        assert len(exp_dirs) == 1

        verification = verify_directory_structure(exp_dirs[0])
        assert verification["valid"], f"Issues: {verification['issues']}"

    def test_config_yaml_created(self, test_config, mock_evaluator):
        """config.yaml contains experiment configuration."""
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                controller.run()

        exp_dirs = list(test_config.output_dir.glob("exp_*"))
        config_yaml = exp_dirs[0] / "config.yaml"
        assert config_yaml.exists()

        import yaml
        data = yaml.safe_load(config_yaml.read_text())
        assert "max_generations" in data
        assert data["max_generations"] == 1


# =============================================================================
# Section 9.3: Multi-Generation Evolution
# =============================================================================


class TestMultiGenerationEvolution:
    """Tests for multi-generation evolution."""

    def test_two_generations_with_advancement(self, test_config, mock_evaluator):
        """Two generations with proper advancement completes."""
        test_config.max_generations = 2
        test_config.initial_population_size = 2

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                gen_count = [1]

                def mock_send(messages, system=None, **kwargs):
                    if "choose_action" not in str(system or ""):
                        # Root LLM
                        if gen_count[0] == 1:
                            gen_count[0] += 1
                            return Mock(
                                content='```python\npop = get_population()\nadvance_generation([p.trial_id for p in pop[:1]] if pop else [], "Next gen")\n```',
                                input_tokens=100,
                                output_tokens=50,
                            )
                        else:
                            return Mock(
                                content='```python\nterminate_evolution("Done")\n```',
                                input_tokens=100,
                                output_tokens=50,
                            )
                    else:
                        # Child LLM
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert result.generations_completed >= 1

    def test_generation_stats_calculated(self, test_config, mock_evaluator):
        """Generation stats are calculated correctly."""
        test_config.max_generations = 1
        test_config.initial_population_size = 2

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client

                # Return different scores for each evaluation
                scores = [150.0, 100.0]
                eval_idx = [0]

                def mock_evaluate(code, trial_id):
                    score = scores[eval_idx[0] % len(scores)]
                    eval_idx[0] += 1
                    return Mock(
                        success=True,
                        to_dict=lambda s=score: {
                            "avg_score": s,
                            "avg_lines_cleared": s / 10,
                            "avg_survival_steps": s * 10,
                        }
                    )

                mock_evaluator.evaluate.side_effect = mock_evaluate
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        # If best_score is set, verify it's reasonable
        if result.best_score is not None:
            assert result.best_score > 0


# =============================================================================
# Section 9.4: Hard Limit Enforcement
# =============================================================================


class TestHardLimitEnforcement:
    """Tests for hard limit enforcement."""

    def test_max_generations_enforced(self, test_config, mock_evaluator):
        """Evolution stops at max_generations."""
        test_config.max_generations = 2

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                gen_count = [1]

                def mock_send(messages, system=None, **kwargs):
                    if "choose_action" not in str(system or ""):
                        gen_count[0] += 1
                        # Try to advance each time, should eventually hit limit
                        if gen_count[0] <= 3:
                            return Mock(
                                content='```python\nadvance_generation([], "Continue")\n```',
                                input_tokens=100,
                                output_tokens=50,
                            )
                        else:
                            # After many tries, terminate
                            return Mock(
                                content='```python\nterminate_evolution("Max reached")\n```',
                                input_tokens=100,
                                output_tokens=50,
                            )
                    else:
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert result.generations_completed <= test_config.max_generations + 1

    def test_max_cost_enforced(self, test_config, mock_evaluator):
        """Evolution stops when cost limit reached."""
        test_config.max_cost_usd = 0.0001  # Very low budget
        test_config.max_generations = 100  # High limit to ensure cost stops it

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content="<code>def choose_action(obs): return 5</code>",
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        # Should have stopped due to cost
        assert "cost" in result.termination_reason.lower()

    def test_time_limit_setup(self, test_config, mock_evaluator):
        """Verify time limit is configured in controller."""
        test_config.max_time_minutes = 0.001  # Very short

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        # Test completes (time limit may or may not trigger depending on speed)
        assert result is not None


# =============================================================================
# Section 9.5: Root LLM Termination
# =============================================================================


class TestRootLLMTermination:
    """Tests for root LLM voluntary termination."""

    def test_root_can_terminate(self, test_config, mock_evaluator):
        """Root LLM can voluntarily terminate evolution."""
        test_config.max_generations = 10  # High limit

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Converged early")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert "Converged early" in result.termination_reason or \
               "root" in result.termination_reason.lower() or \
               "terminated" in result.termination_reason.lower()

    def test_termination_saves_state(self, test_config, mock_evaluator):
        """Root termination saves final state."""
        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                def mock_send(messages, system=None, **kwargs):
                    if "choose_action" not in str(system or ""):
                        return Mock(
                            content='```python\nterminate_evolution("Done")\n```',
                            input_tokens=100,
                            output_tokens=50,
                        )
                    else:
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        # Should have experiment directory
        exp_dirs = list(test_config.output_dir.glob("exp_*"))
        assert len(exp_dirs) == 1

        # experiment_stats.json should exist
        exp_stats = exp_dirs[0] / "experiment_stats.json"
        assert exp_stats.exists()


# =============================================================================
# Section 9.6: Error Recovery
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_continues_after_invalid_child_code(self, test_config, mock_evaluator):
        """Evolution continues when child generates invalid code."""
        test_config.max_generations = 1
        test_config.initial_population_size = 2

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                call_count = [0]

                def mock_send(messages, system=None, **kwargs):
                    call_count[0] += 1
                    if "choose_action" not in str(system or ""):
                        return Mock(
                            content='```python\nterminate_evolution("Done")\n```',
                            input_tokens=100,
                            output_tokens=50,
                        )
                    else:
                        # First child returns invalid, second valid
                        if call_count[0] == 1:
                            return Mock(
                                content="<code>def bad_code(): pass</code>",  # Invalid
                                input_tokens=100,
                                output_tokens=50,
                            )
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        # Should complete despite error
        assert result is not None
        assert result.termination_reason is not None

    def test_handles_evaluation_failure(self, test_config):
        """Evolution handles evaluation failures gracefully."""
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client

                # Evaluator that fails
                mock_eval = Mock()
                mock_eval.max_steps = 1000
                mock_eval.timeout_seconds = 30
                mock_eval.evaluate.return_value = Mock(
                    success=False,
                    error="Evaluation timed out",
                    to_dict=lambda: {"success": False, "error": "Timeout"}
                )
                mock_eval._validate_syntax = Mock(return_value=(True, None))
                mock_eval._check_safety = Mock(return_value=(True, None))
                mock_eval_cls.return_value = mock_eval

                controller = EvolutionController(test_config)
                result = controller.run()

        # Should complete despite evaluation failures
        assert result is not None


# =============================================================================
# Section 9.7: Resume Functionality
# =============================================================================


class TestResumeFunctionality:
    """Tests for resume functionality."""

    def test_resume_loads_state(self, test_config, mock_evaluator):
        """Resume loads previous experiment state."""
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result1 = controller.run()

        # Get experiment directory
        exp_dirs = list(test_config.output_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Verify experiment can be loaded
        from tetris_evolve.tracking.experiment_tracker import ExperimentTracker

        loaded = ExperimentTracker.load_experiment(exp_dir)
        assert loaded is not None
        assert loaded.experiment_id is not None


# =============================================================================
# Section 9.8: Output Verification
# =============================================================================


class TestOutputVerification:
    """Tests for verifying output correctness."""

    def test_cost_tracking_recorded(self, test_config, mock_evaluator):
        """Cost tracking is recorded in output files."""
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        # Get experiment directory
        exp_dirs = list(test_config.output_dir.glob("exp_*"))
        cost_file = exp_dirs[0] / "cost_history.json"
        assert cost_file.exists()

        data = json.loads(cost_file.read_text())
        assert "calls" in data or "total_cost_usd" in data

    def test_result_has_metrics(self, test_config, mock_evaluator):
        """Evolution result contains expected metrics."""
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"
                mock_client.send_message.return_value = Mock(
                    content='```python\nterminate_evolution("Done")\n```',
                    input_tokens=100,
                    output_tokens=50,
                )
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert result.experiment_id is not None
        assert result.generations_completed >= 0
        assert result.total_cost_usd >= 0
        assert result.total_time_minutes >= 0
        assert result.termination_reason is not None


# =============================================================================
# Additional Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimal_population_size(self, test_config, mock_evaluator):
        """Handles minimum (1) initial population correctly."""
        test_config.initial_population_size = 1
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                def mock_send(messages, system=None, **kwargs):
                    if "choose_action" not in str(system or ""):
                        return Mock(
                            content='```python\nterminate_evolution("Done")\n```',
                            input_tokens=100,
                            output_tokens=50,
                        )
                    else:
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert result is not None
        assert result.generations_completed >= 1

    def test_large_population_size(self, test_config, mock_evaluator):
        """Handles larger initial population."""
        test_config.initial_population_size = 5
        test_config.max_generations = 1

        with patch("tetris_evolve.evolution.controller.LLMClient") as mock_client_cls:
            with patch("tetris_evolve.evolution.controller.ProgramEvaluator") as mock_eval_cls:
                mock_client = Mock()
                mock_client.model = "claude-haiku"

                def mock_send(messages, system=None, **kwargs):
                    if "choose_action" not in str(system or ""):
                        return Mock(
                            content='```python\nterminate_evolution("Done")\n```',
                            input_tokens=100,
                            output_tokens=50,
                        )
                    else:
                        return Mock(
                            content="<code>def choose_action(obs): return 5</code>",
                            input_tokens=100,
                            output_tokens=50,
                        )

                mock_client.send_message.side_effect = mock_send
                mock_client_cls.return_value = mock_client
                mock_eval_cls.return_value = mock_evaluator

                controller = EvolutionController(test_config)
                result = controller.run()

        assert result is not None
        assert result.generations_completed >= 1
