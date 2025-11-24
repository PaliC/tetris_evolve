"""
Tests for Root LLM REPL functions.

These tests define the expected behavior of functions available to
the Root LLM in the shared REPL environment.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from tetris_evolve.root_llm.functions import (
    RootLLMFunctions,
    PerformanceAnalysis,
    HistoricalTrends,
    ResourceLimitError,
)
from tetris_evolve.database import Program, ProgramDatabase, ProgramMetrics
from tetris_evolve.environment import TetrisConfig
from tetris_evolve.rlm import SharedREPL


class TestRootLLMFunctionsInit:
    """Tests for RootLLMFunctions initialization."""

    def test_creation(self):
        """Should create RootLLMFunctions with required components."""
        db = ProgramDatabase()
        env_config = TetrisConfig()

        functions = RootLLMFunctions(
            program_database=db,
            env_config=env_config,
            max_generations=100,
            max_child_rllms_per_generation=50,
        )

        assert functions is not None
        assert functions.program_database is db
        assert functions.current_generation == 0

    def test_inject_into_repl(self):
        """Should inject all functions into REPL."""
        db = ProgramDatabase()
        env_config = TetrisConfig()
        repl = SharedREPL()

        functions = RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )
        functions.inject_into_repl(repl)

        # Check that functions are available
        assert repl.get_variable("spawn_rllm") is not None
        assert repl.get_variable("evaluate_program") is not None
        assert repl.get_variable("get_performance_analysis") is not None
        assert repl.get_variable("get_historical_trends") is not None
        assert repl.get_variable("advance_generation") is not None
        assert repl.get_variable("terminate_evolution") is not None
        assert repl.get_variable("modify_parameters") is not None

    def test_inject_context_variables(self):
        """Should inject context variables into REPL."""
        db = ProgramDatabase()
        env_config = TetrisConfig()
        repl = SharedREPL()

        functions = RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )
        functions.inject_into_repl(repl)

        # Check context variables
        assert repl.get_variable("program_database") is db
        assert repl.get_variable("current_generation") == 0


class TestEvaluateProgram:
    """Tests for evaluate_program function."""

    @pytest.fixture
    def functions(self):
        db = ProgramDatabase()
        env_config = TetrisConfig()
        return RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )

    def test_evaluate_valid_code(self, functions):
        """Should evaluate valid player code."""
        code = '''
import numpy as np

class TetrisPlayer:
    def select_action(self, observation):
        return 5  # Hard drop
    def reset(self):
        pass
'''
        result = functions.evaluate_program(code, num_games=2)

        assert "avg_score" in result
        assert "std_score" in result
        assert "games_played" in result
        assert result["games_played"] == 2

    def test_evaluate_invalid_code(self, functions):
        """Should handle invalid code gracefully."""
        code = "this is not valid python {"

        result = functions.evaluate_program(code, num_games=2)

        assert result["success"] is False
        assert "error" in result

    def test_evaluate_stores_in_database(self, functions):
        """Should optionally store results in database."""
        code = '''
class TetrisPlayer:
    def select_action(self, obs): return 5
    def reset(self): pass
'''
        result = functions.evaluate_program(
            code, num_games=2, store_result=True, program_id="test_prog"
        )

        program = functions.program_database.get_program("test_prog")
        assert program is not None
        assert program.metrics is not None


class TestGetPerformanceAnalysis:
    """Tests for get_performance_analysis function."""

    @pytest.fixture
    def functions_with_data(self):
        db = ProgramDatabase()
        env_config = TetrisConfig()

        # Add some programs with metrics
        for i in range(5):
            program = Program(
                program_id=f"prog_{i}",
                generation=0,
                code="",
                metrics=ProgramMetrics(
                    avg_score=100.0 * (i + 1),
                    std_score=10.0,
                    avg_lines_cleared=float(i),
                    games_played=10,
                )
            )
            db.add_program(program)

        return RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )

    def test_returns_analysis(self, functions_with_data):
        """Should return PerformanceAnalysis for generation."""
        analysis = functions_with_data.get_performance_analysis(generation=0)

        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.generation == 0
        assert analysis.num_programs == 5

    def test_analysis_has_statistics(self, functions_with_data):
        """Should include statistical summaries."""
        analysis = functions_with_data.get_performance_analysis(generation=0)

        assert analysis.avg_score > 0
        assert analysis.max_score > 0
        assert analysis.min_score > 0
        assert analysis.std_score >= 0

    def test_analysis_has_top_programs(self, functions_with_data):
        """Should include top programs."""
        analysis = functions_with_data.get_performance_analysis(generation=0)

        assert len(analysis.top_programs) > 0
        # Should be sorted by score descending
        scores = [p.metrics.avg_score for p in analysis.top_programs]
        assert scores == sorted(scores, reverse=True)

    def test_analysis_current_generation(self, functions_with_data):
        """Should default to current generation."""
        analysis = functions_with_data.get_performance_analysis()

        assert analysis.generation == 0


class TestGetHistoricalTrends:
    """Tests for get_historical_trends function."""

    @pytest.fixture
    def functions_with_history(self):
        db = ProgramDatabase()
        env_config = TetrisConfig()

        # Add programs across multiple generations
        for gen in range(3):
            for i in range(3):
                program = Program(
                    program_id=f"prog_g{gen}_{i}",
                    generation=gen,
                    code="",
                    metrics=ProgramMetrics(
                        avg_score=100.0 * (gen + 1) + i * 10,
                        std_score=10.0,
                        avg_lines_cleared=float(gen + i),
                        games_played=10,
                    )
                )
                db.add_program(program)

        funcs = RootLLMFunctions(
            program_database=db,
            env_config=env_config,
        )
        funcs.current_generation = 2
        return funcs

    def test_returns_trends(self, functions_with_history):
        """Should return HistoricalTrends."""
        trends = functions_with_history.get_historical_trends()

        assert isinstance(trends, HistoricalTrends)

    def test_trends_has_generation_data(self, functions_with_history):
        """Should have data for each generation."""
        trends = functions_with_history.get_historical_trends()

        assert len(trends.generations) == 3
        assert 0 in trends.generations
        assert 1 in trends.generations
        assert 2 in trends.generations

    def test_trends_has_improvement_rate(self, functions_with_history):
        """Should calculate improvement rate."""
        trends = functions_with_history.get_historical_trends()

        assert trends.improvement_rate is not None
        # Scores increase with generation, so improvement should be positive
        assert trends.improvement_rate > 0

    def test_trends_has_best_score_history(self, functions_with_history):
        """Should track best score over time."""
        trends = functions_with_history.get_historical_trends()

        assert len(trends.best_scores) == 3
        # Best scores should generally increase
        assert trends.best_scores[-1] >= trends.best_scores[0]


class TestAdvanceGeneration:
    """Tests for advance_generation function."""

    @pytest.fixture
    def functions(self):
        db = ProgramDatabase()
        env_config = TetrisConfig()

        # Add some programs
        for i in range(5):
            db.add_program(Program(
                program_id=f"prog_{i}",
                generation=0,
                code=f"code_{i}",
            ))

        return RootLLMFunctions(
            program_database=db,
            env_config=env_config,
            max_generations=10,
        )

    def test_advance_generation(self, functions):
        """Should advance to next generation."""
        selected = ["prog_0", "prog_1", "prog_2"]

        new_gen = functions.advance_generation(selected)

        assert new_gen == 1
        assert functions.current_generation == 1

    def test_advance_tracks_selected(self, functions):
        """Should track which programs were selected."""
        selected = ["prog_0", "prog_1"]

        functions.advance_generation(selected)

        # Selected programs should be accessible
        assert functions.selected_for_generation[1] == selected

    def test_advance_exceeds_max_raises(self, functions):
        """Should raise error when max generations exceeded."""
        functions.current_generation = 9  # At max - 1

        functions.advance_generation(["prog_0"])  # This is OK

        with pytest.raises(ResourceLimitError):
            functions.advance_generation(["prog_0"])  # This should fail


class TestTerminateEvolution:
    """Tests for terminate_evolution function."""

    @pytest.fixture
    def functions(self):
        db = ProgramDatabase()
        db.add_program(Program(
            program_id="best_prog",
            generation=5,
            code="best code",
            metrics=ProgramMetrics(
                avg_score=1000.0,
                std_score=50.0,
                avg_lines_cleared=100.0,
                games_played=100,
            )
        ))

        return RootLLMFunctions(
            program_database=db,
            env_config=TetrisConfig(),
        )

    def test_terminate_returns_summary(self, functions):
        """Should return evolution summary."""
        summary = functions.terminate_evolution(
            reason="Convergence achieved",
            best_program_id="best_prog"
        )

        assert summary["reason"] == "Convergence achieved"
        assert summary["best_program_id"] == "best_prog"
        assert "final_score" in summary

    def test_terminate_sets_flag(self, functions):
        """Should set terminated flag."""
        functions.terminate_evolution(
            reason="Test",
            best_program_id="best_prog"
        )

        assert functions.is_terminated is True

    def test_cannot_continue_after_termination(self, functions):
        """Should not allow operations after termination."""
        functions.terminate_evolution(
            reason="Test",
            best_program_id="best_prog"
        )

        with pytest.raises(RuntimeError):
            functions.advance_generation(["prog_0"])


class TestModifyParameters:
    """Tests for modify_parameters function."""

    @pytest.fixture
    def functions(self):
        return RootLLMFunctions(
            program_database=ProgramDatabase(),
            env_config=TetrisConfig(),
            episodes_per_evaluation=10,
        )

    def test_modify_episodes_per_eval(self, functions):
        """Should modify episodes_per_evaluation."""
        functions.modify_parameters({"episodes_per_evaluation": 50})

        assert functions.episodes_per_evaluation == 50

    def test_modify_returns_current_params(self, functions):
        """Should return current parameters."""
        result = functions.modify_parameters({"episodes_per_evaluation": 50})

        assert result["episodes_per_evaluation"] == 50

    def test_cannot_modify_hard_limits(self, functions):
        """Should not allow modifying hard limits."""
        with pytest.raises(ValueError):
            functions.modify_parameters({"max_generations": 1000})


class TestSpawnRLLM:
    """Tests for spawn_rllm function."""

    @pytest.fixture
    def functions(self):
        return RootLLMFunctions(
            program_database=ProgramDatabase(),
            env_config=TetrisConfig(),
            max_child_rllms_per_generation=5,
        )

    def test_spawn_rllm_calls_handler(self, functions):
        """Should call the registered RLLM handler."""
        mock_handler = Mock(return_value="generated code")
        functions.set_rllm_handler(mock_handler)

        result = functions.spawn_rllm("Generate improved code")

        mock_handler.assert_called_once()
        assert result == "generated code"

    def test_spawn_rllm_tracks_count(self, functions):
        """Should track number of RLLMs spawned."""
        mock_handler = Mock(return_value="code")
        functions.set_rllm_handler(mock_handler)

        functions.spawn_rllm("prompt 1")
        functions.spawn_rllm("prompt 2")

        assert functions.rllms_spawned_this_generation == 2

    def test_spawn_rllm_respects_limit(self, functions):
        """Should raise error when limit exceeded."""
        mock_handler = Mock(return_value="code")
        functions.set_rllm_handler(mock_handler)

        for _ in range(5):
            functions.spawn_rllm("prompt")

        with pytest.raises(ResourceLimitError):
            functions.spawn_rllm("one more")

    def test_spawn_rllm_resets_on_new_generation(self, functions):
        """Should reset count when generation advances."""
        mock_handler = Mock(return_value="code")
        functions.set_rllm_handler(mock_handler)

        for _ in range(5):
            functions.spawn_rllm("prompt")

        functions.advance_generation([])

        # Should be able to spawn again
        functions.spawn_rllm("prompt")
        assert functions.rllms_spawned_this_generation == 1


class TestPerformanceAnalysis:
    """Tests for PerformanceAnalysis dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        analysis = PerformanceAnalysis(
            generation=5,
            num_programs=10,
            avg_score=100.0,
            std_score=10.0,
            max_score=150.0,
            min_score=50.0,
            top_programs=[],
            improvement_from_prev=0.1,
        )

        d = analysis.to_dict()

        assert d["generation"] == 5
        assert d["num_programs"] == 10
        assert d["avg_score"] == 100.0


class TestHistoricalTrends:
    """Tests for HistoricalTrends dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        trends = HistoricalTrends(
            generations={0: {"avg": 100}, 1: {"avg": 150}},
            best_scores=[100, 150],
            avg_scores=[80, 120],
            improvement_rate=0.5,
            convergence_indicator=0.1,
        )

        d = trends.to_dict()

        assert len(d["generations"]) == 2
        assert d["improvement_rate"] == 0.5
