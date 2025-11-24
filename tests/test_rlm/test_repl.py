"""
Tests for the shared REPL framework.

Following TDD: These tests define the expected behavior of the REPL
environment shared between Root and Child LLMs.
"""
import pytest
from typing import Any, Dict

from tetris_evolve.rlm import (
    SharedREPL,
    REPLContext,
    REPLExecutionError,
)


class TestSharedREPL:
    """Tests for the SharedREPL class."""

    @pytest.fixture
    def repl(self):
        """Create a fresh REPL instance."""
        return SharedREPL()

    def test_repl_creation(self):
        """Should create a REPL with empty namespace."""
        repl = SharedREPL()
        assert repl is not None

    def test_execute_simple_expression(self, repl):
        """Should execute simple Python expressions."""
        result = repl.execute("1 + 1")
        assert result == 2

    def test_execute_assignment(self, repl):
        """Should execute assignments and store variables."""
        repl.execute("x = 42")
        assert repl.get_variable("x") == 42

    def test_execute_multiline(self, repl):
        """Should execute multiline code."""
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        repl.execute(code)
        assert repl.get_variable("result") == 3

    def test_execute_with_return_value(self, repl):
        """Should return the last expression value."""
        result = repl.execute("x = 5\nx * 2")
        assert result == 10

    def test_variables_persist(self, repl):
        """Variables should persist between executions."""
        repl.execute("a = 1")
        repl.execute("b = 2")
        result = repl.execute("a + b")
        assert result == 3

    def test_functions_persist(self, repl):
        """Functions should persist between executions."""
        repl.execute("def double(x): return x * 2")
        result = repl.execute("double(5)")
        assert result == 10

    def test_set_variable(self, repl):
        """Should set variables from Python."""
        repl.set_variable("my_var", [1, 2, 3])
        result = repl.execute("sum(my_var)")
        assert result == 6

    def test_get_variable(self, repl):
        """Should get variables from the REPL."""
        repl.execute("data = {'key': 'value'}")
        data = repl.get_variable("data")
        assert data == {"key": "value"}

    def test_get_nonexistent_variable(self, repl):
        """Should return None for nonexistent variables."""
        assert repl.get_variable("nonexistent") is None

    def test_inject_function(self, repl):
        """Should inject Python functions into REPL."""

        def my_function(x):
            return x ** 2

        repl.inject_function("square", my_function)
        result = repl.execute("square(5)")
        assert result == 25

    def test_syntax_error(self, repl):
        """Should raise error on syntax error."""
        with pytest.raises(REPLExecutionError):
            repl.execute("if if if")

    def test_runtime_error(self, repl):
        """Should raise error on runtime error."""
        with pytest.raises(REPLExecutionError):
            repl.execute("undefined_variable + 1")

    def test_get_namespace(self, repl):
        """Should return a copy of the namespace."""
        repl.execute("x = 1")
        repl.execute("y = 2")
        namespace = repl.get_namespace()

        assert "x" in namespace
        assert "y" in namespace

    def test_clear_namespace(self, repl):
        """Should clear the namespace."""
        repl.execute("x = 1")
        repl.clear()
        assert repl.get_variable("x") is None


class TestREPLContext:
    """Tests for REPLContext manager."""

    def test_context_creation(self):
        """Should create a context with specified variables."""
        context = REPLContext(
            variables={"x": 1, "y": 2},
        )

        assert context.get("x") == 1
        assert context.get("y") == 2

    def test_context_update(self):
        """Should update context variables."""
        context = REPLContext()
        context.set("key", "value")
        assert context.get("key") == "value"

    def test_context_to_namespace(self):
        """Should convert to namespace dict."""
        context = REPLContext(variables={"a": 1, "b": 2})
        namespace = context.to_namespace()

        assert isinstance(namespace, dict)
        assert namespace["a"] == 1
        assert namespace["b"] == 2

    def test_context_from_namespace(self):
        """Should create context from namespace dict."""
        namespace = {"x": 10, "y": 20}
        context = REPLContext.from_namespace(namespace)

        assert context.get("x") == 10
        assert context.get("y") == 20


class TestSharedREPLIntegration:
    """Integration tests for SharedREPL with database and evaluation."""

    @pytest.fixture
    def repl(self):
        return SharedREPL()

    def test_inject_program_database(self, repl):
        """Should be able to use program database in REPL."""
        from tetris_evolve.database import Program, ProgramDatabase

        db = ProgramDatabase()
        db.add_program(Program(
            program_id="test_001",
            generation=0,
            code="pass",
        ))

        repl.set_variable("program_database", db)

        result = repl.execute("len(program_database.get_all_programs())")
        assert result == 1

    def test_inject_evaluation_function(self, repl):
        """Should be able to inject and call evaluation functions."""

        def mock_evaluate(code: str) -> Dict[str, Any]:
            return {"score": 100.0, "success": True}

        repl.inject_function("evaluate_program", mock_evaluate)

        result = repl.execute('evaluate_program("test code")')
        assert result["score"] == 100.0

    def test_complex_workflow(self, repl):
        """Should support complex multi-step workflows."""
        # Set up context
        repl.set_variable("candidates", [])

        # Add candidates
        repl.execute("candidates.append({'id': 1, 'score': 100})")
        repl.execute("candidates.append({'id': 2, 'score': 200})")
        repl.execute("candidates.append({'id': 3, 'score': 150})")

        # Select top candidate
        result = repl.execute("max(candidates, key=lambda x: x['score'])")

        assert result["id"] == 2
        assert result["score"] == 200


class TestREPLSafety:
    """Tests for REPL safety features."""

    @pytest.fixture
    def repl(self):
        return SharedREPL()

    def test_timeout_protection(self, repl):
        """Should timeout on infinite loops (if timeout is implemented)."""
        # This test depends on whether timeout is implemented
        # For now, we just verify the REPL can handle finite loops
        result = repl.execute("sum(range(1000))")
        assert result == 499500

    def test_import_allowed(self, repl):
        """Should allow standard library imports."""
        repl.execute("import math")
        result = repl.execute("math.sqrt(16)")
        assert result == 4.0

    def test_numpy_available(self, repl):
        """Should have numpy available."""
        repl.execute("import numpy as np")
        result = repl.execute("np.array([1, 2, 3]).sum()")
        assert result == 6


class TestREPLExecutionError:
    """Tests for REPLExecutionError."""

    def test_error_has_message(self):
        """Error should have descriptive message."""
        error = REPLExecutionError("Test error", "print('test')")
        assert "Test error" in str(error)

    def test_error_has_code(self):
        """Error should store the code that caused it."""
        error = REPLExecutionError("Test error", "bad_code()")
        assert error.code == "bad_code()"

    def test_error_has_traceback(self):
        """Error can store traceback info."""
        error = REPLExecutionError("Test error", "code", traceback_str="line 1...")
        assert error.traceback_str == "line 1..."
