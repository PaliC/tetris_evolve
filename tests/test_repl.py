"""
Tests for pineapple_evolve.repl module.
"""

from pineapple_evolve import REPLEnvironment


class TestREPLEnvironment:
    """Tests for REPLEnvironment class."""

    def test_basic_execution(self):
        """Execute simple Python code."""
        repl = REPLEnvironment()

        result = repl.execute("x = 5 + 3")

        assert result.success is True
        assert result.error is None
        assert repl.locals["x"] == 8

    def test_stdout_capture(self):
        """Print statements captured."""
        repl = REPLEnvironment()

        result = repl.execute("print('Hello, World!')")

        assert result.success is True
        assert "Hello, World!" in result.stdout

    def test_stderr_capture(self):
        """Errors captured."""
        repl = REPLEnvironment()

        result = repl.execute("raise ValueError('test error')")

        assert result.success is False
        assert "ValueError" in result.stderr or "ValueError" in result.error

    def test_state_persistence(self):
        """Variables persist across executions."""
        repl = REPLEnvironment()

        repl.execute("x = 10")
        result = repl.execute("y = x * 2")

        assert result.success is True
        assert repl.locals["x"] == 10
        assert repl.locals["y"] == 20

    def test_import_capability(self):
        """Can import standard libraries."""
        repl = REPLEnvironment()

        result = repl.execute("import math; x = math.sqrt(16)")

        assert result.success is True
        assert repl.locals["x"] == 4.0

    def test_import_numpy(self):
        """Can import numpy."""
        repl = REPLEnvironment()

        result = repl.execute("import numpy as np; arr = np.array([1, 2, 3])")

        assert result.success is True
        assert list(repl.locals["arr"]) == [1, 2, 3]

    def test_error_handling(self):
        """Exceptions captured gracefully."""
        repl = REPLEnvironment()

        result = repl.execute("1 / 0")

        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_safe_builtins(self):
        """Dangerous builtins blocked."""
        repl = REPLEnvironment()

        # input() should be blocked/None
        _result = repl.execute("x = input")
        assert repl.locals["x"] is None

    def test_blocked_imports(self):
        """Dangerous modules blocked."""
        repl = REPLEnvironment()

        result = repl.execute("import os")

        assert result.success is False
        assert "ImportError" in result.error or "not allowed" in result.error

    def test_custom_functions(self):
        """Can define functions in REPL."""
        repl = REPLEnvironment()

        repl.execute("""
def square(x):
    return x ** 2
""")
        result = repl.execute("y = square(5)")

        assert result.success is True
        assert repl.locals["y"] == 25

    def test_expression_return_value(self):
        """Expression results are returned."""
        repl = REPLEnvironment()

        result = repl.execute("5 + 3")

        assert result.success is True
        assert result.return_value == 8

    def test_inject_function(self):
        """Functions can be injected into namespace."""
        repl = REPLEnvironment()

        def custom_func(x):
            return x * 2

        repl.inject_function("double", custom_func)

        result = repl.execute("y = double(5)")

        assert result.success is True
        assert repl.locals["y"] == 10

    def test_remove_function(self):
        """Functions can be removed from namespace."""
        repl = REPLEnvironment()

        def custom_func(x):
            return x * 2

        repl.inject_function("double", custom_func)
        repl.remove_function("double")

        result = repl.execute("double(5)")

        assert result.success is False
        assert "double" in str(result.error)

    def test_reset(self):
        """Reset clears state but preserves API functions."""
        repl = REPLEnvironment()

        def api_func():
            return 42

        repl.inject_function("api", api_func)
        repl.execute("x = 10")

        repl.reset()

        assert "x" not in repl.locals
        result = repl.execute("y = api()")
        assert result.success is True
        assert repl.locals["y"] == 42

    def test_get_state(self):
        """Get state returns user variables."""
        repl = REPLEnvironment()

        repl.execute("x = 10")
        repl.execute("y = 'hello'")
        repl.execute("_private = 5")

        state = repl.get_state()

        assert "x" in state
        assert "y" in state
        assert "_private" not in state

    def test_api_functions_at_init(self):
        """API functions can be provided at init."""

        def my_api():
            return "api result"

        repl = REPLEnvironment(api_functions={"my_api": my_api})

        result = repl.execute("x = my_api()")

        assert result.success is True
        assert repl.locals["x"] == "api result"

    def test_execution_time_tracked(self):
        """Execution time is tracked."""
        repl = REPLEnvironment()

        result = repl.execute("import time; time.sleep(0.01)")

        assert result.execution_time >= 0.01

    def test_multiline_code(self):
        """Multiline code executes correctly."""
        repl = REPLEnvironment()

        code = """
x = 1
y = 2
z = x + y
"""
        result = repl.execute(code)

        assert result.success is True
        assert repl.locals["z"] == 3
