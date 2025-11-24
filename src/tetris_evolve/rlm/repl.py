"""
Shared REPL environment for Root and Child LLMs.

This module provides a Python REPL that is shared between the Root LLM
and all Child RLLMs, enabling efficient context sharing and communication.
"""
from typing import Any, Callable, Dict, Optional
import traceback
import sys
from io import StringIO


class REPLExecutionError(Exception):
    """Exception raised when REPL execution fails."""

    def __init__(
        self,
        message: str,
        code: str,
        traceback_str: Optional[str] = None
    ):
        super().__init__(message)
        self.code = code
        self.traceback_str = traceback_str

    def __str__(self):
        result = f"{self.args[0]}"
        if self.traceback_str:
            result += f"\n{self.traceback_str}"
        return result


class REPLContext:
    """
    Context manager for REPL variables.

    Provides a structured way to manage variables that should be
    available in the REPL environment.
    """

    def __init__(self, variables: Optional[Dict[str, Any]] = None):
        """
        Initialize context with optional variables.

        Args:
            variables: Initial variables to include
        """
        self._variables: Dict[str, Any] = variables.copy() if variables else {}

    def get(self, name: str, default: Any = None) -> Any:
        """Get a variable value."""
        return self._variables.get(name, default)

    def set(self, name: str, value: Any) -> None:
        """Set a variable value."""
        self._variables[name] = value

    def delete(self, name: str) -> None:
        """Delete a variable."""
        if name in self._variables:
            del self._variables[name]

    def to_namespace(self) -> Dict[str, Any]:
        """Convert to namespace dictionary."""
        return self._variables.copy()

    @classmethod
    def from_namespace(cls, namespace: Dict[str, Any]) -> "REPLContext":
        """Create context from namespace dictionary."""
        # Filter out built-in items
        filtered = {
            k: v for k, v in namespace.items()
            if not k.startswith("__")
        }
        return cls(variables=filtered)

    def __contains__(self, name: str) -> bool:
        return name in self._variables

    def keys(self):
        return self._variables.keys()

    def values(self):
        return self._variables.values()

    def items(self):
        return self._variables.items()


class SharedREPL:
    """
    Shared Python REPL environment.

    This REPL is shared between the Root LLM and all Child RLLMs,
    allowing them to share variables, functions, and context.

    Features:
    - Persistent namespace across executions
    - Variable injection from Python
    - Function injection
    - Error handling with context
    - Standard library imports available
    """

    def __init__(self, initial_namespace: Optional[Dict[str, Any]] = None):
        """
        Initialize the REPL with an optional initial namespace.

        Args:
            initial_namespace: Initial variables/functions to include
        """
        # Create namespace with common imports
        self._namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
        }

        # Add numpy by default (commonly used)
        try:
            import numpy as np
            self._namespace["np"] = np
            self._namespace["numpy"] = np
        except ImportError:
            pass

        # Add initial namespace
        if initial_namespace:
            self._namespace.update(initial_namespace)

        # Track execution history
        self._history: list = []

    def execute(self, code: str) -> Any:
        """
        Execute Python code in the REPL.

        Args:
            code: Python code to execute

        Returns:
            The result of the last expression, or None for statements

        Raises:
            REPLExecutionError: If execution fails
        """
        code = code.strip()
        if not code:
            return None

        # Record in history
        self._history.append(code)

        try:
            # Try to evaluate as expression first
            try:
                result = eval(code, self._namespace)
                return result
            except SyntaxError:
                # Not an expression, execute as statements
                pass

            # Check if the last line is an expression
            lines = code.split("\n")
            last_line = lines[-1].strip() if lines else ""

            # Execute all but possibly the last line
            if len(lines) > 1 or not self._is_expression(last_line):
                exec(code, self._namespace)

                # If last line looks like an expression, try to get its value
                if last_line and self._is_expression(last_line):
                    try:
                        return eval(last_line, self._namespace)
                    except:
                        pass

                return None
            else:
                exec(code, self._namespace)
                return None

        except SyntaxError as e:
            raise REPLExecutionError(
                f"Syntax error: {e}",
                code,
                traceback.format_exc()
            )
        except Exception as e:
            raise REPLExecutionError(
                f"Execution error: {type(e).__name__}: {e}",
                code,
                traceback.format_exc()
            )

    def _is_expression(self, code: str) -> bool:
        """Check if code is likely an expression (not a statement)."""
        code = code.strip()
        if not code:
            return False

        # Check for common statement keywords
        statement_starts = [
            "if ", "for ", "while ", "def ", "class ", "import ",
            "from ", "try:", "except", "with ", "raise ", "assert ",
            "return ", "yield ", "pass", "break", "continue",
        ]

        for start in statement_starts:
            if code.startswith(start):
                return False

        # Check for assignment
        if "=" in code and not any(op in code for op in ["==", "!=", "<=", ">=", "+=", "-=", "*=", "/="]):
            # Could be assignment
            parts = code.split("=")
            if len(parts) >= 2:
                left = parts[0].strip()
                # Check if left side is a valid variable name
                if left.isidentifier() or ("," in left) or ("[" in left):
                    return False

        return True

    def set_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the REPL namespace.

        Args:
            name: Variable name
            value: Variable value
        """
        self._namespace[name] = value

    def get_variable(self, name: str) -> Optional[Any]:
        """
        Get a variable from the REPL namespace.

        Args:
            name: Variable name

        Returns:
            Variable value or None if not found
        """
        return self._namespace.get(name)

    def inject_function(self, name: str, func: Callable) -> None:
        """
        Inject a Python function into the REPL namespace.

        Args:
            name: Function name in the REPL
            func: Python function to inject
        """
        self._namespace[name] = func

    def get_namespace(self) -> Dict[str, Any]:
        """
        Get a copy of the REPL namespace.

        Returns:
            Copy of namespace (excluding builtins)
        """
        return {
            k: v for k, v in self._namespace.items()
            if not k.startswith("__")
        }

    def clear(self) -> None:
        """Clear the REPL namespace (except builtins)."""
        keys_to_remove = [
            k for k in self._namespace.keys()
            if not k.startswith("__") and k not in ("np", "numpy")
        ]
        for key in keys_to_remove:
            del self._namespace[key]
        self._history.clear()

    def get_history(self) -> list:
        """Get execution history."""
        return self._history.copy()

    def apply_context(self, context: REPLContext) -> None:
        """
        Apply a context to the REPL namespace.

        Args:
            context: Context to apply
        """
        self._namespace.update(context.to_namespace())

    def extract_context(self) -> REPLContext:
        """
        Extract current namespace as a context.

        Returns:
            REPLContext with current namespace
        """
        return REPLContext.from_namespace(self._namespace)

    def capture_output(self, code: str) -> tuple:
        """
        Execute code and capture stdout/stderr.

        Args:
            code: Code to execute

        Returns:
            Tuple of (result, stdout, stderr)
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        result = None
        try:
            result = self.execute(code)
        finally:
            stdout_value = sys.stdout.getvalue()
            stderr_value = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return result, stdout_value, stderr_value
