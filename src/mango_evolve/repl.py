"""
REPL (Read-Eval-Print Loop) environment for mango_evolve.

Provides a sandboxed Python execution environment for the Root LLM.
"""

import io
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

# Safe subset of builtins allowed in the REPL
SAFE_BUILTINS = {
    # Types and constructors
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
    "type": type,
    "object": object,
    # Iteration
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "iter": iter,
    "next": next,
    # Math and comparison
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    # String/sequence operations
    "len": len,
    "sorted": sorted,
    "all": all,
    "any": any,
    "repr": repr,
    "chr": chr,
    "ord": ord,
    "format": format,
    "hash": hash,
    "id": id,
    "slice": slice,
    # Attribute access
    "getattr": getattr,
    "setattr": setattr,
    "hasattr": hasattr,
    "delattr": delattr,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    # I/O (limited)
    "print": print,
    "input": None,  # Disabled - would block
    # Import (limited, see below)
    "__import__": __import__,
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    # Boolean constants
    "True": True,
    "False": False,
    "None": None,
}

# Allowed modules for import
ALLOWED_MODULES = {
    "math",
    "random",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "copy",
    "time",
    "datetime",
    "numpy",
    "np",  # alias
    "scipy",
}


@dataclass
class REPLResult:
    """Result of executing code in the REPL."""

    stdout: str
    stderr: str
    return_value: Any
    execution_time: float
    success: bool
    error: str | None = None


def _safe_import(name: str, globals_dict=None, locals_dict=None, fromlist=(), level=0):
    """
    Restricted import function that only allows safe modules.
    """
    # Handle numpy alias
    if name == "np":
        name = "numpy"

    # Check if module is allowed
    base_module = name.split(".")[0]
    if base_module not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the REPL environment")

    return __import__(name, globals_dict, locals_dict, fromlist, level)


class REPLEnvironment:
    """
    A sandboxed Python execution environment.

    Provides:
    - Safe subset of builtins
    - Restricted imports
    - State persistence between executions
    - stdout/stderr capture
    - Optional Evolution API injection (functions and variables)
    """

    def __init__(
        self,
        namespace: dict[str, Any] | None = None,
        api_functions: dict[str, Callable] | None = None,  # Backward compat
    ):
        """
        Initialize the REPL environment.

        Args:
            namespace: Optional dictionary of functions and variables to inject.
                       This includes both API functions (spawn_child_llm, etc.)
                       and variables (trials, etc.).
            api_functions: Deprecated - use namespace instead. Kept for backward
                          compatibility.
        """
        self.globals = self._create_safe_globals()
        self.locals: dict[str, Any] = {}

        # Handle backward compatibility: api_functions is deprecated alias
        if api_functions is not None and namespace is None:
            namespace = api_functions

        self._namespace = namespace or {}

        # Inject namespace items (functions and variables) into globals
        for name, value in self._namespace.items():
            self.globals[name] = value

    def _create_safe_globals(self) -> dict[str, Any]:
        """Create the globals dictionary with safe builtins."""
        safe_builtins = SAFE_BUILTINS.copy()
        safe_builtins["__import__"] = _safe_import

        return {
            "__builtins__": safe_builtins,
            "__name__": "__repl__",
            "__doc__": None,
        }

    def inject(self, name: str, value: Any) -> None:
        """
        Inject a function or variable into the REPL namespace.

        Args:
            name: Name to use in the namespace
            value: The function or value to inject
        """
        self._namespace[name] = value
        self.globals[name] = value

    def inject_function(self, name: str, func: Callable) -> None:
        """
        Inject a function into the REPL namespace.

        Args:
            name: Name to use in the namespace
            func: The function to inject

        Note: This is an alias for inject() for backward compatibility.
        """
        self.inject(name, func)

    def remove(self, name: str) -> None:
        """
        Remove a function or variable from the REPL namespace.

        Args:
            name: Name of the item to remove
        """
        self._namespace.pop(name, None)
        self.globals.pop(name, None)

    def remove_function(self, name: str) -> None:
        """
        Remove a function from the REPL namespace.

        Args:
            name: Name of the function to remove

        Note: This is an alias for remove() for backward compatibility.
        """
        self.remove(name)

    def execute(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with stdout, stderr, timing, and success info
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        return_value = None
        error = None
        success = True

        start_time = time.time()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # First, try to compile as expression to get a return value
                try:
                    compiled = compile(code, "<repl>", "eval")
                    return_value = eval(compiled, self.globals, self.locals)
                except SyntaxError:
                    # Not an expression, execute as statements
                    compiled = compile(code, "<repl>", "exec")
                    exec(compiled, self.globals, self.locals)

        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {str(e)}"
            # Also capture the traceback to stderr
            tb = traceback.format_exc()
            stderr_capture.write(tb)

        execution_time = time.time() - start_time

        return REPLResult(
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            return_value=return_value,
            execution_time=execution_time,
            success=success,
            error=error,
        )

    def reset(self) -> None:
        """Reset the REPL state, clearing all local variables."""
        self.locals.clear()
        self.globals = self._create_safe_globals()

        # Re-inject namespace items (functions and variables)
        for name, value in self._namespace.items():
            self.globals[name] = value

    def get_state(self) -> dict[str, Any]:
        """
        Get a snapshot of user-defined variables.

        Returns:
            Dictionary of variable names to their string representations
        """
        state = {}
        for name, value in self.locals.items():
            if not name.startswith("_"):
                try:
                    state[name] = repr(value)
                except Exception:
                    state[name] = "<unable to repr>"
        return state
