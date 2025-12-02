"""
REPL (Read-Eval-Print Loop) environment for tetris_evolve.

Provides a sandboxed Python execution environment for the Root LLM.
"""

import io
import sys
import traceback
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


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
}


@dataclass
class REPLResult:
    """Result of executing code in the REPL."""

    stdout: str
    stderr: str
    return_value: Any
    execution_time: float
    success: bool
    error: Optional[str] = None


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
    - Optional Evolution API injection
    """

    def __init__(self, api_functions: Optional[Dict[str, Callable]] = None):
        """
        Initialize the REPL environment.

        Args:
            api_functions: Optional dictionary of functions to inject into the namespace
        """
        self.globals = self._create_safe_globals()
        self.locals: Dict[str, Any] = {}
        self._api_functions = api_functions or {}

        # Inject API functions into globals
        for name, func in self._api_functions.items():
            self.globals[name] = func

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create the globals dictionary with safe builtins."""
        safe_builtins = SAFE_BUILTINS.copy()
        safe_builtins["__import__"] = _safe_import

        return {
            "__builtins__": safe_builtins,
            "__name__": "__repl__",
            "__doc__": None,
        }

    def inject_function(self, name: str, func: Callable) -> None:
        """
        Inject a function into the REPL namespace.

        Args:
            name: Name to use in the namespace
            func: The function to inject
        """
        self._api_functions[name] = func
        self.globals[name] = func

    def remove_function(self, name: str) -> None:
        """
        Remove a function from the REPL namespace.

        Args:
            name: Name of the function to remove
        """
        self._api_functions.pop(name, None)
        self.globals.pop(name, None)

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

        # Re-inject API functions
        for name, func in self._api_functions.items():
            self.globals[name] = func

    def get_state(self) -> Dict[str, Any]:
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
