"""
RLM module: Recursive LLM framework.

This module implements the core Recursive LLM framework with shared REPL environment
for efficient context passing between Root and Child LLMs.

Usage:
    from tetris_evolve.rlm import SharedREPL, REPLContext

    repl = SharedREPL()
    repl.set_variable("data", [1, 2, 3])
    result = repl.execute("sum(data)")
"""
from .repl import (
    SharedREPL,
    REPLContext,
    REPLExecutionError,
)

__all__ = [
    "SharedREPL",
    "REPLContext",
    "REPLExecutionError",
]
