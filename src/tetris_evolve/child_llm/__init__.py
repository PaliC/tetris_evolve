"""
Child LLM module: Child RLLM logic.

This module implements the Child Recursive LLMs (Depth=1) that generate code
mutations based on custom prompts from the Root LLM.

Usage:
    from tetris_evolve.child_llm import ChildLLMExecutor, CodeValidator

    executor = ChildLLMExecutor(llm_client, env_config)
    result = executor.execute("Generate an improved player")
"""
from .executor import (
    ChildLLMExecutor,
    CodeGenerator,
    CodeValidator,
    GenerationResult,
    ValidationResult,
    ValidationError,
    LLMClient,
)

__all__ = [
    "ChildLLMExecutor",
    "CodeGenerator",
    "CodeValidator",
    "GenerationResult",
    "ValidationResult",
    "ValidationError",
    "LLMClient",
]
