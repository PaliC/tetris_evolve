"""
LLM client and prompt modules for tetris_evolve.

Provides:
- LLM client abstraction with provider-specific implementations
- Base types for messages, responses, and caching hints
- System prompt utilities with caching support
"""

from .base import (
    BaseLLMClient,
    CacheHint,
    Message,
    SystemPrompt,
)
from .client import (
    AnthropicLLMClient,
    LLMClient,
    LLMResponse,
    MockLLMClient,
    create_llm_client,
)
from .prompts import (
    CHILD_LLM_SYSTEM_PROMPT,
    ROOT_LLM_SYSTEM_PROMPT_DYNAMIC,
    ROOT_LLM_SYSTEM_PROMPT_STATIC,
    format_child_mutation_prompt,
    get_child_system_prompt_obj,
    get_root_system_prompt,
    get_root_system_prompt_obj,
    get_root_system_prompt_parts,
)

__all__ = [
    # Base types
    "BaseLLMClient",
    "CacheHint",
    "Message",
    "SystemPrompt",
    # Client implementations
    "AnthropicLLMClient",
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "create_llm_client",
    # Prompts
    "CHILD_LLM_SYSTEM_PROMPT",
    "ROOT_LLM_SYSTEM_PROMPT_DYNAMIC",
    "ROOT_LLM_SYSTEM_PROMPT_STATIC",
    "format_child_mutation_prompt",
    "get_child_system_prompt_obj",
    "get_root_system_prompt",
    "get_root_system_prompt_obj",
    "get_root_system_prompt_parts",
]
