# OpenRouter Integration Plan

## Overview

Add OpenRouter as an LLM provider, enabling access to 100+ models (Gemini, Claude, Llama, Mistral, etc.) through a single OpenAI-compatible API.

**Why OpenRouter?**
- Uses OpenAI SDK format (well-established, battle-tested)
- Single integration → access to many providers
- Unified billing and rate limiting
- Simple API key authentication

---

## Architecture Design

### Provider Abstraction Pattern

```
llm/
├── __init__.py
├── client.py          # Factory + LLMResponse (keep existing)
├── prompts.py         # Keep as-is
└── providers/
    ├── __init__.py
    ├── base.py        # Abstract base class
    ├── anthropic.py   # Refactored from current client.py
    └── openrouter.py  # NEW: OpenRouter implementation
```

### Provider Interface

```python
# llm/providers/base.py
from abc import ABC, abstractmethod
from typing import Any
from ..client import LLMResponse

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @property
    @abstractmethod
    def supports_caching(self) -> bool:
        """Whether this provider supports prompt caching."""
        pass
```

---

## Implementation Steps

### Step 1: Add OpenAI SDK Dependency

**File: `pyproject.toml`**
```toml
dependencies = [
    "anthropic>=0.18.0",
    "openai>=1.0.0",  # NEW - for OpenRouter
    # ... rest unchanged
]
```

### Step 2: Extend Configuration

**File: `config.py`**

Add `provider` field to `LLMConfig`:

```python
@dataclass
class LLMConfig:
    """Configuration for an LLM (root or child)."""

    model: str
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float
    provider: str = "anthropic"  # NEW: "anthropic" or "openrouter"
    max_iterations: int | None = None
```

**Example YAML config for OpenRouter:**
```yaml
root_llm:
  provider: "openrouter"
  model: "google/gemini-2.0-flash-001"  # OpenRouter model ID
  cost_per_million_input_tokens: 0.10
  cost_per_million_output_tokens: 0.40
  max_iterations: 30

child_llm:
  provider: "openrouter"
  model: "google/gemini-2.0-flash-001"
  cost_per_million_input_tokens: 0.10
  cost_per_million_output_tokens: 0.40
```

### Step 3: Create Provider Base Class

**File: `llm/providers/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    call_id: str
    stop_reason: str | None = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        cost_tracker: "CostTracker",
        llm_type: str,
        max_retries: int = 3,
    ):
        self.model = model
        self.cost_tracker = cost_tracker
        self.llm_type = llm_type
        self.max_retries = max_retries

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @property
    def supports_caching(self) -> bool:
        """Whether this provider supports prompt caching."""
        return False
```

### Step 4: Refactor Anthropic Provider

**File: `llm/providers/anthropic.py`**

Move existing `LLMClient` code here, inherit from `BaseLLMProvider`:

```python
import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from .base import BaseLLMProvider, LLMResponse

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider with prompt caching support."""

    def __init__(self, model: str, cost_tracker, llm_type: str, max_retries: int = 3):
        super().__init__(model, cost_tracker, llm_type, max_retries)
        self._client = anthropic.Anthropic()

    @property
    def supports_caching(self) -> bool:
        return True

    def generate(self, messages, system=None, max_tokens=4096, temperature=0.7, enable_caching=True) -> LLMResponse:
        # ... existing implementation from client.py
```

### Step 5: Create OpenRouter Provider

**File: `llm/providers/openrouter.py`**

```python
import os
import uuid
from typing import Any

from openai import OpenAI
from openai import RateLimitError, APIConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseLLMProvider, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API provider.

    Uses OpenAI SDK with custom base_url for OpenRouter.
    Supports 100+ models including Gemini, Claude, Llama, Mistral, etc.

    Environment variable: OPENROUTER_API_KEY
    """

    def __init__(
        self,
        model: str,
        cost_tracker,
        llm_type: str,
        max_retries: int = 3,
    ):
        super().__init__(model, cost_tracker, llm_type, max_retries)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @property
    def supports_caching(self) -> bool:
        # OpenRouter doesn't support Anthropic-style prompt caching
        return False

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        enable_caching: bool = True,  # Ignored for OpenRouter
    ) -> LLMResponse:
        """Generate a response using OpenRouter API."""

        # Check budget before making the call
        self.cost_tracker.raise_if_over_budget()

        call_id = str(uuid.uuid4())

        # Build messages list with system prompt
        api_messages = []

        # Handle system prompt (convert to first message if present)
        if system:
            system_text = system if isinstance(system, str) else self._extract_system_text(system)
            api_messages.append({"role": "system", "content": system_text})

        # Add conversation messages
        for msg in messages:
            content = msg.get("content", "")
            # Handle Anthropic-style content blocks
            if isinstance(content, list):
                content = self._extract_text_from_blocks(content)
            api_messages.append({"role": msg["role"], "content": content})

        # Make the API call with retries
        response = self._call_with_retry(
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract content
        content = response.choices[0].message.content or ""

        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Record usage
        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_type=self.llm_type,
            call_id=call_id,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            call_id=call_id,
            stop_reason=response.choices[0].finish_reason,
        )

    def _extract_system_text(self, system: list[dict[str, Any]]) -> str:
        """Extract text from Anthropic-style system content blocks."""
        texts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    def _extract_text_from_blocks(self, blocks: list[dict[str, Any]]) -> str:
        """Extract text from content blocks."""
        texts = []
        for block in blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    def _call_with_retry(self, messages: list, max_tokens: int, temperature: float):
        """Make API call with retry logic."""

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
            reraise=True,
        )
        def _make_call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return _make_call()
```

### Step 6: Create Provider Factory

**File: `llm/client.py`** (updated)

```python
from .providers.anthropic import AnthropicProvider
from .providers.openrouter import OpenRouterProvider
from .providers.base import LLMResponse, BaseLLMProvider

# Re-export for backwards compatibility
__all__ = ["LLMResponse", "create_llm_client", "LLMClient", "MockLLMClient"]


def create_llm_client(
    provider: str,
    model: str,
    cost_tracker,
    llm_type: str,
    max_retries: int = 3,
) -> BaseLLMProvider:
    """
    Factory function to create an LLM client for the specified provider.

    Args:
        provider: "anthropic" or "openrouter"
        model: Model identifier
        cost_tracker: CostTracker instance
        llm_type: "root" or "child"
        max_retries: Max retries on transient errors

    Returns:
        LLM client instance

    Raises:
        ValueError: If provider is unknown
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(providers.keys())}")

    return providers[provider](
        model=model,
        cost_tracker=cost_tracker,
        llm_type=llm_type,
        max_retries=max_retries,
    )


# Keep LLMClient as alias for backwards compatibility
LLMClient = AnthropicProvider
```

### Step 7: Update Root LLM Orchestrator

**File: `root_llm.py`**

Change client instantiation to use factory:

```python
# Before:
from .llm.client import LLMClient
self.root_llm = LLMClient(
    model=config.root_llm.model,
    cost_tracker=self.cost_tracker,
    llm_type="root",
)

# After:
from .llm.client import create_llm_client
self.root_llm = create_llm_client(
    provider=config.root_llm.provider,
    model=config.root_llm.model,
    cost_tracker=self.cost_tracker,
    llm_type="root",
)
```

### Step 8: Update Parallel Worker

**File: `parallel_worker.py`**

The parallel worker creates clients directly in worker processes. Update to use provider factory:

```python
# Add provider to worker args tuple
# (prompt, parent_id, model, evaluator_kwargs, max_tokens, temperature,
#  trial_id, generation, experiment_dir, system_prompt, provider)

def child_worker(args: tuple) -> dict[str, Any]:
    # Extract provider from args
    provider = args[10] if len(args) > 10 else "anthropic"

    # Create client based on provider
    if provider == "openrouter":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        # Use OpenAI SDK format for call
    else:
        client = anthropic.Anthropic()
        # Use existing Anthropic format
```

---

## Configuration Examples

### Anthropic Claude (existing behavior)

```yaml
root_llm:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  cost_per_million_input_tokens: 3.0
  cost_per_million_output_tokens: 15.0
  max_iterations: 30
```

### Google Gemini via OpenRouter

```yaml
root_llm:
  provider: "openrouter"
  model: "google/gemini-2.0-flash-001"
  cost_per_million_input_tokens: 0.10
  cost_per_million_output_tokens: 0.40
  max_iterations: 30
```

### DeepSeek via OpenRouter

```yaml
root_llm:
  provider: "openrouter"
  model: "deepseek/deepseek-chat"
  cost_per_million_input_tokens: 0.14
  cost_per_million_output_tokens: 0.28
  max_iterations: 30
```

### Llama via OpenRouter

```yaml
root_llm:
  provider: "openrouter"
  model: "meta-llama/llama-3.3-70b-instruct"
  cost_per_million_input_tokens: 0.40
  cost_per_million_output_tokens: 0.40
  max_iterations: 30
```

---

## Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `ANTHROPIC_API_KEY` | anthropic | Yes (if using) |
| `OPENROUTER_API_KEY` | openrouter | Yes (if using) |

---

## Files to Create/Modify Summary

### Create (new files)
| File | Description |
|------|-------------|
| `src/tetris_evolve/llm/providers/__init__.py` | Provider module init |
| `src/tetris_evolve/llm/providers/base.py` | Abstract base class |
| `src/tetris_evolve/llm/providers/anthropic.py` | Anthropic provider |
| `src/tetris_evolve/llm/providers/openrouter.py` | OpenRouter provider |

### Modify (existing files)
| File | Changes |
|------|---------|
| `pyproject.toml` | Add `openai>=1.0.0` dependency |
| `src/tetris_evolve/config.py` | Add `provider` field to `LLMConfig` |
| `src/tetris_evolve/llm/client.py` | Add factory, keep backwards compat |
| `src/tetris_evolve/root_llm.py` | Use factory for client creation |
| `src/tetris_evolve/parallel_worker.py` | Add provider support |
| `src/tetris_evolve/evolution_api.py` | Pass provider to parallel workers |

### Tests to Add
| File | Description |
|------|-------------|
| `tests/test_openrouter_provider.py` | Unit tests for OpenRouter |
| `tests/test_provider_factory.py` | Factory function tests |

---

## Implementation Order

1. **Add OpenAI dependency** to `pyproject.toml`
2. **Create provider abstraction** (`providers/base.py`)
3. **Refactor Anthropic provider** (`providers/anthropic.py`)
4. **Implement OpenRouter provider** (`providers/openrouter.py`)
5. **Update config** to add `provider` field
6. **Create factory** in `client.py`
7. **Update `root_llm.py`** to use factory
8. **Update `parallel_worker.py`** for multi-provider support
9. **Update `evolution_api.py`** to pass provider info
10. **Add tests**
11. **Create example configs** for different providers

---

## Notes

### Caching Behavior
- **Anthropic**: Supports ephemeral prompt caching (90% discount on cache reads)
- **OpenRouter**: No provider-specific caching (standard pricing)

The `enable_caching` parameter will be ignored for OpenRouter but kept for API compatibility.

### Token Counting
Both providers return token counts in their responses:
- Anthropic: `response.usage.input_tokens`, `response.usage.output_tokens`
- OpenRouter: `response.usage.prompt_tokens`, `response.usage.completion_tokens`

The provider implementations normalize these to a common `LLMResponse` format.

### Model IDs
- **Anthropic**: `claude-sonnet-4-20250514`
- **OpenRouter**: `provider/model-name` format (e.g., `google/gemini-2.0-flash-001`)

OpenRouter model IDs can be found at: https://openrouter.ai/models
