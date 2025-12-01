# Component 4: LLM Client

## Overview
Wrapper for LLM API calls (Anthropic Claude) with token counting and error handling.

**File**: `src/tetris_evolve/llm/client.py`
**Test File**: `tests/test_llm_client.py`
**Dependencies**: None (standalone, easily mocked)

---

## Checklist

### 4.1 Project Setup
- [x] **4.1.1** Create directory structure `src/tetris_evolve/llm/`
  - Dependencies: None
- [x] **4.1.2** Create `__init__.py` files for package
  - Dependencies: 4.1.1
- [x] **4.1.3** Create `tests/test_llm_client.py` skeleton
  - Dependencies: 4.1.1
- [x] **4.1.4** Add `anthropic` to dependencies
  - Dependencies: None

### 4.2 Data Classes
- [x] **4.2.1** Define `LLMResponse` dataclass
  - Dependencies: 4.1.2
  ```python
  @dataclass
  class LLMResponse:
      content: str
      input_tokens: int
      output_tokens: int
      model: str
      stop_reason: str  # "end_turn", "max_tokens", etc.
  ```
- [x] **4.2.2** Write tests for `LLMResponse` creation
  - Dependencies: 4.2.1

### 4.3 LLMClient Class - Initialization
- [x] **4.3.1** Implement `LLMClient.__init__(model, api_key, max_retries)`
  - Dependencies: 4.1.2, 4.1.4
  - Initialize Anthropic client
  - Store model name and retry config
- [x] **4.3.2** Write tests for initialization (with mocked client)
  - Dependencies: 4.3.1
- [x] **4.3.3** Implement API key from environment variable fallback
  - Dependencies: 4.3.1
  - Use `ANTHROPIC_API_KEY` env var if not provided
- [x] **4.3.4** Write tests for env var fallback
  - Dependencies: 4.3.3

### 4.4 Message Sending
- [x] **4.4.1** Implement `send_message(messages, system, temperature, max_tokens) -> LLMResponse`
  - Dependencies: 4.2.1, 4.3.1
  - Call Anthropic API
  - Return structured response with token counts
- [x] **4.4.2** Write tests for `send_message` (with mocked API)
  - Dependencies: 4.4.1
- [x] **4.4.3** Implement retry logic for transient errors
  - Dependencies: 4.4.1
  - Retry on: rate limits, server errors
  - Exponential backoff
- [x] **4.4.4** Write tests for retry logic
  - Dependencies: 4.4.3
- [x] **4.4.5** Implement proper error handling
  - Dependencies: 4.4.1
  - Handle: authentication errors, invalid requests, API errors
  - Raise appropriate exceptions
- [x] **4.4.6** Write tests for error handling
  - Dependencies: 4.4.5

### 4.5 Token Counting
- [x] **4.5.1** Implement `count_tokens(text: str) -> int`
  - Dependencies: 4.3.1
  - Use Anthropic's token counting API or estimate
- [x] **4.5.2** Write tests for `count_tokens`
  - Dependencies: 4.5.1
- [x] **4.5.3** Implement `estimate_cost(input_text, output_tokens) -> float`
  - Dependencies: 4.5.1
  - Estimate cost before making call
- [x] **4.5.4** Write tests for `estimate_cost`
  - Dependencies: 4.5.3

### 4.6 Convenience Methods
- [x] **4.6.1** Implement `send_single(prompt: str, system: str = None) -> str`
  - Dependencies: 4.4.1
  - Simplified interface for single prompt → response
  - Returns just the content string
- [x] **4.6.2** Write tests for `send_single`
  - Dependencies: 4.6.1

---

## Test Summary

| Test | Description | Dependencies |
|------|-------------|--------------|
| `test_llm_response_creation` | LLMResponse dataclass works | 4.2.1 |
| `test_client_init` | Client initializes correctly | 4.3.1 |
| `test_client_env_var_fallback` | Uses env var for API key | 4.3.3 |
| `test_send_message_success` | Returns valid response | 4.4.1 |
| `test_send_message_token_counts` | Token counts populated | 4.4.1 |
| `test_retry_on_rate_limit` | Retries on 429 | 4.4.3 |
| `test_retry_on_server_error` | Retries on 500 | 4.4.3 |
| `test_no_retry_on_auth_error` | No retry on 401 | 4.4.5 |
| `test_error_handling_invalid_key` | Auth error raised | 4.4.5 |
| `test_count_tokens` | Token count reasonable | 4.5.1 |
| `test_estimate_cost` | Cost estimate reasonable | 4.5.3 |
| `test_send_single` | Simplified interface works | 4.6.1 |

---

## Sample Implementation Skeleton

```python
# src/tetris_evolve/llm/client.py

import os
import time
from dataclasses import dataclass
from typing import Optional
import anthropic

@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str

class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMAuthenticationError(LLMClientError):
    """Authentication failed."""
    pass

class LLMRateLimitError(LLMClientError):
    """Rate limit exceeded."""
    pass

class LLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_retries: int = 3
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_retries = max_retries

        if not self.api_key:
            raise LLMAuthenticationError("No API key provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def send_message(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Send message to LLM and return structured response."""
        pass

    def send_single(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Simplified interface: single prompt → response string."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    def estimate_cost(
        self,
        input_text: str,
        estimated_output_tokens: int,
        cost_config: dict = None
    ) -> float:
        """Estimate cost of a call."""
        pass

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        pass
```

---

## Mocking Strategy for Tests

```python
# tests/test_llm_client.py

import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_anthropic():
    with patch("anthropic.Anthropic") as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock_client

def test_send_message_success(mock_anthropic):
    # Setup mock response
    mock_anthropic.messages.create.return_value = Mock(
        content=[Mock(text="Hello!")],
        usage=Mock(input_tokens=10, output_tokens=5),
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn"
    )

    client = LLMClient(api_key="test-key")
    response = client.send_message([{"role": "user", "content": "Hi"}])

    assert response.content == "Hello!"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
```

---

## Acceptance Criteria

- [x] All 12 tests pass
- [x] Code coverage > 90%
- [x] Works with real Anthropic API (manual test)
- [x] Retry logic handles transient failures
- [x] Proper error messages for common failures
- [x] Token counting is reasonably accurate
