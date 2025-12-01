"""Tests for LLM Client (Component 4)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from tetris_evolve.llm.client import (
    LLMClient,
    LLMResponse,
    LLMClientError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMAPIError,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """LLMResponse dataclass creates correctly."""
        response = LLMResponse(
            content="Hello, world!",
            input_tokens=10,
            output_tokens=5,
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
        )

        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.model == "claude-sonnet-4-20250514"
        assert response.stop_reason == "end_turn"


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_client_init_with_api_key(self):
        """Client initializes correctly with provided API key."""
        with patch("anthropic.Anthropic"):
            client = LLMClient(api_key="test-key-123")

            assert client.api_key == "test-key-123"
            assert client.model == "claude-sonnet-4-20250514"

    def test_client_env_var_fallback(self):
        """Uses environment variable for API key if not provided."""
        with patch("anthropic.Anthropic"):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key-456"}):
                client = LLMClient()

                assert client.api_key == "env-key-456"

    def test_client_no_key_raises(self):
        """Raises error when no API key available."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY from environment
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            with pytest.raises(LLMAuthenticationError):
                LLMClient()

    def test_client_custom_model(self):
        """Client accepts custom model name."""
        with patch("anthropic.Anthropic"):
            client = LLMClient(model="claude-haiku-3-5-20241022", api_key="test-key")

            assert client.model == "claude-haiku-3-5-20241022"


class TestLLMClientSendMessage:
    """Tests for send_message functionality."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create mock Anthropic client."""
        with patch("anthropic.Anthropic") as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client

    def test_send_message_success(self, mock_anthropic):
        """Returns valid response on success."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Hello!")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_anthropic.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        response = client.send_message([{"role": "user", "content": "Hi"}])

        assert response.content == "Hello!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_send_message_token_counts(self, mock_anthropic):
        """Token counts are populated correctly."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_anthropic.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        response = client.send_message([{"role": "user", "content": "Test"}])

        assert response.input_tokens == 100
        assert response.output_tokens == 50

    def test_send_message_with_system(self, mock_anthropic):
        """System prompt is passed correctly."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=50, output_tokens=25)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_anthropic.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        client.send_message(
            [{"role": "user", "content": "Hi"}],
            system="You are a helpful assistant.",
        )

        # Verify system was passed
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a helpful assistant."


class TestLLMClientRetry:
    """Tests for retry logic."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create mock Anthropic client."""
        with patch("anthropic.Anthropic") as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client

    def test_retry_on_rate_limit(self, mock_anthropic):
        """Retries on rate limit error."""
        import anthropic

        # First call raises rate limit, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(text="Success")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_anthropic.messages.create.side_effect = [
            anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=Mock(status_code=429),
                body={},
            ),
            mock_response,
        ]

        client = LLMClient(api_key="test-key", max_retries=3, base_delay=0.01)
        response = client.send_message([{"role": "user", "content": "Hi"}])

        assert response.content == "Success"
        assert mock_anthropic.messages.create.call_count == 2

    def test_no_retry_on_auth_error(self, mock_anthropic):
        """Does not retry on authentication error."""
        import anthropic

        mock_anthropic.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key",
            response=Mock(status_code=401),
            body={},
        )

        client = LLMClient(api_key="test-key", max_retries=3, base_delay=0.01)

        with pytest.raises(LLMAuthenticationError):
            client.send_message([{"role": "user", "content": "Hi"}])

        # Should only call once (no retries)
        assert mock_anthropic.messages.create.call_count == 1


class TestLLMClientConvenience:
    """Tests for convenience methods."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create mock Anthropic client."""
        with patch("anthropic.Anthropic") as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client

    def test_send_single(self, mock_anthropic):
        """Simplified interface works correctly."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Hello there!")]
        mock_response.usage = Mock(input_tokens=5, output_tokens=3)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_anthropic.messages.create.return_value = mock_response

        client = LLMClient(api_key="test-key")
        response = client.send_single("Hi")

        assert response == "Hello there!"

    def test_count_tokens(self, mock_anthropic):
        """Token count is reasonable."""
        client = LLMClient(api_key="test-key")

        # ~4 chars per token
        count = client.count_tokens("Hello world, this is a test.")

        assert 5 <= count <= 15  # Reasonable range

    def test_estimate_cost(self, mock_anthropic):
        """Cost estimate is reasonable."""
        client = LLMClient(api_key="test-key")

        cost = client.estimate_cost(
            input_text="A" * 4000,  # ~1000 tokens
            estimated_output_tokens=500,
        )

        # Should be around $0.003 * 1 + $0.015 * 0.5 = $0.0105
        assert 0.005 < cost < 0.02
