"""
Tests for mango_evolve.exceptions module.
"""

import pytest

from mango_evolve.exceptions import (
    BudgetExceededError,
    CodeExtractionError,
    ConfigValidationError,
    EvaluationError,
    MangoEvolveError,
)


class TestExceptions:
    """Tests for exception classes."""

    def test_all_exceptions_inherit_from_base(self):
        """All exceptions should inherit from MangoEvolveError."""
        exceptions = [
            BudgetExceededError,
            ConfigValidationError,
            CodeExtractionError,
            EvaluationError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, MangoEvolveError)

    def test_exceptions_can_be_raised(self):
        """Exceptions can be raised and caught."""
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError("Budget exceeded")

        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError("Invalid config")

        with pytest.raises(CodeExtractionError):
            raise CodeExtractionError("Cannot extract code")

        with pytest.raises(EvaluationError):
            raise EvaluationError("Evaluation failed")

    def test_exceptions_have_message(self):
        """Exceptions should preserve their message."""
        msg = "Test error message"
        exc = BudgetExceededError(msg)
        assert str(exc) == msg

    def test_catch_base_exception(self):
        """All exceptions can be caught with base exception."""
        with pytest.raises(MangoEvolveError):
            raise BudgetExceededError("Test")

        with pytest.raises(MangoEvolveError):
            raise ConfigValidationError("Test")

    def test_import_from_package(self):
        """Exceptions can be imported from main package."""
        from mango_evolve import (
            BudgetExceededError,
            MangoEvolveError,
        )

        assert MangoEvolveError is not None
        assert BudgetExceededError is not None
