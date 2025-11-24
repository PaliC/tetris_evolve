"""
Child LLM executor for code generation.

This module implements the Child RLLM execution system that generates
code mutations based on prompts from the Root LLM.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol
import ast
import re
import time
import uuid

from ..environment.base import EnvironmentConfig
from ..rlm import SharedREPL


class LLMClient(Protocol):
    """Protocol for LLM client."""

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ValidationError(Exception):
    """Raised when code validation fails."""
    pass


@dataclass
class GenerationResult:
    """Result of code generation."""
    success: bool
    code: Optional[str] = None
    rllm_id: str = ""
    prompt: str = ""
    validation_passed: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "code": self.code,
            "rllm_id": self.rllm_id,
            "prompt": self.prompt,
            "validation_passed": self.validation_passed,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_error(
        cls,
        error: str,
        rllm_id: str,
        prompt: str,
    ) -> "GenerationResult":
        """Create an error result."""
        return cls(
            success=False,
            rllm_id=rllm_id,
            prompt=prompt,
            error=error,
        )


class CodeValidator:
    """
    Validates generated Python code.

    Checks for:
    - Syntax errors
    - Required class/method presence
    - Potentially dangerous code
    """

    # Dangerous patterns to check for
    DANGEROUS_PATTERNS = [
        r'\bos\.system\b',
        r'\bsubprocess\b',
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\b__import__\b',
        r'\bopen\s*\([^)]*["\']w["\']',  # Writing to files
        r'\brm\s+-rf\b',
        r'\bshutil\.rmtree\b',
    ]

    def validate(
        self,
        code: str,
        require_class: Optional[str] = None,
        require_method: Optional[str] = None,
        check_safety: bool = False,
    ) -> ValidationResult:
        """
        Validate Python code.

        Args:
            code: Code to validate
            require_class: Class name that must be present
            require_method: Method name that must be present
            check_safety: Whether to check for dangerous patterns

        Returns:
            ValidationResult with is_valid and any errors
        """
        errors = []
        warnings = []

        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check for required class
        if require_class:
            class_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ]
            if require_class not in class_names:
                errors.append(f"Required class '{require_class}' not found")

        # Check for required method
        if require_method:
            method_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == require_method:
                    method_found = True
                    break
            if not method_found:
                errors.append(f"Required method '{require_method}' not found")

        # Check for dangerous patterns
        if check_safety:
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, code):
                    errors.append(f"Potentially dangerous code pattern detected: {pattern}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class CodeGenerator:
    """
    Generates and processes code from LLM responses.
    """

    def __init__(self, env_config: EnvironmentConfig):
        """
        Initialize code generator.

        Args:
            env_config: Environment configuration for context
        """
        self.env_config = env_config

    def extract_code(self, response: str) -> str:
        """
        Extract Python code from an LLM response.

        Handles responses with code blocks (```python) and raw code.

        Args:
            response: Raw LLM response

        Returns:
            Extracted Python code
        """
        # Try to find code blocks
        code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Prefer blocks with class definitions
            for match in matches:
                if 'class' in match and 'Player' in match:
                    return match.strip()
            # Fall back to first block
            return matches[0].strip()

        # No code blocks, try to find class definition
        lines = response.strip().split('\n')

        # Find start of class definition
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                start_idx = i
                break

        if start_idx is not None:
            # Extract from class definition to end
            return '\n'.join(lines[start_idx:]).strip()

        # Return entire response as code
        return response.strip()

    def build_prompt(
        self,
        base_prompt: str,
        parent_code: Optional[str] = None,
        include_env_context: bool = True,
    ) -> str:
        """
        Build a complete prompt for code generation.

        Args:
            base_prompt: Base prompt from Root LLM
            parent_code: Optional parent code to improve
            include_env_context: Whether to include environment description

        Returns:
            Complete prompt string
        """
        parts = []

        # Add environment context
        if include_env_context:
            parts.append("Environment Context:")
            parts.append(self.env_config.get_observation_description())
            parts.append(self.env_config.get_action_description())
            parts.append("")

        # Add parent code if provided
        if parent_code:
            parts.append("Parent Code:")
            parts.append(parent_code)
            parts.append("")

        # Add base prompt
        parts.append("Task:")
        parts.append(base_prompt)

        # Add output format instructions
        parts.append("")
        parts.append("Return only the Python code. Use ```python code blocks.")

        return '\n'.join(parts)


class ChildLLMExecutor:
    """
    Executes Child RLLM code generation.

    Handles:
    - Prompt building with context
    - LLM invocation
    - Code extraction and validation
    - Result tracking
    """

    def __init__(
        self,
        llm_client: LLMClient,
        env_config: EnvironmentConfig,
        validator: Optional[CodeValidator] = None,
        generator: Optional[CodeGenerator] = None,
    ):
        """
        Initialize Child LLM executor.

        Args:
            llm_client: Client for LLM API
            env_config: Environment configuration
            validator: Code validator (created if not provided)
            generator: Code generator (created if not provided)
        """
        self.llm_client = llm_client
        self.env_config = env_config
        self.validator = validator or CodeValidator()
        self.generator = generator or CodeGenerator(env_config)

        self._execution_count = 0

    def execute(
        self,
        prompt: str,
        parent_code: Optional[str] = None,
        validate: bool = True,
        repl_context: Optional[SharedREPL] = None,
        require_class: str = "TetrisPlayer",
        require_method: str = "select_action",
    ) -> GenerationResult:
        """
        Execute code generation.

        Args:
            prompt: Prompt for code generation
            parent_code: Optional parent code to improve
            validate: Whether to validate generated code
            repl_context: Optional REPL for context
            require_class: Required class name
            require_method: Required method name

        Returns:
            GenerationResult with generated code
        """
        start_time = time.time()
        rllm_id = f"rllm_{uuid.uuid4().hex[:8]}"
        self._execution_count += 1

        try:
            # Build complete prompt
            full_prompt = self.generator.build_prompt(
                base_prompt=prompt,
                parent_code=parent_code,
            )

            # Call LLM
            response = self.llm_client.generate(full_prompt)

            # Extract code
            code = self.generator.extract_code(response)

            # Validate if requested
            validation_passed = True
            if validate:
                result = self.validator.validate(
                    code,
                    require_class=require_class,
                    require_method=require_method,
                    check_safety=True,
                )
                validation_passed = result.is_valid

                if not validation_passed:
                    return GenerationResult(
                        success=False,
                        code=code,
                        rllm_id=rllm_id,
                        prompt=prompt,
                        validation_passed=False,
                        execution_time=time.time() - start_time,
                        error="; ".join(result.errors),
                    )

            return GenerationResult(
                success=True,
                code=code,
                rllm_id=rllm_id,
                prompt=prompt,
                validation_passed=validation_passed,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return GenerationResult.from_error(
                error=str(e),
                rllm_id=rllm_id,
                prompt=prompt,
            )

    def create_handler(self) -> Callable[[str], str]:
        """
        Create a handler function for use with RootLLMFunctions.spawn_rllm.

        Returns:
            Handler function that takes prompt and returns code
        """
        def handler(prompt: str) -> str:
            result = self.execute(prompt)
            if result.success and result.code:
                return result.code
            else:
                raise RuntimeError(f"Code generation failed: {result.error}")

        return handler
