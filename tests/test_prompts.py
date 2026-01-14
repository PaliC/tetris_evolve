"""
Tests for the prompts module.
"""

from mango_evolve.llm.prompts import (
    ROOT_LLM_SYSTEM_PROMPT_DYNAMIC,
    format_child_mutation_prompt,
    get_root_system_prompt,
    get_root_system_prompt_parts,
)


class TestRootSystemPrompt:
    """Tests for the Root LLM system prompt."""

    def test_prompt_documents_spawn_children(self):
        """Test that spawn_children is documented."""
        prompt = get_root_system_prompt()
        assert "spawn_children" in prompt

    def test_prompt_documents_core_functions(self):
        """Test that core functions are documented."""
        prompt = get_root_system_prompt()
        assert "spawn_children" in prompt
        assert "update_scratchpad" in prompt
        assert "terminate_evolution" in prompt

    def test_prompt_does_not_document_advance_generation(self):
        """Test that advance_generation is NOT documented (now internal)."""
        prompt = get_root_system_prompt()
        # advance_generation should not be in the available functions
        assert "### advance_generation" not in prompt

    def test_prompt_documents_terminate_evolution(self):
        """Test that terminate_evolution is documented."""
        prompt = get_root_system_prompt()
        assert "terminate_evolution" in prompt
        assert "best_program" in prompt

    def test_prompt_documents_update_scratchpad(self):
        """Test that update_scratchpad is documented."""
        prompt = get_root_system_prompt()
        assert "update_scratchpad" in prompt
        assert "scratchpad" in prompt.lower()

    def test_prompt_does_not_document_internal_functions(self):
        """Test that internal helper functions are not documented."""
        prompt = get_root_system_prompt()
        # These internal methods (prefixed with _) should not be listed as available functions
        assert "### _get_best_trials" not in prompt
        assert "### _get_trial(" not in prompt
        assert "### _get_generation_history" not in prompt
        assert "### _advance_generation" not in prompt

    def test_prompt_describes_problem(self):
        """Test that the problem is described."""
        prompt = get_root_system_prompt()
        assert "26 circles" in prompt
        assert "unit square" in prompt
        assert "2.635" in prompt  # Target

    def test_prompt_documents_code_specification(self):
        """Test that code specification is documented."""
        prompt = get_root_system_prompt()
        assert "construct_packing" in prompt
        assert "run_packing" in prompt

    def test_get_root_system_prompt_returns_string(self):
        """Test that get_root_system_prompt returns a string."""
        prompt = get_root_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_includes_evolution_parameters(self):
        """Test that evolution parameters are included in prompt."""
        prompt = get_root_system_prompt(
            max_children_per_generation=5,
            max_generations=3,
            current_generation=1,
        )
        assert "5" in prompt  # max_children_per_generation
        assert "3" in prompt  # max_generations
        assert "Current generation" in prompt

    def test_prompt_encourages_diversity(self):
        """Test that prompt mentions diversity."""
        prompt = get_root_system_prompt()
        assert "diversity" in prompt.lower() or "diverse" in prompt.lower()

    def test_prompt_documents_code_references(self):
        """Test that trial code access syntax is documented."""
        prompt = get_root_system_prompt()
        # The prompt documents how to access code from trials
        assert 'trials["trial_' in prompt or ".code" in prompt

    def test_prompt_documents_selection_format(self):
        """Test that selection format is documented."""
        prompt = get_root_system_prompt()
        assert "selection" in prompt.lower()
        assert "trial_id" in prompt


class TestFormatChildMutationPrompt:
    """Tests for format_child_mutation_prompt."""

    def test_includes_parent_code(self):
        """Test that parent code is included."""
        parent_code = "def construct_packing(): pass"
        prompt = format_child_mutation_prompt(parent_code, 1.5)

        assert parent_code in prompt

    def test_includes_parent_score(self):
        """Test that parent score is included."""
        prompt = format_child_mutation_prompt("code", 1.5)

        assert "1.5" in prompt or "1.50" in prompt

    def test_includes_guidance(self):
        """Test that guidance is included."""
        prompt = format_child_mutation_prompt(
            "code",
            1.5,
            guidance="Try using hexagonal packing",
        )

        assert "hexagonal" in prompt

    def test_includes_target(self):
        """Test that target score is mentioned."""
        prompt = format_child_mutation_prompt("code", 1.5)

        assert "2.635" in prompt

    def test_is_concise(self):
        """Test that mutation prompt is concise (not overly verbose)."""
        prompt = format_child_mutation_prompt("code", 1.5)
        # The simplified prompt should be short
        assert len(prompt) < 500


class TestRootLLMSystemPromptDynamic:
    """Tests for the ROOT_LLM_SYSTEM_PROMPT_DYNAMIC constant."""

    def test_template_contains_placeholders(self):
        """Test that template contains format placeholders."""
        assert "{max_children_per_generation}" in ROOT_LLM_SYSTEM_PROMPT_DYNAMIC
        assert "{max_generations}" in ROOT_LLM_SYSTEM_PROMPT_DYNAMIC
        assert "{current_generation}" in ROOT_LLM_SYSTEM_PROMPT_DYNAMIC

    def test_function_formats_template(self):
        """Test that function properly formats the template."""
        prompt = get_root_system_prompt(
            max_children_per_generation=7,
            max_generations=5,
            current_generation=2,
        )
        # Placeholders should be replaced
        assert "{max_children_per_generation}" not in prompt
        assert "{max_generations}" not in prompt
        assert "{current_generation}" not in prompt
        # Values should appear
        assert "7" in prompt
        assert "5" in prompt


class TestTimeoutConstraint:
    """Tests for timeout constraint in prompts."""

    def test_prompt_includes_timeout_when_provided(self):
        """Test that timeout is included when provided."""
        prompt = get_root_system_prompt(
            max_children_per_generation=15,
            max_generations=10,
            current_generation=3,
            timeout_seconds=300,
        )

        assert "Timeout per trial" in prompt
        assert "300s" in prompt

    def test_prompt_excludes_timeout_when_none(self):
        """Test that timeout is excluded when None."""
        prompt = get_root_system_prompt(
            max_children_per_generation=15,
            max_generations=10,
            current_generation=3,
            timeout_seconds=None,
        )

        assert "Timeout" not in prompt

    def test_prompt_parts_include_timeout(self):
        """Test that get_root_system_prompt_parts includes timeout."""
        parts = get_root_system_prompt_parts(
            max_children_per_generation=15,
            max_generations=10,
            current_generation=3,
            timeout_seconds=300,
        )

        # Should have 2 parts
        assert len(parts) == 2

        # The dynamic part should include timeout
        dynamic_part = parts[1]["text"]
        assert "300s" in dynamic_part
