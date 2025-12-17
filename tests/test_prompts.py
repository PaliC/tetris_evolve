"""
Tests for the prompts module.
"""

from tetris_evolve.llm.prompts import (
    ROOT_LLM_SYSTEM_PROMPT_DYNAMIC,
    format_child_mutation_prompt,
    get_root_system_prompt,
)


class TestRootSystemPrompt:
    """Tests for the Root LLM system prompt."""

    def test_prompt_documents_spawn_child_llm(self):
        """Test that spawn_child_llm is documented."""
        prompt = get_root_system_prompt()
        assert "spawn_child_llm" in prompt
        assert "prompt: str" in prompt or "prompt:" in prompt

    def test_prompt_documents_spawn_children_parallel(self):
        """Test that spawn_children_parallel is documented."""
        prompt = get_root_system_prompt()
        assert "spawn_children_parallel" in prompt
        assert "PRIMARY FUNCTION" in prompt

    def test_prompt_documents_evaluate_program(self):
        """Test that evaluate_program is documented."""
        prompt = get_root_system_prompt()
        assert "evaluate_program" in prompt

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

    def test_prompt_documents_only_5_functions(self):
        """Test that only 5 core functions are documented."""
        prompt = get_root_system_prompt()
        assert "5 functions" in prompt or "these 5" in prompt.lower()

    def test_prompt_documents_get_trial_code(self):
        """Test that get_trial_code is documented."""
        prompt = get_root_system_prompt()
        assert "get_trial_code" in prompt
        assert "trial_ids" in prompt

    def test_prompt_does_not_document_internal_functions(self):
        """Test that internal helper functions are not documented."""
        prompt = get_root_system_prompt()
        # These internal methods (prefixed with _) should not be listed as available functions
        assert "### _get_best_trials" not in prompt
        assert "### _get_cost_remaining" not in prompt
        assert "### _get_trial(" not in prompt  # Note: get_trial_code IS a public function
        assert "### _get_generation_history" not in prompt
        assert "### _advance_generation" not in prompt

    def test_prompt_explains_repl_usage(self):
        """Test that REPL usage is explained."""
        prompt = get_root_system_prompt()
        assert "```repl" in prompt
        assert "REPL" in prompt

    def test_prompt_describes_problem(self):
        """Test that the problem is described."""
        prompt = get_root_system_prompt()
        assert "26 circles" in prompt
        assert "unit square" in prompt
        assert "2.635" in prompt  # Benchmark

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

    def test_prompt_explains_automatic_generation_advance(self):
        """Test that automatic generation advancement is explained."""
        prompt = get_root_system_prompt()
        assert "automatically" in prompt.lower()
        assert "generation" in prompt.lower()


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

    def test_includes_requirements(self):
        """Test that requirements are included."""
        prompt = format_child_mutation_prompt("code", 1.5)

        assert "26 circles" in prompt
        assert "construct_packing" in prompt
        assert "run_packing" in prompt


class TestRootLLMSystemPromptDynamic:
    """Tests for the ROOT_LLM_SYSTEM_PROMPT_DYNAMIC constant."""

    def test_template_contains_placeholders(self):
        """Test that template contains format placeholders."""
        assert "{max_children_per_generation}" in ROOT_LLM_SYSTEM_PROMPT_DYNAMIC
        assert "{max_generations}" in ROOT_LLM_SYSTEM_PROMPT_DYNAMIC
        assert "{current_generation}" in ROOT_LLM_SYSTEM_PROMPT_DYNAMIC

    def test_template_is_not_empty(self):
        """Test that template is not empty."""
        assert len(ROOT_LLM_SYSTEM_PROMPT_DYNAMIC) > 0

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
