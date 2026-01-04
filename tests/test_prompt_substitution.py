"""
Tests for the prompt substitution module.

Tests the {{CODE_TRIAL_X_Y}} token substitution functionality that allows
Root LLM to reference code from previous trials in child LLM prompts.
"""

import json

from mango_evolve.evolution_api import TrialResult
from mango_evolve.utils.prompt_substitution import (
    TRIAL_CODE_PATTERN,
    find_trial_code_tokens,
    get_trial_code,
    load_trial_code_from_disk,
    substitute_trial_codes,
    substitute_trial_codes_batch,
)


class TestTrialCodePattern:
    """Tests for the TRIAL_CODE_PATTERN regex."""

    def test_matches_simple_token(self):
        """Test that pattern matches simple token."""
        match = TRIAL_CODE_PATTERN.search("{{CODE_TRIAL_0_0}}")
        assert match is not None
        assert match.group(0) == "{{CODE_TRIAL_0_0}}"
        assert match.group(1) == "0"
        assert match.group(2) == "0"

    def test_matches_multi_digit_numbers(self):
        """Test that pattern matches multi-digit generation and trial numbers."""
        match = TRIAL_CODE_PATTERN.search("{{CODE_TRIAL_12_345}}")
        assert match is not None
        assert match.group(1) == "12"
        assert match.group(2) == "345"

    def test_no_match_for_partial_tokens(self):
        """Test that pattern doesn't match partial tokens."""
        assert TRIAL_CODE_PATTERN.search("{{CODE_TRIAL_0_}}") is None
        assert TRIAL_CODE_PATTERN.search("{{CODE_TRIAL__0}}") is None
        assert TRIAL_CODE_PATTERN.search("{CODE_TRIAL_0_0}") is None
        assert TRIAL_CODE_PATTERN.search("CODE_TRIAL_0_0") is None

    def test_matches_in_text(self):
        """Test that pattern matches token embedded in text."""
        text = "Improve this code: {{CODE_TRIAL_0_3}} and make it better."
        match = TRIAL_CODE_PATTERN.search(text)
        assert match is not None
        assert match.group(0) == "{{CODE_TRIAL_0_3}}"


class TestFindTrialCodeTokens:
    """Tests for find_trial_code_tokens function."""

    def test_no_tokens(self):
        """Test with prompt containing no tokens."""
        prompt = "Write a circle packing algorithm."
        tokens = find_trial_code_tokens(prompt)
        assert tokens == []

    def test_single_token(self):
        """Test with single token."""
        prompt = "Improve this: {{CODE_TRIAL_0_5}}"
        tokens = find_trial_code_tokens(prompt)
        assert len(tokens) == 1
        assert tokens[0] == ("{{CODE_TRIAL_0_5}}", 0, 5)

    def test_multiple_tokens(self):
        """Test with multiple tokens."""
        prompt = """Combine these approaches:
        Approach A: {{CODE_TRIAL_0_1}}
        Approach B: {{CODE_TRIAL_0_3}}
        Approach C: {{CODE_TRIAL_1_0}}
        """
        tokens = find_trial_code_tokens(prompt)
        assert len(tokens) == 3
        assert tokens[0] == ("{{CODE_TRIAL_0_1}}", 0, 1)
        assert tokens[1] == ("{{CODE_TRIAL_0_3}}", 0, 3)
        assert tokens[2] == ("{{CODE_TRIAL_1_0}}", 1, 0)

    def test_duplicate_tokens(self):
        """Test with duplicate tokens."""
        prompt = "{{CODE_TRIAL_0_1}} is similar to {{CODE_TRIAL_0_1}}"
        tokens = find_trial_code_tokens(prompt)
        assert len(tokens) == 2
        assert tokens[0] == ("{{CODE_TRIAL_0_1}}", 0, 1)
        assert tokens[1] == ("{{CODE_TRIAL_0_1}}", 0, 1)


class TestLoadTrialCodeFromDisk:
    """Tests for load_trial_code_from_disk function."""

    def test_loads_existing_trial(self, temp_dir):
        """Test loading code from an existing trial file."""
        # Create trial file structure
        gen_dir = temp_dir / "generations" / "gen_0"
        gen_dir.mkdir(parents=True)

        trial_data = {
            "trial_id": "trial_0_3",
            "generation": 0,
            "code": "def construct_packing():\n    return 'test code'",
            "metrics": {"valid": True},
        }
        trial_path = gen_dir / "trial_0_3.json"
        with open(trial_path, "w") as f:
            json.dump(trial_data, f)

        # Load the trial
        code = load_trial_code_from_disk("trial_0_3", temp_dir)
        assert code == "def construct_packing():\n    return 'test code'"

    def test_returns_none_for_missing_trial(self, temp_dir):
        """Test that missing trial returns None."""
        code = load_trial_code_from_disk("trial_99_99", temp_dir)
        assert code is None

    def test_returns_none_for_invalid_trial_id(self, temp_dir):
        """Test that invalid trial ID format returns None."""
        code = load_trial_code_from_disk("invalid_id", temp_dir)
        assert code is None

    def test_returns_none_for_invalid_json(self, temp_dir):
        """Test that invalid JSON file returns None."""
        gen_dir = temp_dir / "generations" / "gen_0"
        gen_dir.mkdir(parents=True)

        trial_path = gen_dir / "trial_0_0.json"
        with open(trial_path, "w") as f:
            f.write("not valid json {{{")

        code = load_trial_code_from_disk("trial_0_0", temp_dir)
        assert code is None

    def test_loads_from_different_generations(self, temp_dir):
        """Test loading from different generation directories."""
        # Gen 0
        gen0_dir = temp_dir / "generations" / "gen_0"
        gen0_dir.mkdir(parents=True)
        with open(gen0_dir / "trial_0_1.json", "w") as f:
            json.dump({"code": "gen0_code"}, f)

        # Gen 1
        gen1_dir = temp_dir / "generations" / "gen_1"
        gen1_dir.mkdir(parents=True)
        with open(gen1_dir / "trial_1_2.json", "w") as f:
            json.dump({"code": "gen1_code"}, f)

        assert load_trial_code_from_disk("trial_0_1", temp_dir) == "gen0_code"
        assert load_trial_code_from_disk("trial_1_2", temp_dir) == "gen1_code"


class TestGetTrialCode:
    """Tests for get_trial_code function."""

    def test_gets_from_memory_first(self, temp_dir):
        """Test that memory is checked before disk."""
        # Create a trial in memory
        trial = TrialResult(
            trial_id="trial_0_1",
            code="memory_code",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        all_trials = {"trial_0_1": trial}

        # Also create on disk with different code
        gen_dir = temp_dir / "generations" / "gen_0"
        gen_dir.mkdir(parents=True)
        with open(gen_dir / "trial_0_1.json", "w") as f:
            json.dump({"code": "disk_code"}, f)

        # Should get memory version
        code = get_trial_code("trial_0_1", all_trials, temp_dir)
        assert code == "memory_code"

    def test_falls_back_to_disk(self, temp_dir):
        """Test that disk is used when not in memory."""
        gen_dir = temp_dir / "generations" / "gen_0"
        gen_dir.mkdir(parents=True)
        with open(gen_dir / "trial_0_2.json", "w") as f:
            json.dump({"code": "disk_code"}, f)

        code = get_trial_code("trial_0_2", all_trials={}, experiment_dir=temp_dir)
        assert code == "disk_code"

    def test_returns_none_when_not_found(self, temp_dir):
        """Test that None is returned when trial not found anywhere."""
        code = get_trial_code("trial_99_99", all_trials={}, experiment_dir=temp_dir)
        assert code is None

    def test_works_with_none_parameters(self):
        """Test that function works when parameters are None."""
        code = get_trial_code("trial_0_0", all_trials=None, experiment_dir=None)
        assert code is None

    def test_memory_with_empty_code_falls_to_disk(self, temp_dir):
        """Test that empty code in memory falls back to disk."""
        trial = TrialResult(
            trial_id="trial_0_1",
            code="",  # Empty code
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=False,
            generation=0,
        )
        all_trials = {"trial_0_1": trial}

        gen_dir = temp_dir / "generations" / "gen_0"
        gen_dir.mkdir(parents=True)
        with open(gen_dir / "trial_0_1.json", "w") as f:
            json.dump({"code": "disk_code"}, f)

        code = get_trial_code("trial_0_1", all_trials, temp_dir)
        assert code == "disk_code"


class TestSubstituteTrialCodes:
    """Tests for substitute_trial_codes function."""

    def test_no_substitution_needed(self):
        """Test prompt without tokens is unchanged."""
        prompt = "Write a simple algorithm."
        result, report = substitute_trial_codes(prompt)
        assert result == prompt
        assert report == []

    def test_single_substitution(self, temp_dir):
        """Test single token substitution."""
        trial = TrialResult(
            trial_id="trial_0_3",
            code="def construct_packing():\n    return solution",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        all_trials = {"trial_0_3": trial}

        prompt = "Improve this code:\n{{CODE_TRIAL_0_3}}\nMake it better."
        result, report = substitute_trial_codes(prompt, all_trials)

        expected = (
            "Improve this code:\ndef construct_packing():\n    return solution\nMake it better."
        )
        assert result == expected
        assert len(report) == 1
        assert report[0]["success"] is True
        assert report[0]["trial_id"] == "trial_0_3"

    def test_multiple_substitutions(self, temp_dir):
        """Test multiple token substitutions."""
        trial1 = TrialResult(
            trial_id="trial_0_1",
            code="code1",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        trial2 = TrialResult(
            trial_id="trial_0_2",
            code="code2",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        all_trials = {"trial_0_1": trial1, "trial_0_2": trial2}

        prompt = "A: {{CODE_TRIAL_0_1}}, B: {{CODE_TRIAL_0_2}}"
        result, report = substitute_trial_codes(prompt, all_trials)

        assert result == "A: code1, B: code2"
        assert len(report) == 2
        assert all(r["success"] for r in report)

    def test_missing_trial_shows_error_marker(self, temp_dir):
        """Test that missing trial shows error marker."""
        prompt = "Improve: {{CODE_TRIAL_99_99}}"
        result, report = substitute_trial_codes(prompt, all_trials={}, experiment_dir=temp_dir)

        assert "[CODE NOT FOUND: trial_99_99]" in result
        assert len(report) == 1
        assert report[0]["success"] is False
        assert "not found" in report[0]["error"]

    def test_mixed_found_and_missing(self, temp_dir):
        """Test prompt with both found and missing trials."""
        trial = TrialResult(
            trial_id="trial_0_1",
            code="found_code",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        all_trials = {"trial_0_1": trial}

        prompt = "Found: {{CODE_TRIAL_0_1}}, Missing: {{CODE_TRIAL_99_99}}"
        result, report = substitute_trial_codes(prompt, all_trials, temp_dir)

        assert "found_code" in result
        assert "[CODE NOT FOUND: trial_99_99]" in result
        assert len(report) == 2
        assert report[0]["success"] is True
        assert report[1]["success"] is False

    def test_duplicate_tokens_substituted(self, temp_dir):
        """Test that duplicate tokens are all substituted."""
        trial = TrialResult(
            trial_id="trial_0_1",
            code="THE_CODE",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        all_trials = {"trial_0_1": trial}

        prompt = "{{CODE_TRIAL_0_1}} and again {{CODE_TRIAL_0_1}}"
        result, report = substitute_trial_codes(prompt, all_trials)

        assert result == "THE_CODE and again THE_CODE"
        # Report contains two entries for the two occurrences
        assert len(report) == 2

    def test_from_disk_when_not_in_memory(self, temp_dir):
        """Test substitution from disk when trial not in memory."""
        gen_dir = temp_dir / "generations" / "gen_0"
        gen_dir.mkdir(parents=True)
        with open(gen_dir / "trial_0_5.json", "w") as f:
            json.dump({"code": "disk_only_code"}, f)

        prompt = "Code: {{CODE_TRIAL_0_5}}"
        result, report = substitute_trial_codes(prompt, all_trials={}, experiment_dir=temp_dir)

        assert result == "Code: disk_only_code"
        assert report[0]["success"] is True


class TestSubstituteTrialCodesBatch:
    """Tests for substitute_trial_codes_batch function."""

    def test_batch_substitution(self, temp_dir):
        """Test batch substitution of multiple prompts."""
        trial = TrialResult(
            trial_id="trial_0_1",
            code="shared_code",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        all_trials = {"trial_0_1": trial}

        prompts = [
            "Prompt A: {{CODE_TRIAL_0_1}}",
            "Prompt B without tokens",
            "Prompt C: {{CODE_TRIAL_0_1}}",
        ]

        results = substitute_trial_codes_batch(prompts, all_trials)

        assert len(results) == 3
        assert results[0][0] == "Prompt A: shared_code"
        assert len(results[0][1]) == 1  # One substitution
        assert results[1][0] == "Prompt B without tokens"
        assert len(results[1][1]) == 0  # No substitutions
        assert results[2][0] == "Prompt C: shared_code"
        assert len(results[2][1]) == 1  # One substitution

    def test_empty_batch(self):
        """Test empty batch returns empty list."""
        results = substitute_trial_codes_batch([])
        assert results == []


class TestEdgeCases:
    """Edge case tests for prompt substitution."""

    def test_token_at_start_of_prompt(self):
        """Test token at the very start of prompt."""
        trial = TrialResult(
            trial_id="trial_0_0",
            code="START",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        prompt = "{{CODE_TRIAL_0_0}} continues here"
        result, _ = substitute_trial_codes(prompt, {"trial_0_0": trial})
        assert result == "START continues here"

    def test_token_at_end_of_prompt(self):
        """Test token at the very end of prompt."""
        trial = TrialResult(
            trial_id="trial_0_0",
            code="END",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        prompt = "Here is the code: {{CODE_TRIAL_0_0}}"
        result, _ = substitute_trial_codes(prompt, {"trial_0_0": trial})
        assert result == "Here is the code: END"

    def test_multiline_code_substitution(self):
        """Test substitution with multiline code."""
        multiline_code = """def construct_packing():
    centers = []
    for i in range(26):
        centers.append([i * 0.04, 0.5])
    return np.array(centers), np.ones(26) * 0.02, 0.52"""

        trial = TrialResult(
            trial_id="trial_0_1",
            code=multiline_code,
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        prompt = "Improve this:\n{{CODE_TRIAL_0_1}}"
        result, _ = substitute_trial_codes(prompt, {"trial_0_1": trial})
        assert multiline_code in result

    def test_code_with_special_characters(self):
        """Test substitution with code containing special regex characters."""
        code_with_special = "pattern = r'\\d+\\.\\d*' # regex pattern"
        trial = TrialResult(
            trial_id="trial_0_0",
            code=code_with_special,
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=0,
        )
        prompt = "Code: {{CODE_TRIAL_0_0}}"
        result, _ = substitute_trial_codes(prompt, {"trial_0_0": trial})
        assert result == f"Code: {code_with_special}"

    def test_large_generation_and_trial_numbers(self):
        """Test with large generation and trial numbers."""
        trial = TrialResult(
            trial_id="trial_100_999",
            code="large_numbers",
            metrics={},
            prompt="",
            response="",
            reasoning="",
            success=True,
            generation=100,
        )
        prompt = "{{CODE_TRIAL_100_999}}"
        result, report = substitute_trial_codes(prompt, {"trial_100_999": trial})
        assert result == "large_numbers"
        assert report[0]["success"] is True
