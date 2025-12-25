"""
Tests for LLM-driven generation selection functionality.
"""

from unittest.mock import MagicMock

import pytest

from mango_evolve import (
    CostTracker,
    EvolutionAPI,
    ExperimentLogger,
)
from mango_evolve.evolution_api import GenerationSummary, TrialSelection
from mango_evolve.llm import MockLLMClient
from mango_evolve.root_llm import RootLLMOrchestrator
from mango_evolve.utils.code_extraction import extract_selection_block


class TestTrialSelection:
    """Tests for TrialSelection dataclass."""

    def test_creation(self):
        """Test creating a TrialSelection."""
        selection = TrialSelection(
            trial_id="trial_0_1",
            reasoning="Highest score of 2.45",
            category="performance",
        )

        assert selection.trial_id == "trial_0_1"
        assert selection.reasoning == "Highest score of 2.45"
        assert selection.category == "performance"

    def test_to_dict(self):
        """Test converting TrialSelection to dictionary."""
        selection = TrialSelection(
            trial_id="trial_0_2",
            reasoning="Novel approach worth exploring",
            category="diversity",
        )

        d = selection.to_dict()

        assert d["trial_id"] == "trial_0_2"
        assert d["reasoning"] == "Novel approach worth exploring"
        assert d["category"] == "diversity"

    def test_from_dict(self):
        """Test creating TrialSelection from dictionary."""
        data = {
            "trial_id": "trial_0_3",
            "reasoning": "Shows potential for improvement",
            "category": "potential",
        }

        selection = TrialSelection.from_dict(data)

        assert selection.trial_id == "trial_0_3"
        assert selection.reasoning == "Shows potential for improvement"
        assert selection.category == "potential"

    def test_from_dict_missing_category(self):
        """Test from_dict with missing category defaults correctly."""
        data = {
            "trial_id": "trial_0_0",
            "reasoning": "Best score",
        }

        selection = TrialSelection.from_dict(data)

        assert selection.category == "performance"  # Default


class TestGenerationSummaryWithSelections:
    """Tests for GenerationSummary with trial_selections field."""

    def test_trial_selections_field(self):
        """Test that GenerationSummary has trial_selections field."""
        summary = GenerationSummary(
            generation_num=0,
            trials=[],
            trial_selections=[
                TrialSelection("trial_0_0", "Best score", "performance"),
                TrialSelection("trial_0_1", "Novel approach", "diversity"),
            ],
        )

        assert len(summary.trial_selections) == 2
        assert summary.trial_selections[0].trial_id == "trial_0_0"

    def test_to_dict_includes_selections(self):
        """Test that to_dict includes trial_selections."""
        summary = GenerationSummary(
            generation_num=0,
            trials=[],
            trial_selections=[
                TrialSelection("trial_0_0", "Best", "performance"),
            ],
        )

        d = summary.to_dict()

        assert "trial_selections" in d
        assert len(d["trial_selections"]) == 1
        assert d["trial_selections"][0]["trial_id"] == "trial_0_0"


class TestExtractSelectionBlock:
    """Tests for extract_selection_block utility."""

    def test_extract_valid_selection(self):
        """Test extracting a valid selection block."""
        text = """Looking at the results, I'll select these trials:

```selection
{
  "selections": [
    {"trial_id": "trial_0_2", "reasoning": "Highest score of 2.45", "category": "performance"},
    {"trial_id": "trial_0_5", "reasoning": "Novel hexagonal approach", "category": "diversity"}
  ],
  "summary": "Selected top performer and diverse approach"
}
```

Now let me continue.
"""
        result = extract_selection_block(text)

        assert result is not None
        assert "selections" in result
        assert len(result["selections"]) == 2
        assert result["selections"][0]["trial_id"] == "trial_0_2"
        assert result["summary"] == "Selected top performer and diverse approach"

    def test_extract_no_selection_block(self):
        """Test when no selection block is present."""
        text = "Just some text without any selection block."

        result = extract_selection_block(text)

        assert result is None

    def test_extract_invalid_json(self):
        """Test handling of invalid JSON in selection block."""
        text = """Here's my selection:

```selection
{
  "selections": [
    {"trial_id": "trial_0_0", "reasoning": "Best score",
  ]
}
```
"""
        result = extract_selection_block(text)

        assert result is None

    def test_extract_selection_with_extra_content(self):
        """Test selection block with surrounding content."""
        text = """## Analysis

I've analyzed all the trials and here are my selections:

```selection
{
  "selections": [
    {"trial_id": "trial_0_1", "reasoning": "Best overall performance", "category": "performance"}
  ],
  "summary": "Focusing on performance"
}
```

## Next Steps

Let me spawn more children...
"""
        result = extract_selection_block(text)

        assert result is not None
        assert len(result["selections"]) == 1


class TestAdvanceGenerationWithSelections:
    """Tests for _advance_generation with LLM selections."""

    @pytest.fixture
    def evolution_api(self, sample_config, temp_dir):
        """Create an EvolutionAPI instance."""
        sample_config.experiment.output_dir = str(temp_dir)

        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "valid": True,
            "score": 2.0,
            "eval_time": 0.1,
        }

        child_llm = MockLLMClient(
            model=sample_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 10,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=5,
        )

        return api

    def test_advance_with_selections(self, evolution_api):
        """Test advancing with explicit LLM selections."""
        # Spawn some children
        evolution_api.spawn_child_llm(prompt="Test 1")
        evolution_api.spawn_child_llm(prompt="Test 2")
        evolution_api.spawn_child_llm(prompt="Test 3")

        # Create selections
        selections = [
            {"trial_id": "trial_0_0", "reasoning": "Best score", "category": "performance"},
            {"trial_id": "trial_0_2", "reasoning": "Novel approach", "category": "diversity"},
        ]

        evolution_api._advance_generation(selections=selections)

        gen0 = evolution_api.generations[0]
        assert len(gen0.trial_selections) == 2
        assert gen0.trial_selections[0].trial_id == "trial_0_0"
        assert gen0.trial_selections[0].reasoning == "Best score"
        assert gen0.trial_selections[1].category == "diversity"

    def test_advance_without_selections_uses_auto(self, evolution_api):
        """Test that advance without selections uses auto-selection."""
        evolution_api.spawn_child_llm(prompt="Test 1")
        evolution_api.spawn_child_llm(prompt="Test 2")

        evolution_api._advance_generation()  # No selections

        gen0 = evolution_api.generations[0]
        # Auto-selection should be used
        assert "Auto-selected" in gen0.selection_reasoning
        assert len(gen0.selected_trial_ids) > 0

    def test_advance_with_empty_selections_uses_auto(self, evolution_api):
        """Test that empty selections list uses auto-selection."""
        evolution_api.spawn_child_llm(prompt="Test 1")

        evolution_api._advance_generation(selections=[])

        gen0 = evolution_api.generations[0]
        assert "Auto-selected" in gen0.selection_reasoning

    def test_selections_recorded_in_summary(self, evolution_api):
        """Test that selections are recorded in generation summary."""
        evolution_api.spawn_child_llm(prompt="Test 1")
        evolution_api.spawn_child_llm(prompt="Test 2")

        selections = [
            {"trial_id": "trial_0_1", "reasoning": "Has potential", "category": "potential"},
        ]
        summary_text = "Selected trial with potential for improvement"

        evolution_api._advance_generation(selections=selections, selection_summary=summary_text)

        gen0 = evolution_api.generations[0]
        assert gen0.selection_reasoning == summary_text
        assert gen0.selected_trial_ids == ["trial_0_1"]

    def test_invalid_trial_id_in_selections_skipped(self, evolution_api):
        """Test that invalid trial IDs in selections are skipped."""
        evolution_api.spawn_child_llm(prompt="Test 1")

        selections = [
            {"trial_id": "trial_0_0", "reasoning": "Valid", "category": "performance"},
            {"trial_id": "trial_99_99", "reasoning": "Invalid ID", "category": "performance"},
        ]

        evolution_api._advance_generation(selections=selections)

        gen0 = evolution_api.generations[0]
        # Only valid trial should be in selections
        assert len(gen0.trial_selections) == 1
        assert gen0.trial_selections[0].trial_id == "trial_0_0"


class TestOrchestratorSelectionFlow:
    """Tests for orchestrator selection request and parsing."""

    @pytest.fixture
    def orchestrator(self, sample_config, temp_dir, sample_valid_packing_code):
        """Create an orchestrator for testing."""
        sample_config.experiment.output_dir = str(temp_dir)
        sample_config.evolution.max_generations = 2

        orchestrator = RootLLMOrchestrator(sample_config)
        cost_tracker = orchestrator.cost_tracker

        # Set up mock child LLM
        mock_child = MockLLMClient(
            model=sample_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
        )
        mock_child.set_responses(
            [
                f"```python\n{sample_valid_packing_code}\n```",
                f"```python\n{sample_valid_packing_code}\n```",
                f"```python\n{sample_valid_packing_code}\n```",
            ]
        )
        orchestrator.evolution_api.child_llm = mock_child

        return orchestrator

    def test_build_selection_request_message(self, orchestrator, sample_valid_packing_code):
        """Test building the selection request message."""
        # Spawn some children first
        orchestrator.evolution_api.spawn_child_llm(prompt="Test 1")
        orchestrator.evolution_api.spawn_child_llm(prompt="Test 2")

        message = orchestrator._build_selection_request_message()

        assert "Trial Results" in message
        assert "trial_0_0" in message
        assert "trial_0_1" in message
        assert "Your Task" in message or "Select" in message
        assert "performance" in message.lower() or "diversity" in message.lower()

    def test_parse_selection_response(self, orchestrator):
        """Test parsing a valid selection response."""
        response = """Based on the results, I select:

```selection
{
  "selections": [
    {"trial_id": "trial_0_0", "reasoning": "Best performance", "category": "performance"},
    {"trial_id": "trial_0_1", "reasoning": "Different strategy", "category": "diversity"}
  ],
  "summary": "Balanced selection"
}
```
"""
        selections, summary = orchestrator._parse_selection_response(response)

        assert selections is not None
        assert len(selections) == 2
        assert selections[0]["trial_id"] == "trial_0_0"
        assert summary == "Balanced selection"

    def test_parse_selection_response_no_block(self, orchestrator):
        """Test parsing response with no selection block returns None."""
        response = "Just thinking about the next generation..."

        selections, summary = orchestrator._parse_selection_response(response)

        assert selections is None
        assert summary is None

    def test_full_selection_flow(self, orchestrator, sample_config, sample_valid_packing_code):
        """Test full orchestration flow with selection."""
        cost_tracker = orchestrator.cost_tracker

        # Mock root LLM that spawns children and provides selection
        mock_root = MockLLMClient(
            model=sample_config.root_llm.model,
            cost_tracker=cost_tracker,
            llm_type="root",
        )

        # First response: spawn children
        # Second response: selection
        # Third response: spawn more children for gen 1
        # Fourth response: selection for gen 1
        mock_root.set_responses(
            [
                """Spawning children:
```repl
children = [{"prompt": "test1"}, {"prompt": "test2"}]
results = spawn_children_parallel(children)
for r in results:
    print(f"{r['trial_id']}: {r['metrics'].get('score', 0):.4f}")
```
""",
                """Based on results:
```selection
{
  "selections": [
    {"trial_id": "trial_0_0", "reasoning": "Best score", "category": "performance"}
  ],
  "summary": "Selected top performer"
}
```
""",
                """Generation 1:
```repl
children = [{"prompt": "improve"}]
results = spawn_children_parallel(children)
```
""",
                """Final selection:
```selection
{
  "selections": [
    {"trial_id": "trial_1_0", "reasoning": "Final best", "category": "performance"}
  ],
  "summary": "Final selection"
}
```
""",
            ]
        )

        orchestrator.root_llm = mock_root

        _result = orchestrator.run()

        # Check that selections were recorded
        gen0 = orchestrator.evolution_api.generations[0]
        assert len(gen0.trial_selections) >= 1
        assert gen0.trial_selections[0].trial_id == "trial_0_0"
        assert gen0.selection_reasoning == "Selected top performer"


class TestSelectionValidation:
    """Tests for selection validation logic."""

    @pytest.fixture
    def evolution_api(self, sample_config, temp_dir):
        """Create an EvolutionAPI instance."""
        sample_config.experiment.output_dir = str(temp_dir)

        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        mock_evaluator = MagicMock()
        call_count = [0]
        scores = [2.0, 1.5, 1.8]

        def mock_evaluate(code):
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return {
                "valid": True,
                "score": score,
                "eval_time": 0.1,
            }

        mock_evaluator.evaluate.side_effect = mock_evaluate

        child_llm = MockLLMClient(
            model=sample_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 10,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
            max_generations=5,
            max_children_per_generation=5,
        )

        return api

    def test_selection_validates_trial_exists(self, evolution_api):
        """Test that selection validates trial IDs exist."""
        evolution_api.spawn_child_llm(prompt="Test")

        # Try to select non-existent trial
        selections = [
            {"trial_id": "trial_5_0", "reasoning": "Does not exist", "category": "performance"},
        ]

        evolution_api._advance_generation(selections=selections)

        gen0 = evolution_api.generations[0]
        # Invalid selection should be filtered out, falling back to auto
        assert len(gen0.trial_selections) == 0 or gen0.trial_selections[0].trial_id != "trial_5_0"

    def test_selection_allows_failed_trials_with_potential(
        self, evolution_api, sample_config, temp_dir
    ):
        """Test that failed trials can be selected for 'potential' category."""
        # Create API with failing evaluator for some trials
        sample_config.experiment.output_dir = str(temp_dir)
        cost_tracker = CostTracker(sample_config)
        logger = ExperimentLogger(sample_config)
        logger.create_experiment_directory()

        mock_evaluator = MagicMock()
        call_count = [0]

        def mock_evaluate(code):
            call_count[0] += 1
            if call_count[0] == 2:
                return {"valid": False, "score": 0, "error": "overlap"}
            return {"valid": True, "score": 1.5}

        mock_evaluator.evaluate.side_effect = mock_evaluate

        child_llm = MockLLMClient(
            model=sample_config.child_llm.model,
            cost_tracker=cost_tracker,
            llm_type="child",
            responses=["```python\ndef run_packing(): pass\n```"] * 10,
        )

        api = EvolutionAPI(
            evaluator=mock_evaluator,
            child_llm=child_llm,
            cost_tracker=cost_tracker,
            logger=logger,
        )

        api.spawn_child_llm(prompt="Test 1")  # Valid
        api.spawn_child_llm(prompt="Test 2")  # Invalid

        # Select the invalid trial for its potential
        selections = [
            {
                "trial_id": "trial_0_1",
                "reasoning": "Interesting approach despite failure",
                "category": "potential",
            },
        ]

        api._advance_generation(selections=selections)

        gen0 = api.generations[0]
        # Should allow selecting failed trial with "potential" category
        assert any(s.trial_id == "trial_0_1" for s in gen0.trial_selections)
