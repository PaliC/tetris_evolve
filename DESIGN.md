# LLM-Driven Tetris Evolution System

## Design Document v2

### Executive Summary

This system evolves Tetris-playing programs using a combination of:
1. **AlphaEvolve's evolutionary approach** - LLMs propose code mutations, automated evaluation provides fitness scores
2. **Recursive LLM (RLM) architecture** - A root LLM orchestrates the evolution with complete autonomy, spawning child LLMs for code generation

**Key Innovation**: The root LLM has **complete freedom** over the evolution process. It decides what survives, how to mutate, when to terminate, and how to spawn child LLMs - all while operating within hard safety constraints.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EVOLUTION CONTROLLER                         │
│  (Enforces hard limits: max generations, max cost, max time)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           ROOT LLM (Depth=0)                        │
│                                                                     │
│  Operates in a Shared REPL Environment                             │
│  Has access to:                                                     │
│    - spawn_child_llm(prompt, parent_id) → code                     │
│    - evaluate_program(code) → metrics                              │
│    - get_population() → list of programs with metrics              │
│    - get_generation_history() → historical stats                   │
│    - advance_generation(selected_ids, reasoning) → void            │
│    - terminate_evolution(reason) → void                            │
│    - get_cost_remaining() → float                                  │
│                                                                     │
│  Decides:                                                           │
│    - Which programs survive to next generation                     │
│    - What prompts to give child LLMs                               │
│    - When to terminate (convergence, good enough, etc.)            │
│    - How many children to spawn per parent                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │ Child LLM 1 │ │ Child LLM 2 │ │ Child LLM N │
            │             │ │             │ │             │
            │ Focus:      │ │ Focus:      │ │ Focus:      │
            │ "Reduce     │ │ "Optimize   │ │ "Novel      │
            │  holes"     │ │  speed"     │ │  approach"  │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────────────────────────────────────┐
            │            TETRIS EVALUATOR                 │
            │                                             │
            │  - Runs code in sandboxed environment       │
            │  - Plays N games per program                │
            │  - Returns: avg_score, lines_cleared,       │
            │             survival_time, game_states      │
            └─────────────────────────────────────────────┘
```

---

## Directory Structure

All data is organized hierarchically: **Experiment → Generation → Trial**

```
experiments/
└── {experiment_id}/                    # e.g., "exp_2024_01_15_143022"
    ├── config.yaml                     # Frozen config for this experiment
    ├── experiment_stats.json           # Aggregate stats across all generations
    ├── cost_tracker.json               # Running cost totals
    │
    └── generations/
        └── gen_{N}/                    # e.g., "gen_001", "gen_002"
            ├── generation_stats.json   # Stats for this generation
            ├── root_llm_reasoning.md   # Root LLM's decision reasoning
            ├── selected_parents.json   # Which programs were selected to reproduce
            │
            └── trials/
                └── trial_{M}/          # e.g., "trial_001", "trial_002"
                    ├── code.py         # The generated code
                    ├── metrics.json    # Evaluation results
                    ├── reasoning.md    # Child LLM's reasoning/notes
                    ├── prompt.txt      # Prompt given to child LLM
                    ├── parent_id.txt   # Which trial this was derived from (if any)
                    └── llm_response.txt # Full LLM response for debugging
```

### Example Directory Tree

```
experiments/
└── exp_2024_01_15_143022/
    ├── config.yaml
    ├── experiment_stats.json
    ├── cost_tracker.json
    │
    └── generations/
        ├── gen_001/
        │   ├── generation_stats.json
        │   ├── root_llm_reasoning.md
        │   ├── selected_parents.json
        │   └── trials/
        │       ├── trial_001/
        │       │   ├── code.py
        │       │   ├── metrics.json
        │       │   ├── reasoning.md
        │       │   ├── prompt.txt
        │       │   └── llm_response.txt
        │       ├── trial_002/
        │       │   └── ...
        │       └── trial_003/
        │           └── ...
        │
        ├── gen_002/
        │   ├── generation_stats.json
        │   ├── root_llm_reasoning.md
        │   ├── selected_parents.json
        │   └── trials/
        │       ├── trial_004/
        │       │   ├── parent_id.txt      # Contains "trial_001"
        │       │   └── ...
        │       └── trial_005/
        │           └── ...
        │
        └── gen_003/
            └── ...
```

---

## Safety Constraints (Hard Limits)

These limits are enforced by the **Evolution Controller**, NOT the Root LLM. The Root LLM cannot exceed them.

| Constraint | Description | Enforcement |
|------------|-------------|-------------|
| `max_generations` | Maximum number of generations | Controller terminates after N generations |
| `max_cost_usd` | Maximum total cost across all LLM calls | Controller terminates when cost exceeded |
| `max_time_minutes` | Maximum wall-clock time | Controller terminates after T minutes |
| `max_children_per_generation` | Cap on child LLMs spawned per generation | Controller rejects excess spawn requests |

The Root LLM can query these limits and decide to terminate early, but **cannot exceed them**.

---

## Cost Tracking

### Token Cost Model (Hardcoded in Config)

```yaml
cost_per_1k_tokens:
  claude-sonnet-4:
    input: 0.003
    output: 0.015
  claude-haiku:
    input: 0.00025
    output: 0.00125
  gpt-4:
    input: 0.03
    output: 0.06
  gpt-4-turbo:
    input: 0.01
    output: 0.03
```

### Cost Tracker Schema (`cost_tracker.json`)

```json
{
  "experiment_id": "exp_2024_01_15_143022",
  "max_cost_usd": 100.0,
  "total_cost_usd": 12.45,
  "budget_remaining_usd": 87.55,
  "calls": [
    {
      "timestamp": "2024-01-15T14:30:05Z",
      "model": "claude-sonnet-4",
      "role": "root",
      "generation": 1,
      "trial_id": null,
      "input_tokens": 3500,
      "output_tokens": 1200,
      "cost_usd": 0.0285
    },
    {
      "timestamp": "2024-01-15T14:30:15Z",
      "model": "claude-sonnet-4",
      "role": "child",
      "generation": 1,
      "trial_id": "trial_001",
      "input_tokens": 2000,
      "output_tokens": 800,
      "cost_usd": 0.018
    }
  ],
  "summary": {
    "root_llm": {
      "calls": 15,
      "input_tokens": 45000,
      "output_tokens": 12000,
      "cost_usd": 4.23
    },
    "child_llm": {
      "calls": 120,
      "input_tokens": 180000,
      "output_tokens": 60000,
      "cost_usd": 8.22
    }
  },
  "per_generation": [
    {"generation": 1, "cost_usd": 2.15, "trials": 5},
    {"generation": 2, "cost_usd": 3.40, "trials": 8}
  ]
}
```

---

## Recording Reasoning

Every trial and generation records reasoning for auditability and learning.

### Trial-Level Reasoning (`reasoning.md`)

Each trial has a `reasoning.md` file containing the child LLM's explanation:

```markdown
# Trial 005 Reasoning

## Parent
- Parent ID: trial_001
- Parent Score: 1,234 points
- Parent Lines Cleared: 45

## Focus Area
Reduce hole creation while maintaining current performance.

## Strategy
1. Identified that the parent creates holes when placing S/Z pieces
2. Added a hole penalty term to the position evaluation function
3. Increased weight of hole penalty from 0.5 to 2.0
4. Added special case for S/Z pieces near the edges

## Expected Improvements
- Fewer holes per game (target: <5 vs current ~12)
- May slightly reduce speed due to more careful placement
- Should maintain or improve long-term stability

## Trade-offs Considered
- Higher hole penalty may cause suboptimal immediate placements
- Decided to accept this trade-off for better long-term board state
```

### Generation-Level Reasoning (`root_llm_reasoning.md`)

Each generation has the root LLM's strategic reasoning:

```markdown
# Generation 3 Root LLM Reasoning

## Current State Analysis
- Best score this generation: 2,500 (trial_007)
- Worst score: 800 (trial_010)
- Average improvement from gen 2: +15%
- Diversity score: 0.7 (good variety in approaches)

## Observations
1. Top 3 trials all use similar hole-minimization strategy
2. trial_007 introduced novel line-clearing lookahead - very promising
3. trial_010 tried aggressive stacking but failed badly

## Selection Decision
Advancing 3 programs to next generation:
- trial_007: Best overall, novel lookahead approach
- trial_006: Second best, solid hole management
- trial_009: Different strategy (speed-focused), maintaining diversity

## Child LLM Assignments

### For trial_007 (2 children):
1. **Child focus: "Extend lookahead"** - Current lookahead is 1 piece, try 2
2. **Child focus: "Optimize evaluation"** - Current eval function is slow

### For trial_006 (1 child):
1. **Child focus: "Combine with trial_007"** - Take trial_007's lookahead

### For trial_009 (1 child):
1. **Child focus: "Reduce risks"** - Speed strategy is risky, add safety

## Termination Consideration
Not terminating yet. 15% improvement is strong. Will reassess after gen 4.
If improvement drops below 5% for 2 consecutive generations, will consider
terminating.
```

---

## Core Components (Testable Chunks)

### Component 1: PufferLib Tetris Environment (External)

We use **PufferLib's built-in Tetris environment** rather than building our own.

```bash
pip install pufferlib
```

#### PufferLib Tetris Specifications

```python
from pufferlib.ocean.tetris import Tetris

# Create environment
env = Tetris(
    num_envs=1,           # Vectorized environments
    n_cols=10,            # Board width
    n_rows=20,            # Board height
    use_deck_obs=True,    # Include piece preview in observations
    n_noise_obs=10,       # Noise bits for curriculum learning
    n_init_garbage=4,     # Initial garbage lines
    render_mode=None,     # Set to enable rendering
)
```

#### Observation Space
- **Shape**: `(244,)` = 200 (board) + 6 (floats) + 28 (deck one-hot) + 10 (noise)
- **Type**: `float32`, values in [0, 2]

| Index Range | Content | Description |
|-------------|---------|-------------|
| 0-199 | Board state | 20x10 grid: 0=empty, 1=placed, 2=current piece |
| 200 | tick/10000 | Normalized game tick |
| 201 | tick_fall/ticks_per_fall | Fall timer progress |
| 202 | row/20 | Normalized current piece row |
| 203 | col/10 | Normalized current piece column |
| 204 | rotation | Current piece rotation (0-3) |
| 205 | can_swap | Can use hold (0 or 1) |
| 206-233 | Deck one-hot | 4 pieces × 7 types (current, 2 preview, hold) |
| 234-243 | Noise bits | Curriculum learning noise |

#### Action Space
- **Type**: `Discrete(7)`

| Action | Description |
|--------|-------------|
| 0 | No-op |
| 1 | Move left |
| 2 | Move right |
| 3 | Rotate clockwise |
| 4 | Soft drop |
| 5 | Hard drop |
| 6 | Hold piece |

#### Reward Structure
- Hard drop: +0.02 per row dropped
- Rotate: +0.01
- Line clears: +0.1 (1 line), +0.3 (2), +0.5 (3), +1.0 (4/Tetris)
- Invalid actions: 0 (no penalty)

#### Episode Termination
- Board overflow (can't spawn new piece)
- Max ticks reached (10,000)

#### Logging Info
Every `log_interval` steps (default 32), returns aggregated stats:
- `score`, `ep_length`, `ep_return`, `lines_deleted`
- `avg_combo`, `game_level`, action fractions

### Component 2: Program Evaluator

Safely executes evolved code and measures Tetris performance.

```python
# src/tetris_evolve/evaluation/evaluator.py

@dataclass
class EvaluationResult:
    """Results from evaluating a program."""
    trial_id: str
    success: bool
    error: Optional[str]

    # Metrics (None if error)
    games_played: int
    avg_score: Optional[float]
    avg_lines_cleared: Optional[float]
    avg_survival_steps: Optional[int]
    max_score: Optional[int]
    min_score: Optional[int]

    # Raw data for analysis
    game_results: list[dict]


class ProgramEvaluator:
    """Evaluates Tetris-playing programs."""

    def __init__(self, num_games: int = 10, max_steps: int = 10000):
        self.num_games = num_games
        self.max_steps = max_steps

    def evaluate(self, code: str, trial_id: str) -> EvaluationResult:
        """
        1. Validate code syntax and safety
        2. Execute in sandboxed environment
        3. Run num_games episodes
        4. Return aggregated metrics
        """
        pass

    def _validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """Check syntax and safety. Returns (valid, error_msg)."""
        pass

    def _execute_safely(self, code: str) -> Optional[Callable]:
        """Execute code and extract the player function."""
        pass

    def _run_game(self, player: Callable) -> dict:
        """Run a single game and return results."""
        pass
```

### Component 3: Cost Tracker

Tracks all LLM costs with per-call granularity.

```python
# src/tetris_evolve/tracking/cost_tracker.py

@dataclass
class LLMCall:
    """Record of a single LLM API call."""
    timestamp: datetime
    model: str
    role: str  # "root" or "child"
    generation: int
    trial_id: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Tracks LLM costs and enforces budget limits."""

    def __init__(self, max_cost_usd: float, cost_config: dict):
        self.max_cost_usd = max_cost_usd
        self.cost_config = cost_config  # Per-model token costs
        self.calls: list[LLMCall] = []

    def record_call(
        self,
        model: str,
        role: str,
        generation: int,
        input_tokens: int,
        output_tokens: int,
        trial_id: Optional[str] = None
    ) -> LLMCall:
        """Record a call and return the LLMCall with computed cost."""
        pass

    def get_total_cost(self) -> float:
        """Total cost so far."""
        pass

    def get_remaining_budget(self) -> float:
        """Remaining budget."""
        pass

    def would_exceed_budget(self, estimated_cost: float) -> bool:
        """Check if an estimated cost would exceed budget."""
        pass

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        pass

    @classmethod
    def load(cls, path: Path, cost_config: dict) -> "CostTracker":
        """Load from JSON file."""
        pass
```

### Component 4: Experiment Tracker

Manages the experiment directory structure and persists all data.

```python
# src/tetris_evolve/tracking/experiment_tracker.py

@dataclass
class TrialData:
    """All data for a single trial."""
    trial_id: str
    generation: int
    parent_id: Optional[str]
    code: str
    prompt: str
    reasoning: str
    llm_response: str
    metrics: Optional[EvaluationResult]


@dataclass
class GenerationData:
    """All data for a generation."""
    generation: int
    trials: list[TrialData]
    selected_parent_ids: list[str]
    root_reasoning: str
    stats: dict


class ExperimentTracker:
    """Manages experiment data persistence."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.current_generation = 0

    def create_experiment(self, config: dict) -> str:
        """Create experiment directory, return experiment_id."""
        pass

    def start_generation(self, generation: int) -> Path:
        """Create generation directory, return path."""
        pass

    def save_trial(self, trial: TrialData) -> Path:
        """Save trial data to disk, return trial directory."""
        pass

    def complete_generation(
        self,
        generation: int,
        selected_ids: list[str],
        root_reasoning: str,
        stats: dict
    ) -> None:
        """Save generation summary and stats."""
        pass

    def get_trial(self, trial_id: str) -> TrialData:
        """Load trial data from disk."""
        pass

    def get_generation_trials(self, generation: int) -> list[TrialData]:
        """Get all trials for a generation."""
        pass

    def update_experiment_stats(self, stats: dict) -> None:
        """Update experiment-level aggregate stats."""
        pass
```

### Component 5: LLM Client

Wrapper for LLM API with token counting and cost tracking.

```python
# src/tetris_evolve/llm/client.py

@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient:
    """Wrapper for LLM API calls."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def send_message(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Send message and return response with token counts."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
```

### Component 6: Child LLM Executor

Generates code from prompts provided by the root LLM.

```python
# src/tetris_evolve/llm/child_llm.py

@dataclass
class ChildLLMResult:
    """Result from child LLM code generation."""
    code: str
    reasoning: str
    raw_response: str
    tokens_used: tuple[int, int]  # (input, output)


class ChildLLMExecutor:
    """Executes child LLM to generate code."""

    def __init__(self, client: LLMClient, evaluator: ProgramEvaluator):
        self.client = client
        self.evaluator = evaluator

    def generate(
        self,
        prompt: str,
        parent_code: Optional[str] = None
    ) -> ChildLLMResult:
        """
        Generate code based on prompt.

        Args:
            prompt: Instructions from root LLM
            parent_code: Code to mutate (if any)

        Returns:
            Generated code, reasoning, and metadata
        """
        pass

    def _build_system_prompt(self) -> str:
        """Build system prompt for child LLM."""
        pass

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        pass

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from LLM response."""
        pass
```

### Component 7: Root LLM Interface

The REPL functions available to the root LLM.

```python
# src/tetris_evolve/llm/root_llm.py

@dataclass
class PopulationMember:
    """Summary of a program in the population."""
    trial_id: str
    generation: int
    parent_id: Optional[str]
    metrics: EvaluationResult
    code_preview: str  # First 500 chars


class RootLLMInterface:
    """Functions available to the root LLM in its REPL."""

    def __init__(
        self,
        child_executor: ChildLLMExecutor,
        evaluator: ProgramEvaluator,
        experiment_tracker: ExperimentTracker,
        cost_tracker: CostTracker,
        config: dict
    ):
        self.child_executor = child_executor
        self.evaluator = evaluator
        self.tracker = experiment_tracker
        self.cost_tracker = cost_tracker
        self.config = config
        self._terminated = False
        self._children_this_gen = 0

    def spawn_child_llm(
        self,
        prompt: str,
        parent_id: Optional[str] = None
    ) -> dict:
        """
        Spawn a child LLM to generate code.

        Args:
            prompt: Complete instructions for the child
            parent_id: Trial ID of parent (for mutations)

        Returns:
            Dict with trial_id, code, metrics, reasoning

        Raises:
            ResourceLimitError: If max children exceeded
        """
        pass

    def evaluate_program(self, code: str) -> dict:
        """Evaluate code without creating a trial."""
        pass

    def get_population(self) -> list[PopulationMember]:
        """Get current generation's programs."""
        pass

    def get_generation_history(self) -> list[dict]:
        """Get stats from all previous generations."""
        pass

    def advance_generation(
        self,
        selected_trial_ids: list[str],
        reasoning: str
    ) -> int:
        """
        Advance to next generation.

        Args:
            selected_trial_ids: Trials to use as parents
            reasoning: Why these were selected

        Returns:
            New generation number
        """
        pass

    def terminate_evolution(self, reason: str) -> dict:
        """End evolution and return summary."""
        pass

    def get_cost_remaining(self) -> float:
        """Get remaining budget in USD."""
        pass

    def get_limits(self) -> dict:
        """Get hard limits (max_gen, max_cost, etc.)."""
        pass
```

### Component 8: Evolution Controller

The main orchestrator that runs the evolution loop and enforces limits.

```python
# src/tetris_evolve/evolution/controller.py

@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    max_generations: int
    max_cost_usd: float
    max_time_minutes: int
    max_children_per_generation: int
    initial_population_size: int
    games_per_evaluation: int
    root_model: str
    child_model: str
    output_dir: Path


@dataclass
class EvolutionResult:
    """Final result of evolution run."""
    experiment_id: str
    generations_completed: int
    total_cost_usd: float
    best_trial_id: str
    best_score: float
    best_code: str
    termination_reason: str


class EvolutionController:
    """Main controller for evolution loop."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.start_time = None
        self.cost_tracker = None
        self.experiment_tracker = None
        self.root_interface = None

    def run(self) -> EvolutionResult:
        """
        Main evolution loop:

        1. Initialize experiment
        2. Generate initial population
        3. While not terminated:
           a. Check hard limits
           b. Give control to root LLM
           c. Record decisions and results
        4. Return final results
        """
        pass

    def _check_hard_limits(self) -> Optional[str]:
        """Check limits, return termination reason if exceeded."""
        pass

    def _run_root_llm_turn(self) -> bool:
        """Execute one turn of root LLM. Returns False if terminated."""
        pass

    def _create_initial_population(self) -> None:
        """Generate initial population using child LLMs."""
        pass
```

---

## Test-Driven Development Plan

Each component is built using TDD: write tests first, then implement.

**Note**: We use **PufferLib's Tetris** (no custom environment needed), reducing Phase 1.

### Phase 1: Core Infrastructure

| # | Component | Test File | Key Tests |
|---|-----------|-----------|-----------|
| 1 | Program Evaluator | `tests/test_evaluator.py` | valid_code, syntax_error, runtime_error, timeout, unsafe_code |
| 2 | Cost Tracker | `tests/test_cost_tracker.py` | record_call, total_cost, budget_check, save/load |
| 3 | Experiment Tracker | `tests/test_experiment_tracker.py` | create_experiment, save_trial, complete_generation |

### Phase 2: LLM Integration

| # | Component | Test File | Key Tests |
|---|-----------|-----------|-----------|
| 4 | LLM Client | `tests/test_llm_client.py` | send_message, token_counting, error_handling |
| 5 | Child LLM Executor | `tests/test_child_llm.py` | generate_code, extract_code, extract_reasoning |
| 6 | Root LLM Interface | `tests/test_root_llm.py` | spawn_child, get_population, advance_gen, terminate |

### Phase 3: Evolution Loop

| # | Component | Test File | Key Tests |
|---|-----------|-----------|-----------|
| 7 | Evolution Controller | `tests/test_controller.py` | init, generation_loop, hard_limits, termination |
| 8 | Integration | `tests/test_integration.py` | child_to_eval_pipeline, root_spawns_children |
| 9 | E2E | `tests/test_e2e.py` | mock_evolution_run, resume_experiment |

---

## Detailed Test Specifications

### Chunk 1: Program Evaluator Tests

```python
# tests/test_evaluator.py

def test_evaluate_valid_code():
    """Valid code runs and returns metrics."""
    evaluator = ProgramEvaluator(num_games=3)
    code = '''
def choose_action(obs):
    return 5  # Always hard drop
'''
    result = evaluator.evaluate(code, "test_001")
    assert result.success is True
    assert result.games_played == 3
    assert result.avg_score is not None

def test_evaluate_syntax_error():
    """Syntax errors are caught and reported."""
    evaluator = ProgramEvaluator()
    code = 'def choose_action(obs) return 5'  # Missing colon
    result = evaluator.evaluate(code, "test_002")
    assert result.success is False
    assert "syntax" in result.error.lower()

def test_evaluate_runtime_error():
    """Runtime errors are caught and reported."""
    evaluator = ProgramEvaluator()
    code = '''
def choose_action(obs):
    return 1 / 0  # Division by zero
'''
    result = evaluator.evaluate(code, "test_003")
    assert result.success is False
    assert result.error is not None

def test_evaluate_timeout():
    """Infinite loops are terminated."""
    evaluator = ProgramEvaluator(timeout_seconds=1)
    code = '''
def choose_action(obs):
    while True:
        pass
'''
    result = evaluator.evaluate(code, "test_004")
    assert result.success is False
    assert "timeout" in result.error.lower()

def test_evaluate_unsafe_code():
    """Dangerous operations are blocked."""
    evaluator = ProgramEvaluator()
    code = '''
import os
def choose_action(obs):
    os.system("rm -rf /")
    return 0
'''
    result = evaluator.evaluate(code, "test_005")
    assert result.success is False
```

### Chunk 3: Cost Tracker Tests

```python
# tests/test_cost_tracker.py

COST_CONFIG = {
    "claude-sonnet-4": {"input": 0.003, "output": 0.015}
}

def test_record_call():
    """Recording a call calculates cost correctly."""
    tracker = CostTracker(max_cost_usd=100.0, cost_config=COST_CONFIG)
    call = tracker.record_call(
        model="claude-sonnet-4",
        role="root",
        generation=1,
        input_tokens=1000,
        output_tokens=500
    )
    expected_cost = (1000 * 0.003 / 1000) + (500 * 0.015 / 1000)
    assert abs(call.cost_usd - expected_cost) < 0.001

def test_total_cost():
    """Total cost sums all calls."""
    tracker = CostTracker(max_cost_usd=100.0, cost_config=COST_CONFIG)
    tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)
    tracker.record_call("claude-sonnet-4", "child", 1, 2000, 1000)
    assert tracker.get_total_cost() > 0

def test_budget_exceeded():
    """Detects when budget would be exceeded."""
    tracker = CostTracker(max_cost_usd=0.01, cost_config=COST_CONFIG)
    tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)
    assert tracker.would_exceed_budget(0.01) is True

def test_save_and_load(tmp_path):
    """Tracker can be saved and loaded."""
    tracker = CostTracker(max_cost_usd=100.0, cost_config=COST_CONFIG)
    tracker.record_call("claude-sonnet-4", "root", 1, 1000, 500)

    path = tmp_path / "cost_tracker.json"
    tracker.save(path)

    loaded = CostTracker.load(path, COST_CONFIG)
    assert loaded.get_total_cost() == tracker.get_total_cost()
```

### Chunk 4: Experiment Tracker Tests

```python
# tests/test_experiment_tracker.py

def test_create_experiment(tmp_path):
    """Creates experiment directory with config."""
    tracker = ExperimentTracker(tmp_path)
    config = {"max_generations": 10}
    exp_id = tracker.create_experiment(config)

    assert (tmp_path / "config.yaml").exists()
    assert exp_id.startswith("exp_")

def test_save_trial(tmp_path):
    """Saves trial with all components."""
    tracker = ExperimentTracker(tmp_path)
    tracker.create_experiment({})
    tracker.start_generation(1)

    trial = TrialData(
        trial_id="trial_001",
        generation=1,
        parent_id=None,
        code="def choose_action(obs): return 0",
        prompt="Create a Tetris player",
        reasoning="Simple baseline approach",
        llm_response="Here is the code...",
        metrics=None
    )

    trial_path = tracker.save_trial(trial)
    assert (trial_path / "code.py").exists()
    assert (trial_path / "reasoning.md").exists()
    assert (trial_path / "prompt.txt").exists()

def test_complete_generation(tmp_path):
    """Saves generation summary."""
    tracker = ExperimentTracker(tmp_path)
    tracker.create_experiment({})
    tracker.start_generation(1)

    tracker.complete_generation(
        generation=1,
        selected_ids=["trial_001", "trial_002"],
        root_reasoning="Selected top performers",
        stats={"avg_score": 1000}
    )

    gen_path = tmp_path / "generations" / "gen_001"
    assert (gen_path / "generation_stats.json").exists()
    assert (gen_path / "root_llm_reasoning.md").exists()
```

---

## Implementation Order

Build in this order to minimize dependencies:

```
Phase 1: Core Infrastructure (No LLM dependencies)
├── 1. Program Evaluator (depends on: PufferLib Tetris)
├── 2. Cost Tracker (standalone)
└── 3. Experiment Tracker (standalone)

Phase 2: LLM Integration (Mocked for unit tests)
├── 4. LLM Client (standalone, easily mocked)
├── 5. Child LLM Executor (depends on: LLM Client, Evaluator)
└── 6. Root LLM Interface (depends on: Child Executor, Trackers)

Phase 3: Integration
├── 7. Evolution Controller (depends on: all above)
├── 8. Integration Tests (full pipeline)
└── 9. E2E Tests (mock LLM, full evolution)
```

**Dependency**: `pip install pufferlib` (provides Tetris environment)

---

## Configuration Schema

```yaml
# config.yaml

experiment:
  name: "tetris_evolution_v1"
  output_dir: "./experiments"

limits:
  max_generations: 50
  max_cost_usd: 100.0
  max_time_minutes: 120
  max_children_per_generation: 20

evaluation:
  games_per_program: 10
  max_steps_per_game: 10000
  timeout_seconds: 30

llm:
  root:
    model: "claude-sonnet-4"
    temperature: 0.7
    max_tokens: 4096
  child:
    model: "claude-sonnet-4"
    temperature: 0.8
    max_tokens: 2048

cost:
  # Per 1K tokens
  claude-sonnet-4:
    input: 0.003
    output: 0.015
  claude-haiku:
    input: 0.00025
    output: 0.00125

initial_population:
  size: 5
  strategies:
    - "Create a basic Tetris player that minimizes holes"
    - "Create a Tetris player focused on clearing lines quickly"
    - "Create a Tetris player that keeps the board flat"
    - "Create a Tetris player using lookahead"
    - "Create a random baseline Tetris player"

logging:
  level: "INFO"
  save_llm_responses: true
```

---

## Root LLM System Prompt

```markdown
You are the Root LLM controlling the evolution of Tetris-playing programs.

## Your Authority
You have COMPLETE FREEDOM over evolution. You decide:
- Which programs survive
- What prompts to give children
- When to stop
- How many children per parent

## Available Functions

```python
# Spawn child LLM to generate/mutate code
result = spawn_child_llm(
    prompt="Focus on reducing holes",
    parent_id="trial_003"  # Optional
)
# Returns: {"trial_id": "trial_007", "code": "...", "metrics": {...}}

# Get current population
population = get_population()
# Returns: [{"trial_id": "...", "metrics": {...}, "code_preview": "..."}, ...]

# Get history
history = get_generation_history()
# Returns: [{"generation": 1, "best_score": 1000, "avg_score": 500}, ...]

# Advance to next generation
advance_generation(
    selected_trial_ids=["trial_003", "trial_007"],
    reasoning="Selected for high scores and diversity"
)

# End evolution
terminate_evolution(reason="Converged at score 5000")

# Check budget
remaining = get_cost_remaining()
limits = get_limits()
```

## Hard Limits (Cannot Exceed)
- Max generations: {max_generations}
- Max cost: ${max_cost}
- Max children per generation: {max_children}

## Current State
Generation: {current_gen}
Cost used: ${cost_used} / ${max_cost}
Best score: {best_score}

## Instructions
1. Analyze current population
2. Decide: continue or terminate?
3. If continuing: select parents, spawn children with focused prompts
4. Call advance_generation() when ready for next gen
5. Provide reasoning for all decisions
```

---

## Child LLM Prompt Template

```markdown
You are generating a Tetris-playing program.

## Task
{prompt_from_root}

## Parent Code (if mutation)
```python
{parent_code}
```

## Interface
Your code must implement:

```python
def choose_action(observation: np.ndarray) -> int:
    """
    PufferLib Tetris observation format.

    Args:
        observation: np.ndarray of shape (244,), dtype float32
            - [0:200]   Board state (20x10 flattened): 0=empty, 1=placed, 2=current piece
            - [200]     tick / 10000 (normalized game progress)
            - [201]     tick_fall / ticks_per_fall (fall timer)
            - [202]     current_row / 20 (normalized piece row)
            - [203]     current_col / 10 (normalized piece column)
            - [204]     rotation (0-3)
            - [205]     can_swap (0 or 1, whether hold is available)
            - [206:234] Deck one-hot (4 pieces × 7 types)
            - [234:244] Noise bits (ignore for decision making)

    Returns:
        action: int (0-6)
            0 = no-op
            1 = move left
            2 = move right
            3 = rotate clockwise
            4 = soft drop
            5 = hard drop
            6 = hold piece
    """
```

## Helper Functions (provided to all generated code)

```python
import numpy as np

def parse_observation(obs: np.ndarray) -> dict:
    """Parse PufferLib observation into readable components."""
    return {
        "board": obs[0:200].reshape(20, 10),  # 0=empty, 1=placed, 2=current
        "tick_progress": obs[200],
        "fall_timer": obs[201],
        "piece_row": int(obs[202] * 20),
        "piece_col": int(obs[203] * 10),
        "rotation": int(obs[204]),
        "can_hold": bool(obs[205]),
        "current_piece": np.argmax(obs[206:213]),  # One-hot to index
        "next_pieces": [np.argmax(obs[213+i*7:220+i*7]) for i in range(2)],
        "hold_piece": np.argmax(obs[227:234]) if obs[227:234].sum() > 0 else None,
    }

# Piece names for reference
PIECES = ["O", "I", "S", "Z", "T", "L", "J"]
```

## Output Format
Provide your reasoning in <reasoning> tags, then code in <code> tags:

<reasoning>
Explain your approach...
</reasoning>

<code>
import numpy as np

def choose_action(observation):
    # Your implementation here
    ...
</code>
```

---

## Summary

This design implements an LLM-driven evolution system where:

1. **Root LLM has complete autonomy** within hard limits
2. **All decisions are recorded** with reasoning at trial and generation levels
3. **Cost is tracked** per-call with budget enforcement
4. **Data is organized** as Experiment → Generation → Trial
5. **Built with TDD** in 9 testable chunks
6. **Uses PufferLib's Tetris** for fast, vectorized evaluation

The system combines AlphaEvolve's evolutionary approach with Recursive LLM's hierarchical control to evolve increasingly better Tetris-playing code.
