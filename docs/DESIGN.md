# LLM Tetris Optimizer - Design Document

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Specifications](#component-specifications)
4. [Data Structures](#data-structures)
5. [REPL API](#repl-api)
6. [Configuration](#configuration)
7. [Observability & Logging](#observability--logging)
8. [Tetris Integration](#tetris-integration)
9. [Cost Management](#cost-management)
10. [Implementation Guide](#implementation-guide)
11. [Testing Strategy](#testing-strategy)
12. [Detailed TODO List](#detailed-todo-list)

---

## Overview

### Goal
Create an LLM-driven evolutionary system that generates code to play Tetris optimally. The system combines:
- **AlphaEvolve**: Evolutionary approach to code generation with program database and evaluation loops
- **RLM (Recursive LLM)**: LLM-driven decision making for trial selection and generation advancement

### Core Innovation
Instead of algorithmic selection mechanisms (like MAP-Elites), a **Root LLM** decides which trials to advance to the next generation. The Root LLM has access to a REPL environment with functions for spawning child LLMs, evaluating code, and managing the evolutionary process.

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT LIFECYCLE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Load Config → 2. Initialize Root LLM → 3. Start REPL Loop          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ROOT LLM REPL LOOP                            │   │
│  │                                                                  │   │
│  │   Root LLM receives: initial prompt + REPL functions            │   │
│  │        │                                                        │   │
│  │        ▼                                                        │   │
│  │   Root LLM outputs Python code blocks                           │   │
│  │        │                                                        │   │
│  │        ▼                                                        │   │
│  │   REPL executes code, returns results                           │   │
│  │        │                                                        │   │
│  │        ▼                                                        │   │
│  │   Root LLM reasons about results, outputs more code             │   │
│  │        │                                                        │   │
│  │        ▼                                                        │   │
│  │   Loop until: terminate_evolution() called OR budget exhausted  │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  4. Save Results → 5. Generate Report                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              tetris_evolve                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   CLI/Runner    │────►│  Orchestrator   │────►│   Root LLM      │       │
│  │                 │     │                 │     │   Controller    │       │
│  └─────────────────┘     └────────┬────────┘     └────────┬────────┘       │
│                                   │                       │                 │
│                                   ▼                       ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │    Config       │     │  Experiment     │     │     REPL        │       │
│  │    Loader       │     │    Store        │◄───►│   Environment   │       │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘       │
│                                                           │                 │
│                          ┌────────────────────────────────┼────────┐       │
│                          │                                │        │       │
│                          ▼                                ▼        ▼       │
│                 ┌─────────────────┐              ┌──────────────────────┐  │
│                 │   Child LLM     │              │    Tetris Evaluator  │  │
│                 │   Spawner       │              │    (PufferLib)       │  │
│                 └─────────────────┘              └──────────────────────┘  │
│                          │                                                 │
│                          ▼                                                 │
│                 ┌─────────────────┐                                        │
│                 │   Cost Tracker  │                                        │
│                 └─────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/
├── tetris_evolve/
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── orchestrator.py         # Main experiment orchestrator
│   ├── config.py               # Configuration loading and validation
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── root_controller.py  # Root LLM management
│   │   ├── child_spawner.py    # Child LLM spawning
│   │   ├── client.py           # LLM API client abstraction
│   │   └── cost_tracker.py     # Token and cost tracking
│   │
│   ├── repl/
│   │   ├── __init__.py
│   │   ├── environment.py      # REPL execution environment
│   │   ├── functions.py        # Built-in REPL functions
│   │   └── sandbox.py          # Code execution sandbox
│   │
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── experiment.py       # Experiment data structures
│   │   ├── generation.py       # Generation management
│   │   └── trial.py            # Trial data structures
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── tetris_evaluator.py # Tetris game evaluation
│   │   └── metrics.py          # Metric collection
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── experiment_store.py # Experiment persistence
│   │   └── schemas.py          # Data schemas
│   │
│   └── logging/
│       ├── __init__.py
│       ├── logger.py           # Structured logging
│       └── formatters.py       # Log formatters
│
tests/
├── unit/
│   ├── test_config.py
│   ├── test_cost_tracker.py
│   ├── test_repl_environment.py
│   ├── test_sandbox.py
│   ├── test_experiment_store.py
│   └── test_tetris_evaluator.py
│
├── integration/
│   ├── test_child_spawning.py
│   ├── test_evolution_flow.py
│   └── test_full_experiment.py
│
└── e2e/
    └── test_tetris_optimization.py
```

---

## Component Specifications

### 1. CLI/Runner (`cli.py`)

**Purpose**: Entry point for running experiments.

```python
# Usage examples:
# tetris-evolve run config.yaml
# tetris-evolve run config.yaml --output ./results
# tetris-evolve resume ./results/experiment_123
# tetris-evolve report ./results/experiment_123
```

**Interface**:
```python
def run_experiment(config_path: str, output_dir: str | None = None) -> ExperimentResult:
    """Run a new experiment from config."""
    pass

def resume_experiment(experiment_dir: str) -> ExperimentResult:
    """Resume an interrupted experiment."""
    pass

def generate_report(experiment_dir: str) -> str:
    """Generate human-readable report from experiment."""
    pass
```

### 2. Orchestrator (`orchestrator.py`)

**Purpose**: Coordinates the entire experiment lifecycle.

**Responsibilities**:
- Load and validate configuration
- Initialize all components
- Start the Root LLM REPL loop
- Handle interruptions and cleanup
- Persist final results

```python
class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.experiment_store = ExperimentStore(config.output_dir)
        self.cost_tracker = CostTracker(config.max_cost)
        self.root_controller = RootLLMController(config, self.cost_tracker)
        self.repl_env = REPLEnvironment(self._create_repl_functions())

    def run(self) -> ExperimentResult:
        """Main entry point."""
        experiment = self.experiment_store.create_experiment(self.config)

        try:
            self.root_controller.start(
                initial_prompt=self.config.initial_prompt,
                repl_env=self.repl_env
            )
        except BudgetExhaustedError:
            # Handle graceful shutdown
            pass
        finally:
            return self.experiment_store.finalize(experiment)
```

### 3. Root LLM Controller (`root_controller.py`)

**Purpose**: Manages the Root LLM's interaction with the REPL.

**Key Behaviors**:
- Sends initial prompt to Root LLM
- Parses code blocks from LLM output
- Executes code in REPL environment
- Returns execution results to LLM
- Tracks conversation history

```python
class RootLLMController:
    def __init__(self, config: Config, cost_tracker: CostTracker):
        self.client = LLMClient(
            model=config.root_lm_model,
            cost_per_input_token=config.root_lm_cost.input,
            cost_per_output_token=config.root_lm_cost.output
        )
        self.cost_tracker = cost_tracker
        self.messages: list[Message] = []

    def start(self, initial_prompt: str, repl_env: REPLEnvironment) -> None:
        """Start the REPL loop with the Root LLM."""
        # System message with REPL function documentation
        system_msg = self._build_system_message(repl_env.get_function_docs())

        # Initial user message with the prompt
        self.messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": initial_prompt}
        ]

        while True:
            # Get LLM response
            response, usage = self.client.chat(self.messages)
            self.cost_tracker.record(usage)

            # Parse and execute code blocks
            code_blocks = self._extract_code_blocks(response.content)

            if not code_blocks:
                # No code to execute, might be final reasoning
                continue

            # Execute each code block
            results = []
            for code in code_blocks:
                result = repl_env.execute(code)
                results.append(result)

                # Check for termination
                if result.terminated:
                    return

            # Add assistant message and execution results
            self.messages.append({"role": "assistant", "content": response.content})
            self.messages.append({
                "role": "user",
                "content": self._format_execution_results(results)
            })
```

### 4. REPL Environment (`repl/environment.py`)

**Purpose**: Executes Python code from the Root LLM in a controlled environment.

```python
class REPLEnvironment:
    def __init__(self, functions: dict[str, Callable]):
        self.functions = functions
        self.variables: dict[str, Any] = {}
        self.execution_count = 0

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in sandboxed environment."""
        self.execution_count += 1

        # Create execution namespace with built-in functions
        namespace = {
            **self.functions,
            **self.variables,
            '__builtins__': self._get_safe_builtins()
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code, namespace)

            # Update variables (excluding functions and builtins)
            self.variables = {
                k: v for k, v in namespace.items()
                if not k.startswith('_') and k not in self.functions
            }

            return ExecutionResult(
                success=True,
                stdout=self._truncate(stdout_capture.getvalue()),
                stderr=stderr_capture.getvalue(),
                variables_updated=list(self.variables.keys())
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )

    def get_function_docs(self) -> str:
        """Generate documentation for available functions."""
        docs = []
        for name, func in self.functions.items():
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "No documentation"
            docs.append(f"{name}{sig}\n    {doc}")
        return "\n\n".join(docs)
```

### 5. Child LLM Spawner (`child_spawner.py`)

**Purpose**: Creates and manages Child LLM instances for code generation.

```python
class ChildLLMSpawner:
    def __init__(self, config: Config, cost_tracker: CostTracker,
                 experiment_store: ExperimentStore):
        self.config = config
        self.cost_tracker = cost_tracker
        self.experiment_store = experiment_store
        self.client = LLMClient(
            model=config.child_lm_model,
            cost_per_input_token=config.child_lm_cost.input,
            cost_per_output_token=config.child_lm_cost.output
        )

    def spawn(self, prompt: str, parent_id: str | None = None) -> TrialResult:
        """
        Spawn a Child LLM to generate code.

        Args:
            prompt: The prompt for code generation
            parent_id: Optional parent trial ID for lineage tracking

        Returns:
            TrialResult with generated code and metadata
        """
        trial_id = self._generate_trial_id()

        try:
            # Check budget before spawning
            self.cost_tracker.check_budget()

            # Call the Child LLM
            response, usage = self.client.chat([
                {"role": "user", "content": prompt}
            ])

            # Record costs
            self.cost_tracker.record(usage)

            # Extract code from response
            code = self._extract_code(response.content)

            # Create trial record
            trial = Trial(
                id=trial_id,
                parent_id=parent_id,
                prompt=prompt,
                response=response.content,
                code=code,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=code is not None
            )

            self.experiment_store.record_trial(trial)

            return TrialResult(
                trial_id=trial_id,
                code=code,
                reasoning=response.content,
                success=code is not None,
                error=None if code else "Failed to extract code from response"
            )

        except BudgetExhaustedError as e:
            return TrialResult(
                trial_id=trial_id,
                code=None,
                reasoning=None,
                success=False,
                error=str(e)
            )
```

### 6. Cost Tracker (`cost_tracker.py`)

**Purpose**: Tracks token usage and costs across all LLM calls.

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

@dataclass
class CostRecord:
    timestamp: datetime
    source: str  # "root" or "child"
    usage: TokenUsage
    cost: float

class CostTracker:
    def __init__(self, max_cost: float):
        self.max_cost = max_cost
        self.records: list[CostRecord] = []
        self._lock = threading.Lock()

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(r.usage.input_tokens for r in self.records),
            "output": sum(r.usage.output_tokens for r in self.records)
        }

    def record(self, usage: TokenUsage, source: str, cost_per_input: float,
               cost_per_output: float) -> None:
        """Record token usage and associated cost."""
        cost = (usage.input_tokens * cost_per_input +
                usage.output_tokens * cost_per_output)

        with self._lock:
            self.records.append(CostRecord(
                timestamp=datetime.now(),
                source=source,
                usage=usage,
                cost=cost
            ))

    def check_budget(self) -> None:
        """Raise BudgetExhaustedError if budget exceeded."""
        if self.total_cost >= self.max_cost:
            raise BudgetExhaustedError(
                f"Budget exhausted: ${self.total_cost:.4f} >= ${self.max_cost:.4f}"
            )

    def get_summary(self) -> dict:
        """Get cost summary for reporting."""
        return {
            "total_cost": self.total_cost,
            "max_cost": self.max_cost,
            "remaining": self.max_cost - self.total_cost,
            "total_tokens": self.total_tokens,
            "by_source": self._group_by_source()
        }
```

### 7. Tetris Evaluator (`tetris_evaluator.py`)

**Purpose**: Evaluates generated Tetris-playing code using PufferLib/Tetris-Gymnasium.

```python
class TetrisEvaluator:
    def __init__(self, num_games: int = 10, max_steps_per_game: int = 10000):
        self.num_games = num_games
        self.max_steps_per_game = max_steps_per_game

    def evaluate(self, code: str, num_games: int | None = None) -> EvaluationResult:
        """
        Evaluate Tetris-playing code.

        Args:
            code: Python code that defines a `select_action(observation, info)` function
            num_games: Number of games to play (overrides default)

        Returns:
            EvaluationResult with metrics
        """
        num_games = num_games or self.num_games

        try:
            # Load the agent function from code
            agent_fn = self._load_agent(code)

            # Run games and collect metrics
            game_results = []
            for seed in range(num_games):
                result = self._run_game(agent_fn, seed)
                game_results.append(result)

            return EvaluationResult(
                success=True,
                metrics=self._aggregate_metrics(game_results),
                game_results=game_results
            )

        except Exception as e:
            return EvaluationResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                metrics=None,
                game_results=[]
            )

    def _load_agent(self, code: str) -> Callable:
        """Load agent function from code string."""
        namespace = {}
        exec(code, namespace)

        if 'select_action' not in namespace:
            raise ValueError("Code must define 'select_action(observation, info)' function")

        return namespace['select_action']

    def _run_game(self, agent_fn: Callable, seed: int) -> GameResult:
        """Run a single Tetris game."""
        import gymnasium as gym

        env = gym.make("tetris_gymnasium/Tetris")
        observation, info = env.reset(seed=seed)

        total_reward = 0
        lines_cleared = 0
        steps = 0

        while steps < self.max_steps_per_game:
            action = agent_fn(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            lines_cleared += info.get('lines_cleared', 0)
            steps += 1

            if terminated or truncated:
                break

        env.close()

        return GameResult(
            seed=seed,
            score=total_reward,
            lines_cleared=lines_cleared,
            steps=steps,
            terminated=terminated
        )

    def _aggregate_metrics(self, results: list[GameResult]) -> dict:
        """Aggregate metrics across games."""
        scores = [r.score for r in results]
        lines = [r.lines_cleared for r in results]
        steps = [r.steps for r in results]

        return {
            "mean_score": statistics.mean(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
            "mean_lines": statistics.mean(lines),
            "max_lines": max(lines),
            "mean_steps": statistics.mean(steps),
            "num_games": len(results)
        }
```

---

## Data Structures

### Experiment Hierarchy

```
Experiment
├── id: str
├── config: Config
├── start_time: datetime
├── end_time: datetime | None
├── status: ExperimentStatus
├── total_cost: float
├── root_lm_log: list[RootLMLogEntry]
│
└── generations: list[Generation]
    ├── id: int
    ├── start_time: datetime
    ├── end_time: datetime | None
    ├── advancement_reasoning: str | None
    │
    └── trials: list[Trial]
        ├── id: str
        ├── parent_id: str | None
        ├── prompt: str
        ├── response: str
        ├── code: str | None
        ├── input_tokens: int
        ├── output_tokens: int
        ├── evaluation: EvaluationResult | None
        └── selected_for_next_gen: bool
```

### Core Data Classes

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

class ExperimentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"

@dataclass
class Trial:
    id: str
    parent_id: Optional[str]
    prompt: str
    response: str
    code: Optional[str]
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    evaluation: Optional['EvaluationResult'] = None
    selected_for_next_gen: bool = False

@dataclass
class Generation:
    id: int
    trials: list[Trial] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    advancement_reasoning: Optional[str] = None

@dataclass
class Experiment:
    id: str
    config: 'Config'
    generations: list[Generation] = field(default_factory=list)
    root_lm_log: list['RootLMLogEntry'] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.RUNNING
    termination_reason: Optional[str] = None

@dataclass
class EvaluationResult:
    success: bool
    metrics: Optional[dict[str, float]] = None
    error: Optional[str] = None
    game_results: list['GameResult'] = field(default_factory=list)

@dataclass
class GameResult:
    seed: int
    score: float
    lines_cleared: int
    steps: int
    terminated: bool

@dataclass
class RootLMLogEntry:
    timestamp: datetime
    role: str  # "assistant" or "user" (execution results)
    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
```

---

## REPL API

The Root LLM has access to these functions in its REPL environment:

### `spawn_child_llm`

```python
def spawn_child_llm(prompt: str, parent_id: Optional[str] = None) -> dict:
    """
    Spawn a Child LLM to generate Tetris-playing code.

    Args:
        prompt: Instructions for the Child LLM about what code to generate.
                Should describe the desired strategy, improvements, or approach.
        parent_id: Optional ID of a parent trial to establish lineage.
                   Useful for tracking evolutionary history.

    Returns:
        dict with keys:
            - trial_id (str): Unique identifier for this trial
            - code (str | None): Generated Python code, or None if generation failed
            - metrics (dict | None): If code was generated, None until evaluate_program() called
            - reasoning (str): The Child LLM's reasoning/explanation
            - success (bool): Whether code was successfully generated
            - error (str | None): Error message if generation failed

    Example:
        >>> result = spawn_child_llm(
        ...     prompt="Write a Tetris agent that prioritizes clearing lines over building height",
        ...     parent_id="trial_001"
        ... )
        >>> if result['success']:
        ...     print(f"Generated trial {result['trial_id']}")
        ...     eval_result = evaluate_program(result['code'], num_games=5)

    Notes:
        - Each call consumes API tokens and counts toward budget
        - Code must define: select_action(observation, info) -> int
        - Maximum of {max_children_per_gen} children per generation
    """
    pass
```

### `evaluate_program`

```python
def evaluate_program(code: str, num_games: int = 10) -> dict:
    """
    Evaluate Tetris-playing code by running games.

    Args:
        code: Python code that defines a `select_action(observation, info)` function.
              The function receives:
                - observation: numpy array of the game board
                - info: dict with game state information
              And must return an integer action (0-6).
        num_games: Number of games to run for evaluation (default: 10)

    Returns:
        dict with keys:
            - success (bool): Whether evaluation completed without errors
            - metrics (dict | None): Aggregated metrics if successful:
                - mean_score: Average score across games
                - max_score: Best score achieved
                - min_score: Worst score achieved
                - std_score: Score standard deviation
                - mean_lines: Average lines cleared
                - max_lines: Most lines cleared in a game
                - mean_steps: Average game length in steps
            - error (str | None): Error message if evaluation failed
            - game_results (list): Per-game results

    Example:
        >>> code = '''
        ... def select_action(observation, info):
        ...     # Simple strategy: always move left
        ...     return 0  # move_left action
        ... '''
        >>> result = evaluate_program(code, num_games=5)
        >>> if result['success']:
        ...     print(f"Mean score: {result['metrics']['mean_score']}")

    Action Space:
        0: move_left
        1: move_right
        2: move_down
        3: rotate_counterclockwise
        4: rotate_clockwise
        5: hard_drop
        6: swap (hold piece)
    """
    pass
```

### `advance_generation`

```python
def advance_generation(selected_trial_ids: list[str], reasoning: str) -> int:
    """
    Advance to the next generation with selected trials.

    This marks the end of the current generation and begins a new one.
    Selected trials become the "parents" for the next generation.

    Args:
        selected_trial_ids: List of trial IDs to carry forward.
                           These trials' code/strategies inform the next generation.
        reasoning: Your reasoning for why these trials were selected.
                   This is logged for analysis and transparency.

    Returns:
        int: The new generation number

    Example:
        >>> # After spawning and evaluating several trials
        >>> trials = [t1, t2, t3, t4, t5]
        >>> # Select the best performers
        >>> selected = sorted(trials, key=lambda t: t['metrics']['mean_score'])[-2:]
        >>> new_gen = advance_generation(
        ...     selected_trial_ids=[t['trial_id'] for t in selected],
        ...     reasoning="Selected top 2 by mean score. Both use line-clearing priority."
        ... )
        >>> print(f"Now in generation {new_gen}")

    Notes:
        - You must call this to progress the experiment
        - Selected trials are marked in the experiment log
        - Previous generation data remains accessible
        - Maximum of {max_generations} generations allowed
    """
    pass
```

### `terminate_evolution`

```python
def terminate_evolution(reason: str) -> dict:
    """
    Terminate the evolutionary process.

    Call this when you've achieved satisfactory results or determined
    that further evolution is unlikely to yield improvements.

    Args:
        reason: Explanation of why you're terminating.
                Examples: "Achieved target score", "Converged - no improvement in 3 generations",
                "Budget constraints", "Found optimal strategy"

    Returns:
        dict with keys:
            - experiment_id (str): Unique experiment identifier
            - total_generations (int): Number of generations completed
            - total_trials (int): Total trials spawned
            - best_trial (dict): The best-performing trial with full details
            - total_cost (float): Total API cost in dollars
            - duration_seconds (float): Total experiment duration

    Example:
        >>> # After several generations with no improvement
        >>> summary = terminate_evolution(
        ...     reason="Score plateau reached. Best strategy: prioritize T-spins"
        ... )
        >>> print(f"Best score: {summary['best_trial']['metrics']['mean_score']}")

    Notes:
        - This ends the experiment immediately
        - All data is persisted before termination
        - Cannot be undone - experiment cannot be resumed after this
    """
    pass
```

### Additional Utility Functions (Optional)

The Root LLM can request these be added or write them itself:

```python
def get_generation_history() -> list[dict]:
    """Get summary of all generations so far."""
    pass

def get_trial(trial_id: str) -> dict:
    """Get full details of a specific trial."""
    pass

def get_current_budget() -> dict:
    """Get current budget status."""
    pass

def get_best_trials(n: int = 5) -> list[dict]:
    """Get top N trials by mean score across all generations."""
    pass
```

---

## Configuration

### YAML Configuration File

```yaml
# config.yaml - LLM Tetris Optimizer Configuration

# Experiment identification
experiment_name: "tetris_evolution_v1"
output_dir: "./results"

# Root LLM Configuration
root_lm:
  model: "claude-sonnet-4-20250514"
  cost:
    input: 0.003    # $ per 1K tokens
    output: 0.015   # $ per 1K tokens
  max_tokens: 4096  # Max response tokens

# Child LLM Configuration
child_lm:
  model: "claude-sonnet-4-20250514"
  cost:
    input: 0.003
    output: 0.015
  max_tokens: 2048

# Evolution Parameters
evolution:
  max_generations: 20
  max_children_per_generation: 10

# Budget
max_cost: 50.00  # Maximum $ to spend

# Evaluation
evaluation:
  num_games: 10
  max_steps_per_game: 10000

# Initial prompt for Root LLM (can also be in separate file)
initial_prompt: |
  You are an evolutionary algorithm controller tasked with evolving code that plays Tetris optimally.

  ## Your Goal
  Generate and evolve Python code that achieves the highest possible Tetris scores.

  ## Available Functions
  You have access to a Python REPL with these functions:
  - spawn_child_llm(prompt, parent_id=None) - Generate new code
  - evaluate_program(code, num_games=10) - Test code performance
  - advance_generation(selected_trial_ids, reasoning) - Move to next generation
  - terminate_evolution(reason) - End the experiment

  ## Code Requirements
  Generated code must define:
  ```python
  def select_action(observation, info) -> int:
      '''
      Args:
          observation: numpy array shape (H, W) representing the board
                      0 = empty, 1-7 = different piece types
          info: dict containing:
              - 'current_piece': int (1-7)
              - 'next_pieces': list[int]
              - 'held_piece': int or None
              - 'lines_cleared': int (total this game)
              - 'score': float (current score)

      Returns:
          int: Action from 0-6
              0: move_left
              1: move_right
              2: move_down
              3: rotate_counterclockwise
              4: rotate_clockwise
              5: hard_drop
              6: swap (hold)
      '''
  ```

  ## Strategy
  1. Start by spawning diverse initial strategies
  2. Evaluate each and analyze what works
  3. Select promising trials and evolve them
  4. Iterate until you achieve good performance or hit limits

  Begin your evolutionary process now.
```

### Config Data Class

```python
@dataclass
class LLMCost:
    input: float   # $ per 1K tokens
    output: float  # $ per 1K tokens

@dataclass
class LLMConfig:
    model: str
    cost: LLMCost
    max_tokens: int = 4096

@dataclass
class EvolutionConfig:
    max_generations: int
    max_children_per_generation: int

@dataclass
class EvaluationConfig:
    num_games: int = 10
    max_steps_per_game: int = 10000

@dataclass
class Config:
    experiment_name: str
    output_dir: str
    root_lm: LLMConfig
    child_lm: LLMConfig
    evolution: EvolutionConfig
    max_cost: float
    evaluation: EvaluationConfig
    initial_prompt: str

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load and validate config from YAML file."""
        pass
```

---

## Observability & Logging

### Log Structure

```
results/
└── experiment_20240115_143022/
    ├── config.yaml                    # Copy of configuration
    ├── experiment.json                # Full experiment data
    ├── summary.json                   # Quick summary stats
    │
    ├── root_lm/
    │   ├── conversation.jsonl         # Full conversation log
    │   └── token_usage.jsonl          # Token usage per turn
    │
    ├── generations/
    │   ├── gen_000/
    │   │   ├── summary.json           # Generation summary
    │   │   └── trials/
    │   │       ├── trial_abc123.json  # Full trial data
    │   │       ├── trial_def456.json
    │   │       └── ...
    │   ├── gen_001/
    │   │   └── ...
    │   └── ...
    │
    └── report.md                      # Human-readable report
```

### Experiment JSON Schema

```json
{
  "id": "exp_20240115_143022",
  "config": { "...": "..." },
  "status": "completed",
  "start_time": "2024-01-15T14:30:22Z",
  "end_time": "2024-01-15T15:45:33Z",
  "termination_reason": "Achieved target score of 10000",

  "summary": {
    "total_generations": 8,
    "total_trials": 45,
    "total_cost": 12.34,
    "total_tokens": {
      "input": 234567,
      "output": 45678
    },
    "best_score": 12543,
    "best_trial_id": "trial_xyz789"
  },

  "generations": [
    {
      "id": 0,
      "start_time": "2024-01-15T14:30:25Z",
      "end_time": "2024-01-15T14:35:12Z",
      "trial_count": 5,
      "selected_trials": ["trial_abc123", "trial_def456"],
      "advancement_reasoning": "Selected top 2 performers...",
      "best_metrics": {
        "mean_score": 1234,
        "max_score": 2000
      }
    }
  ]
}
```

### Trial JSON Schema

```json
{
  "id": "trial_abc123",
  "generation": 0,
  "parent_id": null,
  "timestamp": "2024-01-15T14:31:05Z",

  "prompt": "Write a Tetris agent that...",
  "response": "Here's my implementation...",
  "code": "def select_action(observation, info):\n    ...",

  "tokens": {
    "input": 1234,
    "output": 567
  },

  "evaluation": {
    "success": true,
    "metrics": {
      "mean_score": 1500,
      "max_score": 2500,
      "min_score": 800,
      "std_score": 450,
      "mean_lines": 45,
      "max_lines": 78,
      "mean_steps": 3456
    },
    "game_results": [
      {"seed": 0, "score": 1200, "lines_cleared": 34, "steps": 2890},
      {"seed": 1, "score": 2500, "lines_cleared": 78, "steps": 4500}
    ]
  },

  "selected_for_next_gen": true
}
```

### Root LLM Conversation Log

```jsonl
{"turn": 0, "role": "system", "content": "You are...", "timestamp": "..."}
{"turn": 1, "role": "user", "content": "Initial prompt...", "timestamp": "..."}
{"turn": 2, "role": "assistant", "content": "```python\nresult = spawn_child_llm(...)\n```", "timestamp": "...", "input_tokens": 1234, "output_tokens": 567}
{"turn": 3, "role": "user", "content": "Execution result: {...}", "timestamp": "..."}
```

### Logger Implementation

```python
class ExperimentLogger:
    def __init__(self, output_dir: str, experiment_id: str):
        self.base_dir = Path(output_dir) / experiment_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._init_directories()

    def log_root_turn(self, turn: int, role: str, content: str,
                      input_tokens: int = None, output_tokens: int = None) -> None:
        """Log a turn in the Root LLM conversation."""
        entry = {
            "turn": turn,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        self._append_jsonl(self.base_dir / "root_lm" / "conversation.jsonl", entry)

    def log_trial(self, generation: int, trial: Trial) -> None:
        """Log a trial to its generation directory."""
        gen_dir = self.base_dir / "generations" / f"gen_{generation:03d}" / "trials"
        gen_dir.mkdir(parents=True, exist_ok=True)

        trial_file = gen_dir / f"{trial.id}.json"
        with open(trial_file, 'w') as f:
            json.dump(asdict(trial), f, indent=2, default=str)

    def save_experiment(self, experiment: Experiment) -> None:
        """Save full experiment state."""
        with open(self.base_dir / "experiment.json", 'w') as f:
            json.dump(asdict(experiment), f, indent=2, default=str)

    def generate_report(self, experiment: Experiment) -> str:
        """Generate human-readable Markdown report."""
        pass
```

---

## Tetris Integration

### Tetris-Gymnasium Setup

```python
# Installation
# pip install tetris-gymnasium

import gymnasium as gym
from tetris_gymnasium.envs import Tetris

# Create environment
env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")

# Standard Gymnasium API
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
```

### Observation Space

The observation is a numpy array representing the game board:
- Shape: `(height, width)` typically `(20, 10)`
- Values: `0` = empty, `1-7` = different piece types

### Action Space

```python
ACTIONS = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_counterclockwise",
    4: "rotate_clockwise",
    5: "hard_drop",
    6: "swap"  # Hold piece
}
```

### Info Dictionary

```python
info = {
    'current_piece': int,      # Current piece type (1-7)
    'next_pieces': list[int],  # Queue of upcoming pieces
    'held_piece': int | None,  # Currently held piece
    'lines_cleared': int,      # Lines cleared this game
    'score': float             # Current score
}
```

### Agent Interface

Code generated by Child LLMs must implement:

```python
def select_action(observation: np.ndarray, info: dict) -> int:
    """
    Select an action for the current Tetris game state.

    Args:
        observation: 2D numpy array of the board
        info: Dictionary with game state information

    Returns:
        Action integer 0-6
    """
    # Example: Random agent
    return np.random.randint(0, 7)
```

### Evaluation Runner

```python
def run_evaluation(agent_code: str, num_games: int = 10) -> dict:
    """
    Run multiple Tetris games with the given agent code.

    Returns metrics dictionary.
    """
    # Load agent
    namespace = {}
    exec(agent_code, namespace)
    select_action = namespace['select_action']

    results = []
    for seed in range(num_games):
        env = gym.make("tetris_gymnasium/Tetris")
        obs, info = env.reset(seed=seed)

        total_reward = 0
        while True:
            action = select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        results.append({
            'seed': seed,
            'score': total_reward,
            'lines_cleared': info['lines_cleared']
        })
        env.close()

    return aggregate_results(results)
```

---

## Cost Management

### Cost Calculation

```python
def calculate_cost(input_tokens: int, output_tokens: int, config: LLMConfig) -> float:
    """Calculate cost in dollars."""
    return (
        (input_tokens / 1000) * config.cost.input +
        (output_tokens / 1000) * config.cost.output
    )
```

### Budget Enforcement

The `CostTracker` enforces the budget by:

1. **Pre-check**: Before each LLM call, verify budget isn't exhausted
2. **Record**: After each call, record the actual token usage
3. **Alert**: When budget reaches 80%, log a warning
4. **Terminate**: When budget exceeded, raise `BudgetExhaustedError`

```python
class BudgetExhaustedError(Exception):
    """Raised when the cost budget is exhausted."""
    pass

class CostTracker:
    def check_budget(self) -> None:
        """Raise if budget exhausted."""
        if self.total_cost >= self.max_cost:
            raise BudgetExhaustedError(
                f"Budget exhausted: ${self.total_cost:.4f} >= ${self.max_cost:.4f}"
            )

        # Warning at 80%
        if self.total_cost >= self.max_cost * 0.8:
            logger.warning(f"Budget at {self.total_cost/self.max_cost*100:.1f}%")
```

### Cost Visibility in REPL

The Root LLM can check budget status:

```python
def get_current_budget() -> dict:
    """
    Get current budget status.

    Returns:
        dict with keys:
            - total_spent: float (dollars spent so far)
            - max_budget: float (total budget)
            - remaining: float (dollars remaining)
            - percentage_used: float (0-100)
            - breakdown: dict with per-source costs
    """
    pass
```

---

## Implementation Guide

### Phase 1: Foundation (Components 1-3)

**Goal**: Basic infrastructure that can load config and track costs.

1. **Config loader** - Parse YAML, validate, create Config dataclass
2. **Cost tracker** - Track tokens, calculate costs, enforce budget
3. **Basic data structures** - Trial, Generation, Experiment dataclasses

### Phase 2: LLM Integration (Components 4-5)

**Goal**: Ability to call LLMs and parse responses.

4. **LLM client abstraction** - Unified interface for different providers
5. **Code extraction** - Parse code blocks from LLM responses

### Phase 3: REPL Environment (Components 6-7)

**Goal**: Functional REPL that can execute code safely.

6. **Sandbox execution** - Safe Python code execution
7. **REPL environment** - Variable management, function injection

### Phase 4: Evolution Core (Components 8-9)

**Goal**: Complete evolutionary loop.

8. **Child spawner** - Generate code via Child LLMs
9. **Tetris evaluator** - Evaluate code in Tetris games

### Phase 5: Orchestration (Components 10-11)

**Goal**: End-to-end experiment execution.

10. **Root LLM controller** - Manage Root LLM REPL loop
11. **Orchestrator** - Coordinate all components

### Phase 6: Observability (Components 12-13)

**Goal**: Complete logging and reporting.

12. **Experiment store** - Persist all data
13. **Report generator** - Human-readable summaries

### Phase 7: CLI & Polish (Components 14-15)

**Goal**: User-friendly interface.

14. **CLI** - Command-line interface
15. **Documentation** - Usage docs, examples

---

## Testing Strategy

### Unit Tests

Each component should have isolated unit tests:

```python
# test_cost_tracker.py
def test_cost_calculation():
    tracker = CostTracker(max_cost=10.0)
    tracker.record(TokenUsage(input=1000, output=500),
                   source="root", cost_per_input=0.003, cost_per_output=0.015)
    assert tracker.total_cost == pytest.approx(0.003 + 0.0075)

def test_budget_enforcement():
    tracker = CostTracker(max_cost=0.01)
    tracker.record(TokenUsage(input=10000, output=0),
                   source="root", cost_per_input=0.003, cost_per_output=0.015)
    with pytest.raises(BudgetExhaustedError):
        tracker.check_budget()
```

### Integration Tests

Test component interactions:

```python
# test_child_spawning.py
def test_spawn_and_evaluate():
    """Test spawning a child and evaluating its output."""
    config = load_test_config()
    spawner = ChildLLMSpawner(config, MockCostTracker(), MockStore())

    result = spawner.spawn("Write a simple Tetris agent")
    assert result['success']
    assert result['code'] is not None

    evaluator = TetrisEvaluator(num_games=2)
    eval_result = evaluator.evaluate(result['code'])
    assert eval_result['success']
```

### End-to-End Tests

Full experiment runs with mocked LLMs:

```python
# test_full_experiment.py
def test_complete_experiment():
    """Run a complete experiment with mock LLMs."""
    config = Config(
        max_generations=2,
        max_children_per_generation=3,
        max_cost=1.0,
        # ... other config
    )

    with mock_llm_responses():
        result = run_experiment(config)

    assert result.status == ExperimentStatus.COMPLETED
    assert result.total_generations == 2
    assert len(result.all_trials()) <= 6
```

---

## Detailed TODO List

Below is a comprehensive task list with dependencies. Each task is marked with:
- `[P#]` = Phase number
- Dependencies listed below each task

---

### Component 1: Configuration System
**Location**: `src/tetris_evolve/config.py`

- [ ] **1.1** Define `LLMCost` dataclass
  - Dependencies: None

- [ ] **1.2** Define `LLMConfig` dataclass
  - Dependencies: 1.1

- [ ] **1.3** Define `EvolutionConfig` dataclass
  - Dependencies: None

- [ ] **1.4** Define `EvaluationConfig` dataclass
  - Dependencies: None

- [ ] **1.5** Define `Config` dataclass with all nested configs
  - Dependencies: 1.2, 1.3, 1.4

- [ ] **1.6** Implement `Config.from_yaml()` class method
  - Dependencies: 1.5

- [ ] **1.7** Implement config validation (required fields, value ranges)
  - Dependencies: 1.6

- [ ] **1.8** Write unit tests for config loading and validation
  - Dependencies: 1.7

---

### Component 2: Cost Tracker
**Location**: `src/tetris_evolve/llm/cost_tracker.py`

- [ ] **2.1** Define `TokenUsage` dataclass
  - Dependencies: None

- [ ] **2.2** Define `CostRecord` dataclass
  - Dependencies: 2.1

- [ ] **2.3** Define `BudgetExhaustedError` exception
  - Dependencies: None

- [ ] **2.4** Implement `CostTracker.__init__()` with max_cost parameter
  - Dependencies: 2.2, 2.3

- [ ] **2.5** Implement `CostTracker.record()` method
  - Dependencies: 2.4

- [ ] **2.6** Implement `CostTracker.check_budget()` method
  - Dependencies: 2.4

- [ ] **2.7** Implement `CostTracker.get_summary()` method
  - Dependencies: 2.4

- [ ] **2.8** Add thread safety with locking
  - Dependencies: 2.5

- [ ] **2.9** Write unit tests for cost tracking and budget enforcement
  - Dependencies: 2.6, 2.7

---

### Component 3: Core Data Structures
**Location**: `src/tetris_evolve/evolution/`

- [ ] **3.1** Define `ExperimentStatus` enum
  - Dependencies: None

- [ ] **3.2** Define `GameResult` dataclass
  - Dependencies: None

- [ ] **3.3** Define `EvaluationResult` dataclass
  - Dependencies: 3.2

- [ ] **3.4** Define `Trial` dataclass
  - Dependencies: 3.3

- [ ] **3.5** Define `Generation` dataclass
  - Dependencies: 3.4

- [ ] **3.6** Define `Experiment` dataclass
  - Dependencies: 3.1, 3.5

- [ ] **3.7** Define `RootLMLogEntry` dataclass
  - Dependencies: None

- [ ] **3.8** Write unit tests for data structure serialization
  - Dependencies: 3.6, 3.7

---

### Component 4: LLM Client Abstraction
**Location**: `src/tetris_evolve/llm/client.py`

- [ ] **4.1** Define `LLMResponse` dataclass
  - Dependencies: None

- [ ] **4.2** Define `LLMClient` abstract base class
  - Dependencies: 4.1, 2.1

- [ ] **4.3** Implement `AnthropicClient` for Claude models
  - Dependencies: 4.2

- [ ] **4.4** Implement response parsing for token usage
  - Dependencies: 4.3

- [ ] **4.5** Add retry logic with exponential backoff
  - Dependencies: 4.3

- [ ] **4.6** Write unit tests with mocked API responses
  - Dependencies: 4.5

---

### Component 5: Code Extraction
**Location**: `src/tetris_evolve/llm/code_extraction.py`

- [ ] **5.1** Implement `extract_code_blocks()` function
  - Dependencies: None

- [ ] **5.2** Handle multiple code blocks in a single response
  - Dependencies: 5.1

- [ ] **5.3** Handle code blocks with/without language specifiers
  - Dependencies: 5.1

- [ ] **5.4** Implement `extract_python_function()` for select_action
  - Dependencies: 5.1

- [ ] **5.5** Write unit tests for various code block formats
  - Dependencies: 5.4

---

### Component 6: Sandbox Execution
**Location**: `src/tetris_evolve/repl/sandbox.py`

- [ ] **6.1** Define safe builtins whitelist
  - Dependencies: None

- [ ] **6.2** Implement `Sandbox` class with restricted globals
  - Dependencies: 6.1

- [ ] **6.3** Implement timeout for code execution
  - Dependencies: 6.2

- [ ] **6.4** Implement output capture (stdout/stderr)
  - Dependencies: 6.2

- [ ] **6.5** Implement output truncation for long outputs
  - Dependencies: 6.4

- [ ] **6.6** Add memory limit protection (optional)
  - Dependencies: 6.2

- [ ] **6.7** Write unit tests for sandbox safety
  - Dependencies: 6.6

---

### Component 7: REPL Environment
**Location**: `src/tetris_evolve/repl/environment.py`

- [ ] **7.1** Define `ExecutionResult` dataclass
  - Dependencies: None

- [ ] **7.2** Implement `REPLEnvironment.__init__()` with function injection
  - Dependencies: 6.2, 7.1

- [ ] **7.3** Implement `REPLEnvironment.execute()` method
  - Dependencies: 7.2

- [ ] **7.4** Implement variable persistence between executions
  - Dependencies: 7.3

- [ ] **7.5** Implement `REPLEnvironment.get_function_docs()` method
  - Dependencies: 7.2

- [ ] **7.6** Write unit tests for REPL execution and state
  - Dependencies: 7.5

---

### Component 8: Child LLM Spawner
**Location**: `src/tetris_evolve/llm/child_spawner.py`

- [ ] **8.1** Define `TrialResult` dataclass for spawn return value
  - Dependencies: None

- [ ] **8.2** Implement `ChildLLMSpawner.__init__()`
  - Dependencies: 4.3, 2.4

- [ ] **8.3** Implement `ChildLLMSpawner.spawn()` method
  - Dependencies: 8.2, 5.4

- [ ] **8.4** Implement trial ID generation
  - Dependencies: 8.3

- [ ] **8.5** Integrate cost tracking for each spawn
  - Dependencies: 8.3, 2.5

- [ ] **8.6** Write unit tests with mocked LLM
  - Dependencies: 8.5

---

### Component 9: Tetris Evaluator
**Location**: `src/tetris_evolve/evaluation/tetris_evaluator.py`

- [ ] **9.1** Implement `TetrisEvaluator.__init__()`
  - Dependencies: None

- [ ] **9.2** Implement `TetrisEvaluator._load_agent()` method
  - Dependencies: 9.1

- [ ] **9.3** Implement `TetrisEvaluator._run_game()` method
  - Dependencies: 9.2

- [ ] **9.4** Implement `TetrisEvaluator._aggregate_metrics()` method
  - Dependencies: 9.3

- [ ] **9.5** Implement `TetrisEvaluator.evaluate()` method
  - Dependencies: 9.4

- [ ] **9.6** Add error handling for invalid agent code
  - Dependencies: 9.5

- [ ] **9.7** Add timeout for runaway agents
  - Dependencies: 9.5

- [ ] **9.8** Write unit tests with simple test agents
  - Dependencies: 9.7

---

### Component 10: REPL Functions
**Location**: `src/tetris_evolve/repl/functions.py`

- [ ] **10.1** Implement `create_spawn_child_llm()` factory
  - Dependencies: 8.3

- [ ] **10.2** Implement `create_evaluate_program()` factory
  - Dependencies: 9.5

- [ ] **10.3** Implement `create_advance_generation()` factory
  - Dependencies: 3.5

- [ ] **10.4** Implement `create_terminate_evolution()` factory
  - Dependencies: 3.6

- [ ] **10.5** Implement `create_get_budget()` factory
  - Dependencies: 2.7

- [ ] **10.6** Implement `create_repl_functions()` to create all functions
  - Dependencies: 10.1, 10.2, 10.3, 10.4, 10.5

- [ ] **10.7** Write unit tests for each REPL function
  - Dependencies: 10.6

---

### Component 11: Root LLM Controller
**Location**: `src/tetris_evolve/llm/root_controller.py`

- [ ] **11.1** Implement `RootLLMController.__init__()`
  - Dependencies: 4.3, 2.4

- [ ] **11.2** Implement `_build_system_message()` method
  - Dependencies: 11.1, 7.5

- [ ] **11.3** Implement `_extract_code_blocks()` method
  - Dependencies: 11.1, 5.1

- [ ] **11.4** Implement `_format_execution_results()` method
  - Dependencies: 11.1

- [ ] **11.5** Implement `RootLLMController.start()` REPL loop
  - Dependencies: 11.2, 11.3, 11.4, 7.3

- [ ] **11.6** Handle termination signal from REPL
  - Dependencies: 11.5

- [ ] **11.7** Handle budget exhaustion
  - Dependencies: 11.5, 2.6

- [ ] **11.8** Write integration tests with mocked LLM
  - Dependencies: 11.7

---

### Component 12: Experiment Store
**Location**: `src/tetris_evolve/storage/experiment_store.py`

- [ ] **12.1** Implement `ExperimentStore.__init__()` with directory setup
  - Dependencies: None

- [ ] **12.2** Implement `create_experiment()` method
  - Dependencies: 12.1, 3.6

- [ ] **12.3** Implement `record_trial()` method
  - Dependencies: 12.1, 3.4

- [ ] **12.4** Implement `record_generation()` method
  - Dependencies: 12.1, 3.5

- [ ] **12.5** Implement `save_experiment()` method
  - Dependencies: 12.1

- [ ] **12.6** Implement `load_experiment()` for resume capability
  - Dependencies: 12.5

- [ ] **12.7** Implement `finalize()` method
  - Dependencies: 12.5

- [ ] **12.8** Write unit tests for storage operations
  - Dependencies: 12.7

---

### Component 13: Logging System
**Location**: `src/tetris_evolve/logging/`

- [ ] **13.1** Implement `ExperimentLogger.__init__()` with directory setup
  - Dependencies: None

- [ ] **13.2** Implement `log_root_turn()` method
  - Dependencies: 13.1, 3.7

- [ ] **13.3** Implement `log_trial()` method
  - Dependencies: 13.1, 3.4

- [ ] **13.4** Implement `generate_report()` Markdown report
  - Dependencies: 13.1, 3.6

- [ ] **13.5** Implement cost summary in report
  - Dependencies: 13.4, 2.7

- [ ] **13.6** Write unit tests for logging
  - Dependencies: 13.5

---

### Component 14: Orchestrator
**Location**: `src/tetris_evolve/orchestrator.py`

- [ ] **14.1** Implement `Orchestrator.__init__()`
  - Dependencies: 1.5, 2.4, 11.1, 7.2, 12.1, 13.1

- [ ] **14.2** Implement `_create_repl_functions()` method
  - Dependencies: 14.1, 10.6

- [ ] **14.3** Implement `Orchestrator.run()` method
  - Dependencies: 14.2, 11.5

- [ ] **14.4** Implement graceful shutdown on interrupt
  - Dependencies: 14.3

- [ ] **14.5** Implement experiment finalization
  - Dependencies: 14.3, 12.7

- [ ] **14.6** Write integration tests for full flow
  - Dependencies: 14.5

---

### Component 15: CLI
**Location**: `src/tetris_evolve/cli.py`

- [ ] **15.1** Implement `run_experiment()` CLI command
  - Dependencies: 14.3

- [ ] **15.2** Implement `resume_experiment()` CLI command
  - Dependencies: 14.3, 12.6

- [ ] **15.3** Implement `generate_report()` CLI command
  - Dependencies: 13.4

- [ ] **15.4** Add argument parsing with argparse/click
  - Dependencies: 15.1, 15.2, 15.3

- [ ] **15.5** Add progress display during execution
  - Dependencies: 15.4

- [ ] **15.6** Write CLI tests
  - Dependencies: 15.5

---

### Component 16: Documentation & Examples

- [ ] **16.1** Write README.md with setup instructions
  - Dependencies: 15.4

- [ ] **16.2** Create example config files
  - Dependencies: 1.7

- [ ] **16.3** Create example prompts for different strategies
  - Dependencies: 14.3

- [ ] **16.4** Write API documentation
  - Dependencies: All components

---

### Component 17: End-to-End Testing

- [ ] **17.1** Create mock LLM responses for testing
  - Dependencies: 4.3

- [ ] **17.2** Write E2E test for 1-generation experiment
  - Dependencies: 14.6, 17.1

- [ ] **17.3** Write E2E test for multi-generation experiment
  - Dependencies: 17.2

- [ ] **17.4** Write E2E test for budget exhaustion
  - Dependencies: 17.2

- [ ] **17.5** Write E2E test for experiment resume
  - Dependencies: 17.2, 12.6

---

## Dependency Graph

```
Phase 1: Foundation
  1.1 → 1.2 → 1.5 → 1.6 → 1.7 → 1.8
  1.3 ─────────┘
  1.4 ─────────┘
  2.1 → 2.2 → 2.4 → 2.5 → 2.8 → 2.9
              └──→ 2.6 ─────────┘
              └──→ 2.7 ─────────┘
  3.1 ─────────────────→ 3.6 → 3.8
  3.2 → 3.3 → 3.4 → 3.5 ──┘
  3.7 ──────────────────────────┘

Phase 2: LLM Integration
  4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
  5.1 → 5.2
  └──→ 5.3
  └──→ 5.4 → 5.5

Phase 3: REPL Environment
  6.1 → 6.2 → 6.3
         └──→ 6.4 → 6.5
         └──→ 6.6 → 6.7
  7.1 ──┐
  6.2 ──┴→ 7.2 → 7.3 → 7.4
              └──→ 7.5 → 7.6

Phase 4: Evolution Core
  8.1 ──┐
  4.3 ──┤
  2.4 ──┴→ 8.2 → 8.3 → 8.4
                  └──→ 8.5 → 8.6
  9.1 → 9.2 → 9.3 → 9.4 → 9.5 → 9.6
                              └──→ 9.7 → 9.8

Phase 5: Orchestration
  8.3 → 10.1 ──┐
  9.5 → 10.2 ──┤
  3.5 → 10.3 ──┼→ 10.6 → 10.7
  3.6 → 10.4 ──┤
  2.7 → 10.5 ──┘

  4.3 ──┐
  2.4 ──┴→ 11.1 → 11.2 → 11.5 → 11.6
                  11.3 ──┘     └──→ 11.7 → 11.8
                  11.4 ──┘

Phase 6: Storage & Logging
  12.1 → 12.2 → 12.5 → 12.6 → 12.8
         12.3 ──┘     12.7 ──┘
         12.4 ──┘

  13.1 → 13.2 → 13.4 → 13.5 → 13.6
         13.3 ──┘

Phase 7: Final Integration
  1.5, 2.4, 7.2, 10.6, 11.1, 12.1, 13.1 → 14.1 → 14.2 → 14.3 → 14.4 → 14.5 → 14.6

  14.3 → 15.1 ──┐
  12.6 → 15.2 ──┼→ 15.4 → 15.5 → 15.6
  13.4 → 15.3 ──┘
```

---

## Appendix: Sample Initial Prompt

```markdown
You are an evolutionary algorithm controller tasked with evolving code that plays Tetris optimally.

## Your Goal
Generate and evolve Python code that achieves the highest possible Tetris scores. You control the evolutionary process by spawning child LLMs to generate code, evaluating that code, and deciding which strategies to pursue further.

## Available Functions

You have access to a Python REPL environment with these functions:

### spawn_child_llm(prompt: str, parent_id: str | None = None) -> dict
Spawn a Child LLM to generate Tetris-playing code.
- Returns: {trial_id, code, reasoning, success, error}

### evaluate_program(code: str, num_games: int = 10) -> dict
Evaluate Tetris code by running games.
- Returns: {success, metrics, error}
- Metrics include: mean_score, max_score, min_score, std_score, mean_lines

### advance_generation(selected_trial_ids: list[str], reasoning: str) -> int
Move to next generation with selected trials.
- Returns: new generation number

### terminate_evolution(reason: str) -> dict
End the experiment.
- Returns: final summary with best results

## Code Requirements

Generated code MUST define this function:
```python
def select_action(observation, info) -> int:
    '''
    Args:
        observation: numpy array shape (20, 10) - the board
                    0 = empty, 1-7 = piece types
        info: dict with:
            - 'current_piece': int (1-7)
            - 'next_pieces': list[int]
            - 'held_piece': int | None
            - 'lines_cleared': int
            - 'score': float

    Returns:
        int: Action 0-6
            0=left, 1=right, 2=down,
            3=rotate_ccw, 4=rotate_cw,
            5=hard_drop, 6=swap
    '''
```

## Budget
You have a limited budget. Use get_current_budget() to check status.
Be efficient - don't spawn unnecessary trials.

## Strategy Suggestions
1. Start with diverse strategies: random, greedy, heuristic
2. Evaluate and compare performance
3. Select best performers and iterate on them
4. Consider: line clearing, hole avoidance, height management
5. Terminate when you've found good strategies or exhausted useful variations

## Begin
Start by spawning 3-5 diverse initial strategies, then evaluate and iterate.
```

---

## References

- [AlphaEvolve Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
- [RLM Blog Post](https://alexzhang13.github.io/blog/2025/rlm/)
- [RLM GitHub](https://github.com/alexzhang13/rlm)
- [Tetris Gymnasium](https://max-we.github.io/Tetris-Gymnasium/)
- [PufferLib Documentation](https://puffer.ai/docs.html)
- [OpenEvolve Implementation](https://huggingface.co/blog/codelion/openevolve)
