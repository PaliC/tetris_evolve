# LLM-Evolve Design Document

## Overview

This document describes the architecture for an evolutionary code generation system that uses LLMs to iteratively develop optimization algorithms. The system combines ideas from:

1. **AlphaEvolve** (DeepMind): Evolutionary algorithm for program generation using LLMs
2. **Recursive LLMs (RLM)**: Hierarchical LLM spawning for complex decision-making

**Current Target Problem**: Circle Packing - Pack n circles into a unit square to maximize the sum of their radii.

- **Benchmark**: AlphaEvolve achieved 2.635 for n=26
- **Evaluation**: Deterministic, fast (~60ms), geometric validation

The goal is to evolve increasingly better algorithms by having a "Root LLM" orchestrate the evolution process, spawning "Child LLMs" to generate candidate programs, evaluating them, and selecting the best to inform future generations.

---

## Core Concepts

### From AlphaEvolve

AlphaEvolve uses an evolutionary approach to program generation:

1. **Programs Database**: Maintains a population of candidate programs with their evaluation scores
2. **Prompt Sampler**: Assembles prompts that incorporate high-performing programs from history
3. **Evaluation Loop**: Automatically verifies, runs, and scores proposed programs
4. **Evolutionary Selection**: Determines which programs merit inclusion in future prompts
5. **Mutation**: Programs evolve through LLM-guided mutations over generations

Key insight: The LLM acts as both the mutation operator AND can reason about what makes programs better.

### From Recursive LLMs

RLMs provide a framework for hierarchical LLM computation:

1. **REPL Environment**: A Python execution environment where LLMs can write and execute code
2. **Recursive Spawning**: Parent LLMs can spawn child LLMs with transformed context
3. **Context Passing**: Children receive query + context subset, return results to parent
4. **Autonomous Decomposition**: The LLM decides how to break down problems

Key insight: The Root LLM can programmatically control the evolution process, writing its own helper functions and making strategic decisions.

### Our Synthesis

We combine these ideas:

- **Root LLM** = The "brain" that orchestrates evolution (RLM-style REPL)
- **Child LLMs** = Program generators that propose solutions (AlphaEvolve-style)
- **Evaluation** = Automated scoring via problem-specific evaluator
- **Selection** = Root LLM decides which trials to advance (not just fitness sorting)

The key innovation: The Root LLM has agency over the evolutionary process itself. It can:
- Decide mutation strategies
- Choose how many children to spawn and with what prompts
- Analyze patterns in successful vs unsuccessful programs
- Write custom analysis functions in the REPL

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXPERIMENT                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         ROOT LLM (REPL)                                 │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Available Functions:                                             │  │ │
│  │  │  - spawn_child_llm(prompt, parent_id) -> trial_result             │  │ │
│  │  │  - evaluate_program(code) -> metrics                              │  │ │
│  │  │  - advance_generation(trial_ids, reasoning) -> gen_num            │  │ │
│  │  │  - terminate_evolution(reason) -> final_result                    │  │ │
│  │  │  - get_best_trials(n) -> list[trial]                              │  │ │
│  │  │  - get_cost_remaining() -> float                                  │  │ │
│  │  │  - [custom functions the Root LLM can define]                     │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                    │ │
│  │  │  GENERATION 0       │    │  GENERATION 1       │    ...             │ │
│  │  │  ┌───────────────┐  │    │  ┌───────────────┐  │                    │ │
│  │  │  │ Trial 0.1     │  │    │  │ Trial 1.1     │  │                    │ │
│  │  │  │ - code        │  │    │  │ - code        │  │                    │ │
│  │  │  │ - metrics     │  │    │  │ - metrics     │  │                    │ │
│  │  │  │ - reasoning   │  │    │  │ - reasoning   │  │                    │ │
│  │  │  └───────────────┘  │    │  └───────────────┘  │                    │ │
│  │  │  ┌───────────────┐  │    │  ┌───────────────┐  │                    │ │
│  │  │  │ Trial 0.2     │  │    │  │ Trial 1.2     │  │                    │ │
│  │  │  └───────────────┘  │    │  └───────────────┘  │                    │ │
│  │  │         ...         │    │         ...         │                    │ │
│  │  └─────────────────────┘    └─────────────────────┘                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         LOGGING SYSTEM                                  │ │
│  │  experiment.json                                                        │ │
│  │  ├── generations/                                                       │ │
│  │  │   ├── gen_0/                                                         │ │
│  │  │   │   ├── trial_0_1.json                                             │ │
│  │  │   │   └── trial_0_2.json                                             │ │
│  │  │   └── gen_1/                                                         │ │
│  │  ├── root_llm_log.jsonl                                                 │ │
│  │  └── cost_tracking.json                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Circle Packing Problem

### Problem Definition

Pack 26 circles into a unit square [0,1] x [0,1] to maximize the sum of their radii.

**Constraints**:
- All circles must be entirely inside the unit square
- No two circles may overlap
- All radii must be non-negative

**Benchmark**: AlphaEvolve achieved sum = 2.635

### Code Specification

Child LLMs must generate code with this function:

```python
import numpy as np

def construct_packing():
    """
    Construct a circle packing for n=26 circles in a unit square.

    Returns:
        centers: np.array of shape (26, 2) - (x, y) coordinates
        radii: np.array of shape (26,) - radius of each circle
        sum_radii: float - sum of all radii
    """
    # Implementation here
    pass

def run_packing():
    """Entry point called by evaluator."""
    return construct_packing()
```

### Evaluation Metrics

```python
{
    "valid": bool,           # True if packing satisfies all constraints
    "sum_radii": float,      # Sum of all radii (0 if invalid)
    "target_ratio": float,   # sum_radii / 2.635 (0 if invalid)
    "combined_score": float, # target_ratio if valid, else 0
    "eval_time": float,      # Seconds to evaluate
    "error": Optional[str],  # Error message if any
}
```

### Validation Logic

```python
def validate_packing(centers, radii, n_circles=26):
    tolerance = 1e-6

    # Check shapes
    if centers.shape != (n_circles, 2) or radii.shape != (n_circles,):
        return False, "Invalid shapes"

    # Check for NaN
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values"

    # Check radii non-negative
    if (radii < 0).any():
        return False, "Negative radii"

    # Check within bounds
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tolerance or x + r > 1 + tolerance:
            return False, f"Circle {i} outside x-bounds"
        if y - r < -tolerance or y + r > 1 + tolerance:
            return False, f"Circle {i} outside y-bounds"

    # Check no overlaps
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - tolerance:
                return False, f"Circles {i} and {j} overlap"

    return True, None
```

---

## Component Specifications

### 1. Configuration System (`config.py`)

```yaml
# experiment_config.yaml
experiment:
  name: "circle_packing_001"
  output_dir: "./experiments"
  seed: 42

root_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_input_token: 0.000003
  cost_per_output_token: 0.000015
  max_iterations: 30

child_llm:
  model: "claude-sonnet-4-20250514"
  cost_per_input_token: 0.000003
  cost_per_output_token: 0.000015

# Evaluation config points to an evaluator function/class
# This makes the system pluggable for different evaluation problems
evaluation:
  evaluator_fn: "tetris_evolve.evaluation.circle_packing:CirclePackingEvaluator"
  evaluator_kwargs:
    n_circles: 26
    target: 2.635
    timeout_seconds: 30

evolution:
  max_generations: 10
  max_children_per_generation: 10

budget:
  max_total_cost: 20.0
```

**Data Classes:**

```python
@dataclass
class Config:
    experiment: ExperimentConfig
    root_llm: LLMConfig
    child_llm: LLMConfig
    evaluation: EvaluationConfig  # Required - points to evaluator
    evolution: EvolutionConfig
    budget: BudgetConfig

@dataclass
class EvaluationConfig:
    evaluator_fn: str  # Module path like "module.path:ClassName"
    evaluator_kwargs: Dict[str, Any] = field(default_factory=dict)

# Load evaluator dynamically at runtime:
def load_evaluator(config: EvaluationConfig) -> Any:
    """Load evaluator from module path."""
    module_path, obj_name = config.evaluator_fn.rsplit(":", 1)
    module = importlib.import_module(module_path)
    evaluator_class = getattr(module, obj_name)
    return evaluator_class(**config.evaluator_kwargs)
```

### 2. Cost Tracking System (`cost_tracker.py`)

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
    llm_type: str  # "root" or "child"
    call_id: str

class CostTracker:
    def __init__(self, config: Config):
        self.config = config
        self.usage_log: List[TokenUsage] = []
        self.total_cost: float = 0.0

    def record_usage(self, input_tokens: int, output_tokens: int,
                     llm_type: str, call_id: str) -> TokenUsage:
        """Record token usage and compute cost."""

    def check_budget(self) -> bool:
        """Return True if within budget, False if exceeded."""

    def get_remaining_budget(self) -> float:
        """Return remaining budget in USD."""

    def raise_if_over_budget(self):
        """Raise BudgetExceededError if over budget."""

class BudgetExceededError(Exception):
    """Raised when the cost budget is exceeded."""
    pass
```

### 3. Circle Packing Evaluator (`evaluation/circle_packing.py`)

**Already Implemented** - See `src/tetris_evolve/evaluation/circle_packing.py`

```python
class CirclePackingEvaluator:
    def __init__(self, target: float = 2.635, n_circles: int = 26,
                 timeout_seconds: int = 30):
        self.target = target
        self.n_circles = n_circles
        self.timeout_seconds = timeout_seconds

    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        Evaluate circle packing code.

        Returns:
            {
                'valid': bool,
                'sum_radii': float,
                'target_ratio': float,
                'combined_score': float,
                'eval_time': float,
                'error': Optional[str]
            }
        """
```

Key features:
- Executes code in isolated subprocess (safety + timeout)
- Validates geometric constraints
- Returns standardized metrics

### 4. REPL Environment (`repl.py`)

```python
@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: Dict[str, Any]
    execution_time: float

class REPLEnvironment:
    def __init__(self, evolution_api: 'EvolutionAPI'):
        self.globals = self._create_safe_globals()
        self.locals = {}
        self.evolution_api = evolution_api

    def _create_safe_globals(self) -> Dict:
        """Create globals dict with safe builtins + evolution API."""
        return {
            '__builtins__': {
                'print': print, 'len': len, 'str': str, 'int': int,
                'float': float, 'list': list, 'dict': dict,
                'range': range, 'enumerate': enumerate,
                'max': max, 'min': min, 'sorted': sorted, 'sum': sum,
                '__import__': __import__,
            },
            # Evolution API functions
            'spawn_child_llm': self.evolution_api.spawn_child_llm,
            'evaluate_program': self.evolution_api.evaluate_program,
            'advance_generation': self.evolution_api.advance_generation,
            'terminate_evolution': self.evolution_api.terminate_evolution,
            'get_generation_history': self.evolution_api.get_generation_history,
            'get_best_trials': self.evolution_api.get_best_trials,
            'get_cost_remaining': self.evolution_api.get_cost_remaining,
        }

    def execute(self, code: str) -> REPLResult:
        """Execute Python code in the REPL environment."""
```

### 5. Evolution API (`evolution_api.py`)

```python
@dataclass
class TrialResult:
    trial_id: str
    code: str
    metrics: Dict[str, float]
    reasoning: str
    success: bool
    error: Optional[str]

class EvolutionAPI:
    def __init__(self, config: Config, cost_tracker: CostTracker,
                 logger: ExperimentLogger, child_llm: LLMClient,
                 evaluator: CirclePackingEvaluator):
        self.config = config
        self.cost_tracker = cost_tracker
        self.logger = logger
        self.child_llm = child_llm
        self.evaluator = evaluator

        self.current_generation = 0
        self.generations: List[List[TrialResult]] = [[]]

    def spawn_child_llm(self, prompt: str,
                        parent_id: Optional[str] = None) -> Dict:
        """
        Spawn a child LLM to generate a circle packing program.

        Returns:
            {
                'trial_id': str,
                'code': str,
                'metrics': {'sum_radii': float, 'target_ratio': float, ...},
                'reasoning': str,
                'success': bool,
                'error': Optional[str]
            }
        """

    def evaluate_program(self, code: str) -> Dict[str, float]:
        """Evaluate a circle packing program directly."""
        return self.evaluator.evaluate(code)

    def advance_generation(self, selected_trial_ids: List[str],
                          reasoning: str) -> int:
        """Move to next generation with selected trials as parents."""

    def terminate_evolution(self, reason: str) -> Dict:
        """End evolution and return final results."""

    def get_best_trials(self, n: int = 5) -> List[Dict]:
        """Get top n trials by sum_radii."""

    def get_cost_remaining(self) -> float:
        """Get remaining budget in USD."""
```

### 6. Root LLM Orchestrator (`root_llm.py`)

```python
class RootLLMOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.cost_tracker = CostTracker(config)
        self.logger = ExperimentLogger(...)
        self.evaluator = CirclePackingEvaluator(config.evaluation)
        self.evolution_api = EvolutionAPI(...)
        self.repl = REPLEnvironment(self.evolution_api)

    def run(self) -> Dict:
        """Run the evolution process."""
        self.messages = self._build_initial_messages()

        for iteration in range(self.config.root_llm.max_iterations):
            response = self.llm_client.generate(self.messages)
            code_blocks = self._extract_code_blocks(response)

            for code in code_blocks:
                result = self.repl.execute(code)
                # Add result to conversation

            if self._check_termination(response):
                break

        return self.logger.save_experiment()
```

---

## Data Flow

### 1. Experiment Initialization

```
User runs: python -m tetris_evolve --config config.yaml

1. Load and validate configuration
2. Create experiment directory
3. Initialize CostTracker, ExperimentLogger
4. Initialize LLM clients (root + child)
5. Initialize CirclePackingEvaluator
6. Initialize EvolutionAPI
7. Initialize REPLEnvironment
8. Create RootLLMOrchestrator
9. Start Root LLM loop
```

### 2. Root LLM Turn Cycle

```
1. Root LLM receives conversation history
2. Root LLM generates response
3. System extracts ```repl``` code blocks
4. For each code block:
   a. Execute in REPL environment
   b. Capture stdout/stderr
   c. Append to conversation as user message
5. Check for terminate_evolution() call
6. Check budget
7. Repeat until termination or max iterations
```

### 3. Child LLM Spawn Cycle

```
Root LLM calls: spawn_child_llm(prompt, parent_id)

1. Check budget (raise if exceeded)
2. Pass the prompt directly to the child LLM
   - The Root LLM is responsible for crafting the full prompt
   - This includes problem specification, strategy guidance, and parent code
   - No template wrapping - the Root LLM has full control over child prompts
3. Call child LLM API with the provided prompt
4. Extract code from response (```python``` blocks)
5. Evaluate using the configured evaluator:
   - Run in subprocess with timeout
   - Validate constraints
   - Compute metrics
6. Log trial
7. Return result to Root LLM
```

---

## Prompts

### Root LLM System Prompt

The Root LLM receives a system prompt documenting the available functions and REPL usage.
The Root LLM is responsible for:
1. Crafting all prompts sent to child LLMs (no predefined templates)
2. Deciding what problem specification, constraints, and guidance to include
3. Managing the evolutionary process through the available functions

```
You are orchestrating an evolutionary process to develop algorithms.

## Available Functions

### spawn_child_llm(prompt: str, parent_id: Optional[str] = None) -> dict
Spawn a child LLM with the given prompt.
- prompt: The complete prompt to send to the child LLM (you control the full content)
- parent_id: Optional trial ID to associate as parent (for tracking lineage)
- Returns: {trial_id, code, metrics, reasoning, success, error}

### evaluate_program(code: str) -> dict
Evaluate code directly using the configured evaluator.
- Returns: {valid, ..., eval_time, error} (metrics depend on evaluator)

### advance_generation(selected_trial_ids: list[str], reasoning: str) -> int
Move to next generation with selected trials as parents.

### terminate_evolution(reason: str) -> dict
End evolution and return final results.

### get_best_trials(n: int = 5) -> list[dict]
Get top n trials by score.

### get_cost_remaining() -> float
Get remaining budget in USD.

## How to Use the REPL

Write Python code in ```repl``` blocks:

```repl
# Craft a detailed prompt for the child LLM
prompt = """
You are tasked with writing a circle packing algorithm.
Pack 26 circles into a unit square [0,1] x [0,1] to maximize sum of radii.
...your detailed instructions...
"""
result = spawn_child_llm(prompt)
print(f"Score: {result['metrics']}")
```

## Guidelines

1. You are responsible for crafting effective prompts for child LLMs
2. Include problem specification, constraints, and strategy guidance in your prompts
3. When mutating, include the parent code in your prompt to the child
4. Monitor budget with get_cost_remaining()
5. Terminate when improvement plateaus or budget is low
```

**Note**: There is no predefined "Child LLM Prompt Template". The Root LLM has full control
over what prompts are sent to child LLMs. This allows the Root LLM to:
- Adapt prompts based on what strategies are working
- Include different levels of detail for different approaches
- Incorporate learned insights into subsequent prompts

---

## PoC Results

Our proof-of-concept testing validated the architecture:

| Strategy | Valid | Sum of Radii | Target Ratio |
|----------|-------|--------------|--------------|
| Hexagonal | Yes | 2.08 | 0.79 |
| Grid | Yes | 1.77 | 0.67 |
| Corner-first | Yes | 1.65 | 0.63 |
| Concentric | Yes | 1.08 | 0.41 |
| Greedy | No | - | - |

**Key findings**:
- Hexagonal packing performs best among simple strategies
- Greedy approaches can fail validation due to numerical issues
- Optimization-based approaches (scipy) have potential but need careful implementation
- Full pipeline (spawn → evaluate → select → advance) works end-to-end

---

## File Structure

```
src/tetris_evolve/
├── __init__.py
├── config.py                           # Configuration loading
├── cost_tracker.py                     # Budget enforcement
├── logger.py                           # Experiment logging
├── repl.py                             # REPL environment
├── evolution_api.py                    # Evolution API
├── root_llm.py                         # Root orchestrator
├── main.py                             # CLI entry point
├── exceptions.py                       # Custom exceptions
├── evaluation/
│   ├── __init__.py
│   └── circle_packing.py               # Circle packing evaluator (DONE)
├── llm/
│   ├── __init__.py
│   └── client.py                       # Anthropic wrapper
└── utils/
    ├── __init__.py
    └── code_extraction.py              # Code parsing utilities

tests/
├── __init__.py
├── conftest.py
├── test_config.py
├── test_cost_tracker.py
├── test_logger.py
├── test_repl.py
├── test_evaluator.py
├── test_evolution_api.py
├── test_root_llm.py
├── test_integration.py
└── test_e2e.py

experiments/
├── poc_repl.py                         # REPL PoC (DONE)
├── poc_cost_tracker.py                 # Cost tracking PoC (DONE)
├── poc_evaluator.py                    # Evaluator PoC (DONE)
├── poc_integration.py                  # Mock integration PoC (DONE)
└── poc_circle_packing_integration.py   # Circle packing integration (DONE)

configs/
└── example_config.yaml

docs/
├── DESIGN.md                           # This document
├── DESIGN_CIRCLE_PACKING.md            # Circle packing specifics
└── IMPLEMENTATION_TODO.md              # Task list
```

---

## Metrics Tracked

### Per Trial
- trial_id
- generation
- parent_id (if mutation)
- code (full source)
- prompt (what was asked)
- response (full LLM response)
- reasoning (extracted explanation)
- valid (geometric validity)
- sum_radii
- target_ratio
- combined_score
- eval_time
- input_tokens
- output_tokens
- cost_usd
- timestamp

### Per Generation
- generation_num
- num_trials
- num_successful_trials
- best_sum_radii
- best_trial_id
- selected_trial_ids
- selection_reasoning
- total_cost_this_gen
- timestamp

### Per Experiment
- experiment_id
- config (full copy)
- num_generations
- total_trials
- best_trial_overall
- final_metrics
- total_cost
- total_input_tokens
- total_output_tokens
- start_time
- end_time
- duration
- termination_reason
