# LLM Environment Evolution Design Document

## Executive Summary

This system combines **AlphaEvolve's evolutionary coding approach** with **Recursive LLM (RLM) hierarchical decision-making** to evolve code that plays PufferLib environments optimally. While initially targeting Tetris, the architecture is **environment-agnostic** and can be applied to any PufferLib/Gymnasium-compatible game or task.

The Root LLM acts as an evolutionary strategist, selecting promising candidates and guiding evolution, while Recursive Child LLMs generate code mutations and implement new variants.

## Core Concepts

### 1. AlphaEvolve Integration
- **Evolutionary Framework**: Maintain a program database with generations of environment-playing algorithms
- **Automated Evaluation**: Each candidate is scored by running environment simulations with environment-specific metrics
- **Iterative Improvement**: High-performing programs inform future generations
- **Dual Strategy**: Use both exploration (many diverse ideas) and exploitation (refine best candidates)
- **Environment Agnostic**: Works with any PufferLib/Gymnasium environment

### 2. Recursive LLM Integration
- **Root LLM (Depth=0)**: Autonomous evolutionary orchestrator that:
  - **Full control over evolution lifecycle**: Decides when to continue, pivot, or terminate
  - Analyzes performance data across all candidates in current generation
  - **Dynamically decides how many programs advance to next generation** (not hardcoded)
  - **Assigns same solution to multiple RLLMs to explore different improvement directions**
  - **Decides specific focus areas for each Recursive LLM** (e.g., "improve hole management", "optimize piece placement scoring", "increase lookahead depth")
  - **Determines termination criteria**: Stops evolution when satisfied (convergence, time limits, performance targets)
  - **Adapts strategy mid-evolution**: Can change approach based on what's working
  - Provides rich context to Child LLMs about what to optimize
- **Child LLMs (Depth=1+)**: Code generators that:
  - Receive parent programs + strategic guidance from Root
  - Generate new code variants based on Root's directives
  - Apply specific mutations/improvements
  - Return generated code to Root for evaluation

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                      ROOT LLM                           │
│  (Strategic Decision Maker - Depth 0)                   │
│                                                          │
│  Responsibilities:                                       │
│  • Control entire evolution process autonomously        │
│  • Decide whether to continue or terminate evolution    │
│  • Analyze generation performance metrics               │
│  • Dynamically decide N candidates for next generation  │
│  • Assign same solution to multiple RLLMs               │
│  • Craft specific focus areas for each RLLM             │
│  • Adapt strategies based on progress                   │
│  • Maintain evolutionary memory/insights                │
└─────────────┬───────────────────────────────────────────┘
              │
              │ Spawns Child LLMs with:
              │ - Parent program code
              │ - Mutation strategy
              │ - Performance context
              │ - Specific optimization goals
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              CHILD LLMs (Parallel)                      │
│         (Code Generators - Depth 1)                      │
│                                                          │
│  Child 1         Child 2         Child 3                │
│  Parent: Prog A  Parent: Prog A  Parent: Prog B        │
│  Focus: holes    Focus: speed    Focus: lookahead      │
│  (Same parent, different improvement directions)        │
└─────────────┬───────────────────────────────────────────┘
              │
              │ Return generated code variants
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              EVALUATION ENGINE                          │
│                                                          │
│  • Run Tetris simulations for each variant             │
│  • Collect metrics (score, lines, survival time)       │
│  • Store results in Program Database                   │
└─────────────┬───────────────────────────────────────────┘
              │
              │ Metrics feed back to Root LLM
              │
              ▼
           [Next Generation]
```

### Component Architecture

#### 1. Program Database
```python
{
  "generation": int,
  "program_id": str,
  "code": str,
  "parent_ids": [str],
  "mutation_strategy": str,
  "metrics": {
    "avg_score": float,
    "avg_lines_cleared": float,
    "avg_survival_time": float,
    "games_played": int,
    "std_dev": float
  },
  "metadata": {
    "created_at": timestamp,
    "root_llm_notes": str,
    "child_llm_id": str
  }
}
```

#### 2. Shared REPL Environment

**Key Design**: Root LLM and all Child RLLMs share the same Python REPL environment. This means:
- Variables defined by Root are accessible to Children
- Children can see and use functions defined in the REPL
- Shared context (program database, metrics) is available to all
- This enables efficient communication and context sharing

The Root LLM operates in this shared REPL with access to:

```python
# Available in Root LLM context
current_generation = 5
program_database = [...] # All programs from all generations
current_gen_programs = [...] # Programs from current generation only
metrics_summary = {...} # Statistical analysis of current generation

# Functions available to Root LLM
def spawn_rllm(prompt: str):
    """
    Spawn a Recursive Child LLM with a custom prompt.
    The Root LLM has complete control over the prompt content.

    The Child RLLM will execute in the SAME REPL environment as Root,
    so they share access to variables, functions, and context.

    CONSTRAINT: Total RLLMs spawned per generation cannot exceed max_child_rllms_per_generation.
    The function will raise an error if limit is exceeded.

    Args:
        prompt: Complete prompt for the Child RLLM. Root decides all content:
                - What code to improve
                - What aspects to focus on
                - What context to provide
                - What constraints to enforce
                - What output format is expected

    Returns:
        Result from Child RLLM execution (typically generated code as string)

    Raises:
        ResourceLimitError: If max_child_rllms_per_generation exceeded

    Example:
        prompt = '''
        You are improving a Tetris player. Here's the current best program:

        {parent_code}

        This program scores an average of 12,345 points but creates too many holes.
        Your task: Reduce hole creation while maintaining performance.

        Return only the improved Python code.
        '''

        new_code = spawn_rllm(prompt)
    """
    pass

def evaluate_program(code, num_games=100):
    """
    Run Tetris simulations and return metrics.

    Returns:
        Metrics dict with performance data
    """
    pass

def get_performance_analysis(generation=None):
    """
    Get detailed analysis of a generation's performance.
    If generation is None, analyzes current generation.

    Returns:
        Analysis dict with statistics, trends, insights
    """
    pass

def get_historical_trends():
    """
    Get performance trends across all generations.

    Returns:
        Trend data: improvement rate, convergence indicators, etc.
    """
    pass

def advance_generation(selected_programs):
    """
    Move to next generation with dynamically selected programs.
    Number of programs is decided by Root LLM, not hardcoded.

    CONSTRAINT: Will raise error if current generation >= max_generations.

    Returns:
        New generation number

    Raises:
        ResourceLimitError: If max_generations exceeded
    """
    pass

def terminate_evolution(reason, best_program):
    """
    Terminate the evolution process.

    Args:
        reason: Explanation for termination (convergence, target reached, etc.)
        best_program: The final best program to save

    Returns:
        Summary of evolution results
    """
    pass

def modify_parameters(param_updates):
    """
    Dynamically modify evolution parameters mid-run.

    Args:
        param_updates: Dict of parameter changes (e.g., {'games_per_evaluation': 200})
    """
    pass
```

The Root LLM can:
- **Control the evolution loop**: Decide when to continue or stop
- Inspect performance data and historical trends
- Identify patterns in successful programs
- Decide selection criteria dynamically (not hardcoded)
- **Craft complete custom prompts for Child RLLMs** (full control over prompt content)
- Spawn multiple Child LLMs in parallel (all sharing the same REPL)
- Modify parameters mid-evolution (e.g., increase games_per_eval for closer races)
- Terminate evolution when satisfied with results

**REPL Sharing**: All Child RLLMs execute in the same REPL as Root, enabling:
- Access to shared variables (program_database, metrics_summary, etc.)
- Ability to call shared functions
- Efficient context passing without explicit serialization

#### 3. Child RLLM Execution

Each Child RLLM:
- Receives a **custom prompt crafted by Root** (Root has complete control)
- Executes in the **same REPL environment as Root** (shared context)
- Can access variables in the REPL (program_database, metrics_summary, etc.)
- Returns a result (typically generated code as string)

**Example Child RLLM Execution:**

```python
# Root crafts a custom prompt
prompt = """
You are improving a Tetris-playing program.

Current program (Generation 5, Score: 12,345):
{code_here}

Problem: Creates too many holes (avg 15 holes per game).
Top performer has only 5 holes per game.

Task: Modify the evaluate_position() function to heavily penalize holes.
Add a hole_penalty term to the scoring.

Available in REPL: program_database, current_gen_programs, metrics_summary

Return only the improved Python code (no explanations).
"""

# Root spawns Child RLLM with this prompt
improved_code = spawn_rllm(prompt)

# Child executes in same REPL, can access program_database if needed
# Returns the improved code
```

## Evolutionary Strategy

### Evolution Lifecycle (Root LLM Controlled)

The Root LLM has complete autonomy over the evolution process within resource constraints. There is no hardcoded loop - the Root decides what to do at each step.

**Resource Constraints:**
- Maximum generations (hard limit)
- Maximum RLLMs per generation (hard limit)
- Maximum time (hard limit)

The system will enforce these limits and terminate evolution if exceeded.

**Root LLM Decision Flow:**

```
1. Initialize →
2. Loop: Analyze Current State →
   - Should I continue evolution?
     - If NO → Terminate with best program
     - If YES → Continue to next decision
   - What programs should advance?
   - How many RLLMs per program?
   - What focus areas for each RLLM?
   - Should I modify any parameters?
   - Spawn RLLMs → Evaluate results → Store in DB
   - Go to step 2
```

**Example Root LLM Reasoning (Generation 5):**

```python
# Root's internal reasoning (pseudocode)
current_analysis = get_performance_analysis()
trends = get_historical_trends()

# Decision: Should I continue?
if trends['improvement_rate'] < 0.01 and current_analysis['top_score'] > 10000:
    terminate_evolution(
        reason="Convergence achieved: improvement rate < 1% and top score > 10k",
        best_program=current_analysis['best_program']
    )
else:
    # Decision: This generation shows promise, continue
    # Top 3 programs use lookahead depth 2 and score 40% better
    # Strategy: Focus on lookahead optimization

    selected = current_analysis['programs'][0:8]  # Dynamic selection

    # Decision: Assign multiple RLLMs to best program with custom prompts

    # RLLM 1: Focus on lookahead
    prompt1 = f"""
    Improve this Tetris player (current score: {selected[0].score}):
    {selected[0].code}

    Focus: Increase lookahead depth from 2 to 3.
    The current program uses 2-piece lookahead. Extend to 3 while maintaining speed.
    Return only the improved Python code.
    """
    new_code1 = spawn_rllm(prompt1)

    # RLLM 2: Focus on hole management
    prompt2 = f"""
    Improve this Tetris player:
    {selected[0].code}

    Focus: Minimize hole creation during lookahead evaluation.
    Add hole penalty to the scoring function.
    Return only the improved Python code.
    """
    new_code2 = spawn_rllm(prompt2)

    # RLLM 3: Speed optimization
    prompt3 = f"""
    Optimize this Tetris player for speed:
    {selected[0].code}

    Add caching for lookahead computations to improve performance.
    Return only the improved Python code.
    """
    new_code3 = spawn_rllm(prompt3)

    advance_generation(selected)
    # Loop continues...
```

**Key Phases (Root-Controlled):**

1. **Initialization (Generation 0)**
   - Root LLM generates initial diverse population
   - Strategies: random, greedy, hole minimization, etc.
   - Population size decided by Root

2. **Continuous Decision Loop**
   - **Termination Check**: Root decides whether to continue based on:
     - Performance convergence
     - Target metrics achieved
     - Time/cost budgets
     - Diminishing returns
   - **Analysis**: Root analyzes current generation metrics and historical trends
   - **Selection**: Root dynamically decides which programs advance (N varies)
   - **RLLM Assignment**: Root assigns focus areas and spawns RLLMs (multiple per promising program)
   - **Parameter Adjustment**: Root can modify parameters (e.g., more games for evaluation if scores are close)
   - **Generation Advancement**: Root decides when to advance to next generation

3. **Termination**
   - Root terminates when satisfied with results
   - No hardcoded generation limit
   - Root provides termination reasoning and final best program

### Mutation Strategies

The Root LLM can direct various mutation types:

1. **Targeted Improvement**
   - "Reduce hole creation in the current top performer"
   - "Optimize the scoring function for long-term stability"

2. **Feature Addition**
   - "Add piece lookahead to this program"
   - "Implement hold piece optimization"

3. **Algorithmic Shift**
   - "Convert this greedy approach to use beam search"
   - "Add Monte Carlo tree search exploration"

4. **Crossover/Recombination**
   - "Combine the hole management from Program A with the scoring from Program B"

5. **Exploration**
   - "Create a completely new approach using genetic algorithms"
   - "Try a reinforcement learning-inspired value function"

## Environment Abstraction (Generic Design)

### Environment Interface

The system is designed to work with **any PufferLib/Gymnasium environment**. This abstraction allows applying the same evolution framework to Tetris, Atari, board games, or custom environments.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pufferlib
import gymnasium as gym

class EnvironmentConfig(ABC):
    """
    Abstract base class for environment configuration.
    Each environment (Tetris, Atari, etc.) implements this interface.
    """

    @abstractmethod
    def get_env_id(self) -> str:
        """Return PufferLib/Gymnasium environment ID"""
        pass

    @abstractmethod
    def get_env_kwargs(self) -> Dict[str, Any]:
        """Return environment-specific configuration"""
        pass

    @abstractmethod
    def get_observation_description(self) -> str:
        """Return human-readable description of observations for LLM"""
        pass

    @abstractmethod
    def get_action_description(self) -> str:
        """Return human-readable description of action space for LLM"""
        pass

    @abstractmethod
    def get_reward_description(self) -> str:
        """Return description of reward structure for LLM"""
        pass

    @abstractmethod
    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """Extract environment-specific metrics from episode info"""
        pass

    @abstractmethod
    def get_player_interface_template(self) -> str:
        """Return code template for player interface"""
        pass


class GenericEnvironmentWrapper:
    """
    Generic wrapper around any PufferLib/Gymnasium environment.
    Works with Tetris, Atari, custom environments, etc.
    """

    def __init__(self, env_config: EnvironmentConfig, num_envs: int = 1):
        self.env_config = env_config
        self.num_envs = num_envs

        # Initialize PufferLib vectorized environment
        self.env = pufferlib.vector.make(
            env_config.get_env_id(),
            num_envs=num_envs,
            **env_config.get_env_kwargs()
        )

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        """Reset environment and return initial observation"""
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        return self.env.step(action)

    def get_observation_description(self) -> str:
        """Get human-readable observation description for LLM"""
        return self.env_config.get_observation_description()

    def get_action_description(self) -> str:
        """Get human-readable action description for LLM"""
        return self.env_config.get_action_description()

    def extract_metrics(self, info: Dict) -> Dict[str, float]:
        """Extract environment-specific metrics"""
        return self.env_config.extract_episode_metrics(info)
```

### Example: Tetris Environment Configuration

```python
class TetrisConfig(EnvironmentConfig):
    """Tetris-specific environment configuration"""

    def get_env_id(self) -> str:
        return "Tetris-v0"  # or custom Tetris environment ID

    def get_env_kwargs(self) -> Dict[str, Any]:
        return {
            "width": 10,
            "height": 20,
            "preview_pieces": 5,
            "enable_hold": True
        }

    def get_observation_description(self) -> str:
        return """
        Observation: dict with keys:
        - board: (height, width) array, 0=empty, 1=filled
        - current_piece: dict with type, rotation, position
        - next_pieces: list of upcoming pieces (for lookahead)
        - hold_piece: held piece or None
        - score: current score
        - lines_cleared: total lines cleared
        """

    def get_action_description(self) -> str:
        return """
        Action: dict with keys:
        - rotation: 0-3 (piece rotation)
        - column: 0-9 (target column)
        - use_hold: bool (whether to use hold piece)
        """

    def get_reward_description(self) -> str:
        return """
        Reward structure:
        - Line clears: +100 * lines_cleared^2
        - Game over: -1000
        - Each move: -0.01 (encourages efficiency)
        """

    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """Extract Tetris-specific metrics"""
        return {
            "score": info.get("score", 0),
            "lines_cleared": info.get("lines_cleared", 0),
            "survival_time": info.get("moves", 0),
            "max_height": info.get("max_height", 0),
            "holes": info.get("holes", 0)
        }

    def get_player_interface_template(self) -> str:
        return """
class TetrisPlayer:
    def select_action(self, observation):
        '''
        Given observation, return action.

        Args:
            observation: dict from environment

        Returns:
            action: dict with rotation, column, use_hold
        '''
        # Your code here
        pass
"""
```

### Example: Atari Environment Configuration

```python
class AtariConfig(EnvironmentConfig):
    """Atari-specific environment configuration"""

    def __init__(self, game: str = "Pong-v5"):
        self.game = game

    def get_env_id(self) -> str:
        return self.game

    def get_env_kwargs(self) -> Dict[str, Any]:
        return {
            "frameskip": 4,
            "repeat_action_probability": 0.0
        }

    def get_observation_description(self) -> str:
        return """
        Observation: (210, 160, 3) RGB image array
        - Pixel values in range [0, 255]
        - Represents current game screen
        """

    def get_action_description(self) -> str:
        return f"""
        Action: integer from action space
        - Depends on specific Atari game
        - Typically: 0-5 (NOOP, FIRE, UP, DOWN, LEFT, RIGHT)
        """

    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """Extract Atari-specific metrics"""
        return {
            "score": info.get("episode_reward", 0),
            "survival_time": info.get("episode_length", 0)
        }

    def get_player_interface_template(self) -> str:
        return """
class AtariPlayer:
    def select_action(self, observation):
        '''
        Given observation (game screen), return action.

        Args:
            observation: (210, 160, 3) RGB array

        Returns:
            action: integer action
        '''
        # Your code here
        pass
"""
```

### Player Interface (Generic)

Generated code implements a player that follows the environment's interface:

```python
class Player:
    """
    Generic player interface.
    Environment-specific implementations follow this pattern.
    """

    def select_action(self, observation):
        """
        Given current observation, return action.

        Args:
            observation: Environment-specific observation

        Returns:
            action: Environment-specific action
        """
        # Generated code implements this logic
        pass
```

### Adding New Environments

To add support for a new environment (e.g., Chess, StarCraft, custom game):

1. **Create Environment Config** (`src/env_evolve/environment/your_game.py`):
```python
class YourGameConfig(EnvironmentConfig):
    def get_env_id(self) -> str:
        return "YourGame-v0"

    def get_env_kwargs(self) -> Dict[str, Any]:
        return {"param": "value"}

    # Implement all abstract methods (observations, actions, rewards, metrics, template)
```

2. **Create Configuration File** (`configs/your_game.yaml`):
```yaml
environment:
  type: "your_game"
  config:
    env_id: "YourGame-v0"
    # Your game-specific parameters

evolution:
  initial_population_size: 30
  # ... other evolution parameters
```

3. **Run Evolution**:
```bash
python -m env_evolve.main --config configs/your_game.yaml
```

That's it! The entire evolution system (Root LLM, Child RLLMs, evaluation, logging) works automatically with your new environment.

### Evaluation Framework (Generic)

```python
def evaluate_player(
    player_code: str,
    env_config: EnvironmentConfig,
    num_episodes: int = 100
) -> Dict[str, Any]:
    """
    Run player through multiple episodes and collect metrics.
    Works with any environment that implements EnvironmentConfig.

    Args:
        player_code: Generated player code as string
        env_config: Environment configuration (Tetris, Atari, etc.)
        num_episodes: Number of episodes to run

    Returns:
        {
            "env_metrics": {
                # Environment-specific metrics from extract_episode_metrics()
                "metric_name": {"mean": float, "std": float, "min": float, "max": float},
                ...
            },
            "generic_metrics": {
                "total_reward": {"mean": float, "std": float, "min": float, "max": float},
                "episode_length": {"mean": float, "std": float, "min": float, "max": float},
            },
            "code_errors": int,
            "episode_results": [
                {"episode_id": 0, "reward": float, "length": int, "env_metrics": {...}},
                ...
            ]
        }
    """
    pass
```

**Key Insight**: The evaluation framework extracts both:
1. **Generic metrics** (reward, episode length) - work for all environments
2. **Environment-specific metrics** (lines cleared, holes, etc.) - defined by EnvironmentConfig

## Implementation Phases

**NOTE: Each phase follows Test-Driven Development (TDD). Write tests first, then implement to pass tests.**

### Phase 0: Project Setup
**Setup Tasks:**
- [ ] Create project directory structure (src, tests, configs, etc.)
- [ ] Set up `pyproject.toml` with dependencies
- [ ] Install PufferLib, Gymnasium and verify installation
- [ ] Research available PufferLib/Gymnasium environments (Tetris, Atari, etc.)
- [ ] Set up pytest configuration and test structure
- [ ] Create initial configuration files (tetris.yaml, atari_pong.yaml templates)
- [ ] Set up logging configuration
- [ ] Initialize git repository structure
- [ ] Create README with setup instructions and environment extensibility guide

**Directory Structure:**
```
env_evolve/                       # Generic name (not tetris-specific)
├── src/
│   ├── env_evolve/
│   │   ├── __init__.py
│   │   ├── environment/          # Generic environment wrapper + configs
│   │   │   ├── __init__.py
│   │   │   ├── base.py           # EnvironmentConfig ABC
│   │   │   ├── wrapper.py        # GenericEnvironmentWrapper
│   │   │   ├── tetris.py         # TetrisConfig
│   │   │   ├── atari.py          # AtariConfig
│   │   │   └── ...               # Other environment configs
│   │   ├── evaluation/           # Player evaluation framework
│   │   ├── database/             # Program database
│   │   ├── rlm/                  # Recursive LLM framework
│   │   ├── root_llm/             # Root LLM logic
│   │   ├── child_llm/            # Child RLLM logic
│   │   └── evolution/            # Evolution loop
├── tests/
│   ├── test_environment/
│   ├── test_evaluation/
│   ├── test_database/
│   ├── test_rlm/
│   └── test_evolution/
├── configs/
│   ├── tetris.yaml               # Tetris-specific config
│   ├── atari_pong.yaml           # Atari Pong config
│   └── ...                       # Other environment configs
├── pyproject.toml
├── DESIGN.md
└── README.md
```

### Phase 1: Core Infrastructure
**Tests First:**
- [ ] Write tests for generic environment wrapper (test with mock environment)
- [ ] Write tests for Tetris environment config
- [ ] Write tests for player evaluation framework (generic)
- [ ] Write tests for program database CRUD operations
- [ ] Write tests for basic REPL environment execution

**Implementation:**
- [ ] Implement EnvironmentConfig ABC and GenericEnvironmentWrapper
- [ ] Implement TetrisConfig (and optionally another env for testing generality)
- [ ] Implement evaluation framework (environment-agnostic)
- [ ] Create program database (in-memory + file persistence, no SQLite)
- [ ] Build shared REPL framework (Root + Children share same REPL)
- [ ] Verify all tests pass

**Deliverables:**
- Generic environment wrapper that works with any PufferLib environment
- At least one environment config (Tetris) fully implemented
- Evaluation framework that works with any environment
- Program database (in-memory + file persistence)
- Shared REPL execution environment (Root + Children)
- All tests passing

### Phase 2: Root LLM Integration
**Tests First:**
- [ ] Write tests for Root LLM REPL function injection
- [ ] Write tests for performance analysis functions
- [ ] Write tests for dynamic selection (variable N programs)
- [ ] Write tests for multi-RLLM assignment to same parent

**Implementation:**
- [ ] Implement Root LLM REPL environment with injected functions
- [ ] Add functions for program analysis and metrics
- [ ] Build generation management system
- [ ] Create prompt templates emphasizing dynamic decisions
- [ ] Test Root's ability to select N programs and assign multiple RLLMs
- [ ] Verify all tests pass

**Deliverables:**
- Root LLM REPL environment with function injection
- Performance analysis and metrics functions
- Generation management system
- Dynamic selection working (variable N programs)
- Multi-RLLM assignment capability
- All tests passing

### Phase 3: Recursive Child LLM Integration
**Tests First:**
- [ ] Write tests for RLLM spawning with focus areas
- [ ] Write tests for mutation directive creation
- [ ] Write tests for code generation pipeline
- [ ] Write tests for code validation and safety
- [ ] Write tests for parallel RLLM execution

**Implementation:**
- [ ] Implement RLLM spawning with focus_area parameter
- [ ] Create mutation directive templates
- [ ] Build code generation pipeline
- [ ] Add code validation and safety checks
- [ ] Implement parallel RLLM generation
- [ ] Verify all tests pass

**Deliverables:**
- RLLM spawning system with focus_area parameter
- Mutation directive templates
- Code generation and validation pipeline
- Parallel RLLM execution
- All tests passing

### Phase 4: Evolution Loop
**Tests First:**
- [ ] Write tests for full generation lifecycle
- [ ] Write tests for Root's dynamic decision-making
- [ ] Write tests for checkpoint save/restore
- [ ] Write tests for edge cases (no improvements, code errors, etc.)

**Implementation:**
- [ ] Connect all components into evolution loop
- [ ] Implement full generation lifecycle
- [ ] Add comprehensive logging and visualization
- [ ] Create checkpoint/resume functionality
- [ ] Run initial evolution experiments
- [ ] Verify all tests pass

**Deliverables:**
- Complete end-to-end evolution system
- Full generation lifecycle working
- Logging and visualization
- Checkpoint/resume functionality
- Initial evolution experiments completed
- All tests passing

### Phase 5: Optimization & Scaling
**Tests First:**
- [ ] Write tests for parallel evaluation
- [ ] Write tests for evaluation caching
- [ ] Write tests for diversity metrics

**Implementation:**
- [ ] Parallelize game evaluation
- [ ] Optimize LLM token usage
- [ ] Add caching for repeated evaluations
- [ ] Implement diversity preservation mechanisms
- [ ] Performance profiling and optimization
- [ ] Verify all tests pass

**Deliverables:**
- Parallel evaluation system
- Evaluation caching
- Diversity preservation mechanisms
- Performance profiling results
- Optimized token usage
- All tests passing

## Technical Stack

### Core Components
- **Language**: Python 3.10+
- **LLM API**: OpenAI API (GPT-4 for Root, GPT-3.5/4 for Children) or Anthropic Claude
- **Game Engine**: PufferLib (gymnasium-compatible RL environment library)
- **Database**: SQLite for program storage
- **REPL**: Python `exec()` based sandbox (from RLM framework)
- **Testing**: pytest with test-driven development (TDD) approach

### Libraries
```
- pufferlib: Environment library (gymnasium-compatible, vectorized)
- gymnasium: RL environment interface (base for PufferLib)
- numpy: Array operations and state management
- openai / anthropic: LLM API calls
- multiprocessing: Parallel evaluation
- matplotlib/plotly: Visualization
- pytest: Test framework (TDD approach)
- pydantic: Data validation
- PyYAML: Configuration parsing
```

**Note**: No SQLite dependency for now - using file-based storage only.

## Data Persistence & Logging

### Directory Structure

Each evolution run creates a timestamped directory containing all data about that run:

```
tetris_evolve/
├── runs/
│   ├── run_2025-01-23_14-30-00/              # Timestamped run directory
│   │   ├── config.yaml                        # Config used for this run
│   │   ├── evolution_log.jsonl                # Line-delimited JSON log of all events
│   │   ├── root_llm_log.jsonl                 # All Root LLM decisions and reasoning
│   │   ├── summary.json                       # Final summary statistics
│   │   │
│   │   ├── generations/
│   │   │   ├── gen_000/
│   │   │   │   ├── generation_summary.json    # Generation-level stats
│   │   │   │   ├── programs/
│   │   │   │   │   ├── prog_00001/
│   │   │   │   │   │   ├── metadata.json      # Program metadata
│   │   │   │   │   │   ├── code.py            # The program code
│   │   │   │   │   │   ├── metrics.json       # Evaluation metrics
│   │   │   │   │   │   ├── parent_info.json   # Parent IDs and mutation info
│   │   │   │   │   │   └── evaluation_logs/   # Individual game logs
│   │   │   │   │   │       ├── game_0.json
│   │   │   │   │   │       ├── game_1.json
│   │   │   │   │   │       └── ...
│   │   │   │   │   ├── prog_00002/
│   │   │   │   │   │   └── ...
│   │   │   │   │   └── ...
│   │   │   │   └── rllm_logs/                 # Child RLLM generation logs
│   │   │   │       ├── rllm_00001.json
│   │   │   │       ├── rllm_00002.json
│   │   │   │       └── ...
│   │   │   ├── gen_001/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   ├── checkpoints/
│   │   │   ├── checkpoint_gen_000.json
│   │   │   ├── checkpoint_gen_005.json
│   │   │   └── ...
│   │   │
│   │   └── visualizations/                    # Generated plots and visualizations
│   │       ├── performance_over_time.png
│   │       ├── diversity_analysis.png
│   │       └── ...
│   │
│   ├── run_2025-01-23_16-45-00/
│   │   └── ...
│   └── ...
```

### Data Schemas

#### 1. Program Metadata (`metadata.json`)
```json
{
  "program_id": "prog_00042",
  "generation": 5,
  "created_at": "2025-01-23T14:35:22.123Z",
  "parent_ids": ["prog_00023", "prog_00024"],
  "mutation_type": "exploitation",
  "focus_area": "hole_management",
  "rllm_id": "rllm_00123",
  "root_reasoning": "Best program from gen 4, focus on reducing holes"
}
```

#### 2. Program Metrics (`metrics.json`)
```json
{
  "program_id": "prog_00042",
  "evaluation": {
    "num_games": 100,
    "completed_at": "2025-01-23T14:40:15.456Z",
    "avg_score": 12345.67,
    "std_score": 1234.56,
    "avg_lines_cleared": 234.5,
    "std_lines_cleared": 45.2,
    "avg_survival_time": 5678.9,
    "std_survival_time": 789.1,
    "max_score": 18900,
    "min_score": 3400,
    "success_rate": 0.92,
    "code_errors": 0,
    "game_results": [
      {"game_id": 0, "score": 12500, "lines": 240, "moves": 5800},
      {"game_id": 1, "score": 11900, "lines": 225, "moves": 5200},
      "... (all games)"
    ]
  }
}
```

#### 3. Parent Info (`parent_info.json`)
```json
{
  "program_id": "prog_00042",
  "parents": [
    {
      "parent_id": "prog_00023",
      "parent_generation": 4,
      "parent_score": 11234.5
    }
  ],
  "mutation_info": {
    "focus_area": "hole_management",
    "mutation_directive": {
      "strategy": "targeted_improvement",
      "guidance": "Reduce hole creation by improving placement scoring",
      "constraints": ["maintain_performance"]
    },
    "rllm_explanation": "Modified the evaluate_position() function to heavily penalize holes...",
    "expected_improvements": [
      "Fewer holes in final board state",
      "Better long-term stability"
    ]
  }
}
```

#### 4. Generation Summary (`generation_summary.json`)
```json
{
  "generation": 5,
  "started_at": "2025-01-23T14:35:00.000Z",
  "completed_at": "2025-01-23T14:55:30.000Z",
  "duration_seconds": 1230,
  "num_programs": 42,
  "num_rllms_spawned": 35,
  "statistics": {
    "avg_score": 10234.5,
    "std_score": 2345.6,
    "max_score": 15678.9,
    "min_score": 3456.7,
    "top_5_programs": ["prog_00042", "prog_00043", "prog_00044", "prog_00045", "prog_00046"],
    "improvement_from_prev_gen": 0.15,
    "diversity_score": 0.68
  },
  "root_llm_decision": {
    "selected_for_next_gen": ["prog_00042", "prog_00043", "..."],
    "num_selected": 8,
    "rllm_assignments": [
      {
        "parent_id": "prog_00042",
        "focus_areas": ["lookahead_depth", "hole_management", "speed"]
      },
      "..."
    ],
    "reasoning": "Top programs show strong hole management. Focus on increasing lookahead...",
    "continue_evolution": true,
    "parameter_modifications": {}
  }
}
```

#### 5. RLLM Log (`rllm_00001.json`)
```json
{
  "rllm_id": "rllm_00123",
  "generation": 5,
  "created_at": "2025-01-23T14:35:25.000Z",
  "completed_at": "2025-01-23T14:36:10.000Z",
  "duration_seconds": 45,
  "parent_program_id": "prog_00023",
  "focus_area": "hole_management",
  "mutation_directive": {
    "strategy": "targeted_improvement",
    "guidance": "Reduce hole creation...",
    "constraints": ["maintain_performance"]
  },
  "llm_model": "gpt-4",
  "tokens_used": {
    "prompt": 3500,
    "completion": 1200,
    "total": 4700
  },
  "generated_program_id": "prog_00042",
  "explanation": "Modified evaluate_position() to heavily penalize holes..."
}
```

#### 6. Root LLM Log Entry (`root_llm_log.jsonl`)
Line-delimited JSON, one entry per Root decision:
```json
{"timestamp": "2025-01-23T14:35:00.000Z", "generation": 5, "event": "analysis_start", "data": {...}}
{"timestamp": "2025-01-23T14:35:15.000Z", "generation": 5, "event": "performance_analysis", "data": {"top_score": 12345, "avg_score": 10234, "improvement_rate": 0.15}}
{"timestamp": "2025-01-23T14:35:18.000Z", "generation": 5, "event": "decision", "decision_type": "continue", "reasoning": "Strong improvement trend continues..."}
{"timestamp": "2025-01-23T14:35:20.000Z", "generation": 5, "event": "selection", "num_selected": 8, "selected_ids": ["prog_00042", ...]}
{"timestamp": "2025-01-23T14:35:22.000Z", "generation": 5, "event": "rllm_spawn", "rllm_id": "rllm_00123", "parent_id": "prog_00042", "focus_area": "hole_management"}
```

#### 7. Evolution Log (`evolution_log.jsonl`)
High-level events for the entire run:
```json
{"timestamp": "2025-01-23T14:30:00.000Z", "event": "run_start", "config": {...}}
{"timestamp": "2025-01-23T14:30:05.000Z", "event": "generation_start", "generation": 0}
{"timestamp": "2025-01-23T14:32:30.000Z", "event": "generation_complete", "generation": 0, "duration": 145, "num_programs": 30}
{"timestamp": "2025-01-23T14:32:31.000Z", "event": "root_decision", "generation": 0, "continue": true, "selected": 10}
{"timestamp": "2025-01-23T14:55:30.000Z", "event": "generation_complete", "generation": 5, "duration": 1230}
{"timestamp": "2025-01-23T15:00:00.000Z", "event": "run_complete", "reason": "convergence", "total_generations": 6, "best_score": 15678.9}
```

### In-Memory Program Database

For now, we use **file-based storage only** (no SQLite). The program database is:
- Loaded into memory at startup from filesystem
- Maintained in memory during evolution (as Python data structures)
- Persisted to filesystem after each generation
- Available in the shared REPL as `program_database` variable

**Future Enhancement**: Add SQLite for fast querying when the database grows large.

### Logging API

```python
class EvolutionLogger:
    """Handles all logging for evolution run"""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.evolution_log = run_dir / "evolution_log.jsonl"
        self.root_llm_log = run_dir / "root_llm_log.jsonl"

    def log_event(self, event_type: str, data: dict):
        """Log high-level evolution event"""
        pass

    def log_root_decision(self, generation: int, decision: dict):
        """Log Root LLM decision"""
        pass

    def save_program(self, program: Program, generation: int):
        """Save program with all metadata and metrics"""
        pass

    def save_generation_summary(self, generation: int, summary: dict):
        """Save generation summary"""
        pass

    def log_rllm_spawn(self, rllm_info: dict):
        """Log RLLM spawning"""
        pass

    def create_checkpoint(self, generation: int, state: dict):
        """Create checkpoint for resuming"""
        pass
```

### Benefits of This Structure

1. **Reproducibility**: Every run is fully logged with config and all decisions
2. **Analysis**: Easy to analyze trends, compare programs, visualize progress
3. **Debugging**: Detailed logs for every component
4. **Resume**: Checkpoints allow resuming interrupted runs
5. **Visualization**: All data available for creating plots and dashboards
6. **Audit Trail**: Complete history of Root LLM decisions and reasoning
7. **Program Inspection**: Easy to examine any program from any generation
8. **Performance Analysis**: Detailed metrics for every program and generation

## Configuration

### System Parameters
```yaml
# Environment configuration (supports any PufferLib environment)
environment:
  type: "tetris"  # or "atari", "custom", etc.
  config:
    # Tetris-specific (if type: tetris)
    env_id: "Tetris-v0"
    width: 10
    height: 20
    preview_pieces: 5
    enable_hold: true

    # Atari-specific (if type: atari)
    # game: "Pong-v5"
    # frameskip: 4

evolution:
  initial_population_size: 30
  episodes_per_evaluation: 100  # Root can modify this dynamically

  # Hard limits (Root CANNOT exceed these)
  max_generations: 100  # Maximum generations allowed (Root can terminate earlier)
  max_child_rllms_per_generation: 50  # Maximum RLLMs Root can spawn per generation
  max_time_minutes: 240  # Maximum runtime (4 hours)

  # Note: Root LLM has control over evolution within the constraints above
  # Values below are starting guidance - Root can modify them at any time

  # Guidance for Root LLM (suggestions, not constraints)
  suggested_selection_range: [5, 15]  # Root can choose outside this range
  suggested_rllms_per_top_program: [2, 4]  # Multiple RLLMs per promising solution

  # Termination criteria guidance (Root decides when to stop)
  termination_guidance:
    min_improvement_rate: 0.01  # Consider stopping if improvement < 1%
    target_score: 15000  # Consider stopping if this score is achieved
    convergence_window: 3  # Generations with no improvement before considering termination

  mutation_distribution_guidance:  # Suggestions, not requirements
    exploitation: 0.6  # Improve top performers
    exploration: 0.3   # Novel approaches
    crossover: 0.1     # Recombination

llm:
  root_model: "gpt-4"
  child_model: "gpt-4"  # or "gpt-3.5-turbo" for cost
  temperature_root: 0.7
  temperature_child: 0.8
  max_tokens: 4000

evaluation:
  num_episodes: 100
  parallel_episodes: 10
  timeout_per_episode: 300  # seconds
  max_steps_per_episode: 10000
```

## Root LLM Prompting Strategy

### Initial Prompt Template
```
You are the Root LLM - an autonomous agent controlling the evolution of optimal
Tetris-playing code. You have complete authority over the evolution process.

**Your Mission:**
Evolve the best possible Tetris-playing program. You decide strategy, timing, and termination.

**Your Authority:**
- Full control over evolution lifecycle (continue or terminate)
- Dynamic parameter adjustment
- Selection strategy (how many programs, which ones)
- RLLM assignment (how many per program, focus areas) - up to {max_child_rllms_per_generation} RLLMs per generation
- Evaluation depth (how many games to run)
- Termination criteria (decide when you're satisfied)

**Hard Constraints:**
- Maximum generations: {max_generations}
- Maximum RLLMs per generation: {max_child_rllms_per_generation}
- Maximum time: {max_time_minutes} minutes
- You will be automatically terminated if you exceed these limits

**Available Functions:**
- spawn_rllm(prompt: str) → result  # You craft the complete prompt
- evaluate_program(code, num_games) → metrics
- get_performance_analysis(generation=None) → analysis
- get_historical_trends() → trends
- advance_generation(selected_programs) → new_generation_number
- terminate_evolution(reason, best_program) → summary
- modify_parameters(param_updates) → None

**Current State:**
Generation: {generation} / {max_generations}
Time Elapsed: {time_elapsed} / {max_time_minutes} minutes
RLLMs Available This Generation: {max_child_rllms_per_generation}
Budget Used: {budget_used}

Current Generation Data:
{program_data}

Performance Summary:
{metrics_summary}

Historical Trends:
{historical_trends}

**Your Decision Process:**

1. **Should evolution continue?**
   - Have we converged? (improvement rate < threshold)
   - Have we hit performance targets?
   - Are we making progress or stuck?
   - Is there potential for more improvement?

   If NO: Call terminate_evolution(reason, best_program)
   If YES: Continue to step 2

2. **What's the strategy for next generation?**
   - How many programs should advance?
   - Which programs are most promising?
   - Should any get multiple RLLMs? With what focus areas?
   - How many total RLLMs to spawn? (max: {max_child_rllms_per_generation})
   - Should I modify parameters? (e.g., more games_per_eval?)

3. **Execute your strategy**
   - Spawn RLLMs with specific directives (respect RLLM limit!)
   - Advance generation
   - The system will evaluate and return control to you

**Important:**
- You are NOT bound by suggested ranges or guidelines (except hard constraints)
- Trust your analysis over heuristics
- Terminate when satisfied, not after max_generations
- Be bold in strategy changes if data suggests it
- Respect resource limits: max generations, max RLLMs per generation, max time
- Plan your RLLM budget carefully each generation

Think step-by-step and make your decision.
```

### Child RLLM Prompting (Root-Controlled)

**Key Design**: Root LLM crafts the ENTIRE prompt for each Child RLLM. No templates or structured format required.

**Example Root-Crafted Prompts:**

**Example 1: Targeted Improvement**
```
You are improving a Tetris player that scores 12,345 points on average.

Current code:
{code}

Analysis: The program creates too many holes (avg 15/game).
Top performers have only 5 holes/game by penalizing hole creation in scoring.

Task: Modify evaluate_position() to add hole penalty. Weight it heavily.

Return only the improved Python code.
```

**Example 2: Exploration**
```
You are creating a NEW Tetris-playing approach.

Context: Current best uses greedy 2-lookahead (score: 12,345).
All top 5 programs use similar greedy approaches.

Task: Implement a beam search algorithm with width=3, depth=2.
Maintain the same TetrisPlayer interface: select_action(observation) -> action

Return only the complete Python code.
```

**Example 3: Crossover**
```
You are combining features from two Tetris programs.

Program A (score: 12,000):
{code_a}
Strength: Excellent hole management

Program B (score: 11,500):
{code_b}
Strength: Fast lookahead with caching

Task: Combine A's hole management with B's caching strategy.

Return only the combined Python code.
```

**Note**: Root has complete freedom in prompt design. No required structure, no required fields. Root decides everything based on strategic analysis.

## Success Metrics

### System Performance
- **Convergence**: Improvement in average score over generations
- **Diversity**: Variety of approaches in population
- **Efficiency**: Time per generation, cost per generation
- **Code Quality**: Syntactic correctness, runtime stability

### Tetris Performance (Target Goals)
- **Lines Cleared**: >500 lines average per game
- **Survival Time**: >5000 moves per game
- **Score**: Context-dependent, aim for top 1% of baseline algorithms

### Evolution Quality
- **Innovation**: Novel strategies discovered
- **Robustness**: Performance across different game seeds
- **Interpretability**: Understandable strategies in evolved code

## Risk Mitigation

### Code Safety
- **Sandboxing**: Execute generated code in isolated environment
- **Timeout**: Limit execution time per move and per game
- **Resource Limits**: Cap memory and CPU usage
- **Validation**: Parse and check code before execution

### Resource Control
- **Hard Limits**:
  - Max generations (e.g., 100)
  - Max RLLMs per generation (e.g., 50)
  - Max time (e.g., 4 hours)
- **Token Budgets**: Limit per generation and total
- **Model Selection**: Use cheaper models for Child LLMs when possible
- **Caching**: Reuse evaluations for identical code
- **Root Termination**: Root can terminate earlier, but cannot exceed limits

### Quality Assurance
- **Baseline Comparison**: Maintain hand-written baseline players
- **Regression Testing**: Ensure new generations don't catastrophically fail
- **Human Review**: Periodic inspection of evolved strategies
- **Checkpointing**: Save state frequently for recovery

## Extensions & Future Work

### Short-term Enhancements
1. **Multi-objective Optimization**: Balance score, style, code simplicity
2. **Transfer Learning**: Use insights from Tetris for other games
3. **Interactive Evolution**: Allow human guidance/feedback
4. **Visualization Dashboard**: Real-time evolution monitoring

### Long-term Research Directions
1. **Self-Modifying Evolution**: Root LLM modifies its own selection strategy
2. **Hierarchical Specialization**: Multi-level RLM tree (depth > 1)
3. **Meta-Learning**: Learn mutation strategies that work best
4. **Curriculum Learning**: Start with simplified Tetris, increase difficulty
5. **Ensemble Players**: Combine multiple evolved programs

## Conclusion

This system uniquely combines evolutionary coding (AlphaEvolve) with hierarchical
LLM decision-making (RLM) to evolve Tetris-playing code. The Root LLM provides
strategic intelligence for evolution, while Child LLMs provide creative code
generation. This separation of concerns leverages each LLM's strengths:

- **Root LLM**: Strategic thinking, pattern recognition, long-term planning
- **Child LLMs**: Creative coding, mutation implementation, diverse exploration

The result is an evolution system that is both more intelligent in selection
and more creative in generation than traditional evolutionary algorithms or
single-LLM approaches.
