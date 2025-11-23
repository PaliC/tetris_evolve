# LLM Tetris AlphaEvolve Design Document

## Executive Summary

This system combines **AlphaEvolve's evolutionary coding approach** with **Recursive LLM (RLM) hierarchical decision-making** to evolve code that plays Tetris optimally. The Root LLM acts as an evolutionary strategist, selecting promising candidates and guiding evolution, while Recursive Child LLMs generate code mutations and implement new variants.

## Core Concepts

### 1. AlphaEvolve Integration
- **Evolutionary Framework**: Maintain a program database with generations of Tetris-playing algorithms
- **Automated Evaluation**: Each candidate is scored by running Tetris simulations (lines cleared, survival time, score)
- **Iterative Improvement**: High-performing programs inform future generations
- **Dual Strategy**: Use both exploration (many diverse ideas) and exploitation (refine best candidates)

### 2. Recursive LLM Integration
- **Root LLM (Depth=0)**: Strategic decision-maker that:
  - Analyzes performance data across all candidates in current generation
  - Selects which programs advance to next generation
  - Decides mutation strategies (e.g., "improve hole management", "optimize piece placement scoring")
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
│  • Analyze generation performance metrics               │
│  • Select top K candidates for next generation          │
│  • Identify improvement strategies                      │
│  • Craft mutation directives for Child LLMs             │
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
│  Child 1         Child 2         Child N                │
│  Mutate top #1   Mutate top #2   Explore new approach  │
│  Focus: holes    Focus: speed    Focus: lookahead      │
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

#### 2. Root LLM Environment (REPL-based)

The Root LLM operates in a Python REPL environment with access to:

```python
# Available in Root LLM context
current_generation = 5
program_database = [...] # All programs from all generations
current_gen_programs = [...] # Programs from current generation only
metrics_summary = {...} # Statistical analysis of current generation

# Functions available to Root LLM
def spawn_child_llm(parent_program, mutation_directive, context):
    """Spawn a Child LLM to generate new code variant"""
    pass

def evaluate_program(code, num_games=100):
    """Run Tetris simulations and return metrics"""
    pass

def get_performance_analysis(generation):
    """Get detailed analysis of a generation's performance"""
    pass

def advance_generation(selected_programs, mutation_strategies):
    """Move to next generation with selected programs"""
    pass
```

The Root LLM can:
- Inspect performance data
- Identify patterns in successful programs
- Decide selection criteria dynamically (not hardcoded)
- Craft contextual mutation directives
- Spawn multiple Child LLMs in parallel

#### 3. Child LLM Interface

Each Child LLM receives:

```python
{
  "task": "generate_variant",
  "parent_program": {
    "code": str,
    "metrics": {...},
    "generation": int
  },
  "mutation_directive": {
    "strategy": "improve_hole_management",  # or other strategy
    "guidance": "The parent program leaves too many holes. Focus on...",
    "constraints": ["maintain_performance", "keep_structure"],
    "examples": [...]  # Optional: similar successful mutations
  },
  "context": {
    "top_performers": [...],  # For reference
    "common_patterns": [...],
    "generation_insights": str
  }
}
```

Child LLM outputs:
```python
{
  "code": str,
  "explanation": str,
  "expected_improvements": [str]
}
```

## Evolutionary Strategy

### Generation Lifecycle

1. **Initialization (Generation 0)**
   - Root LLM generates initial diverse population of Tetris players
   - Strategies: random placement, greedy scoring, hole minimization, etc.
   - Population size: 20-50 programs

2. **Evaluation Phase**
   - Run each program through N Tetris games (e.g., 100 games)
   - Collect comprehensive metrics
   - Store in Program Database

3. **Selection Phase (Root LLM Decision)**
   - Root LLM analyzes all metrics
   - Identifies top K performers (e.g., top 10)
   - Discovers patterns in successful programs
   - Decides selection criteria (may vary by generation)
   - Example Root reasoning:
     ```
     "Generation 5 shows that programs focusing on minimizing
     holes outperform greedy scorers by 40%. The top 3 programs
     all use lookahead depth of 2. I'll select the top 10
     performers and direct mutations to explore deeper lookahead
     while maintaining hole management strategies."
     ```

4. **Mutation Phase (Child LLMs)**
   - Root spawns Child LLMs with specific directives:
     - **Exploitation**: Mutate top performers with targeted improvements
     - **Exploration**: Create novel variants with different approaches
     - **Crossover**: Combine features from multiple top performers
   - Each Child LLM generates 1-3 variants
   - Parallel generation for efficiency

5. **Population Assembly**
   - Root collects all Child outputs
   - Optionally filters obviously broken code
   - Combines: selected parents + new children
   - Population maintained at ~20-50 programs

6. **Iteration**
   - Increment generation counter
   - Return to Evaluation Phase
   - Continue until convergence or iteration limit

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

## Tetris Environment

### Game Interface

```python
class TetrisGame:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        """Reset game state"""
        pass

    def get_state(self):
        """Return current game state"""
        return {
            "board": np.array,  # 2D grid
            "current_piece": Piece,
            "next_pieces": [Piece],  # lookahead
            "hold_piece": Piece | None,
            "score": int,
            "lines_cleared": int
        }

    def get_valid_actions(self):
        """Return all valid placements for current piece"""
        return [Action]  # rotation + column position

    def step(self, action):
        """Execute action, return new_state, reward, done"""
        pass
```

### Player Interface (What Generated Code Implements)

```python
class TetrisPlayer:
    def select_action(self, game_state):
        """
        Given current game state, return best action.

        Args:
            game_state: dict with board, current_piece, next_pieces, etc.

        Returns:
            action: dict with rotation and column
        """
        # Generated code implements this logic
        pass
```

### Evaluation Metrics

```python
def evaluate_player(player_code, num_games=100):
    """
    Run player through multiple games and collect metrics.

    Returns:
        {
            "avg_score": float,
            "avg_lines_cleared": float,
            "avg_survival_time": float,
            "max_score": int,
            "max_lines": int,
            "std_dev_score": float,
            "std_dev_lines": float,
            "success_rate": float,  # games lasting > threshold
            "code_errors": int  # runtime errors
        }
    """
    pass
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Implement Tetris game engine
- [ ] Implement evaluation framework
- [ ] Create program database (SQLite or JSON)
- [ ] Build basic RLM framework (REPL environment)
- [ ] Test with manual code submissions

### Phase 2: Root LLM Integration (Week 2-3)
- [ ] Implement Root LLM REPL environment
- [ ] Add functions for program analysis
- [ ] Build generation management system
- [ ] Create prompt templates for Root decision-making
- [ ] Test Root's ability to analyze and select programs

### Phase 3: Child LLM Integration (Week 3-4)
- [ ] Implement Child LLM spawning
- [ ] Create mutation directive templates
- [ ] Build code generation pipeline
- [ ] Add code validation and safety checks
- [ ] Test parallel Child LLM generation

### Phase 4: Evolution Loop (Week 4-5)
- [ ] Connect all components
- [ ] Implement full generation lifecycle
- [ ] Add logging and visualization
- [ ] Create checkpoint/resume functionality
- [ ] Run initial evolution experiments

### Phase 5: Optimization & Scaling (Week 5-6)
- [ ] Parallelize game evaluation
- [ ] Optimize LLM token usage
- [ ] Add caching for repeated evaluations
- [ ] Implement diversity preservation mechanisms
- [ ] Tune population size and mutation rates

## Technical Stack

### Core Components
- **Language**: Python 3.10+
- **LLM API**: OpenAI API (GPT-4 for Root, GPT-3.5/4 for Children) or Anthropic Claude
- **Game Engine**: NumPy for board state, custom Tetris implementation
- **Database**: SQLite for program storage
- **REPL**: Python `exec()` based sandbox (from RLM framework)

### Libraries
```
- numpy: Game state management
- openai / anthropic: LLM API calls
- sqlite3: Program database
- multiprocessing: Parallel evaluation
- matplotlib/plotly: Visualization
- pytest: Testing
- pydantic: Data validation
```

## Configuration

### System Parameters
```yaml
evolution:
  population_size: 30
  num_generations: 100
  games_per_evaluation: 100
  top_k_selection: 10

  mutation_distribution:
    exploitation: 0.6  # Improve top performers
    exploration: 0.3   # Novel approaches
    crossover: 0.1     # Recombination

llm:
  root_model: "gpt-4"
  child_model: "gpt-4"  # or "gpt-3.5-turbo" for cost
  temperature_root: 0.7
  temperature_child: 0.8
  max_tokens: 4000

tetris:
  board_width: 10
  board_height: 20
  preview_pieces: 5  # Number of next pieces visible
  enable_hold: true

evaluation:
  num_games: 100
  parallel_games: 10
  timeout_per_game: 300  # seconds
  max_moves_per_game: 10000
```

## Root LLM Prompting Strategy

### Initial Prompt Template
```
You are the Root LLM in an evolutionary system designed to evolve optimal
Tetris-playing code. You have access to a program database containing all
programs from previous generations, their performance metrics, and metadata.

Your responsibilities:
1. Analyze the current generation's performance
2. Select the best programs to advance
3. Identify patterns in successful programs
4. Design mutation strategies for the next generation
5. Spawn Child LLMs with specific directives to generate new code

Current Generation: {generation}

Available Functions:
- spawn_child_llm(parent_program, mutation_directive, context)
- evaluate_program(code, num_games)
- get_performance_analysis(generation)
- advance_generation(selected_programs, strategies)

Current Generation Data:
{program_data}

Performance Summary:
{metrics_summary}

Please analyze this generation and decide:
1. Which programs should advance?
2. What mutation strategies should be applied?
3. How many new variants to create?

Think step-by-step and use the available functions to implement your strategy.
```

### Child LLM Prompt Template
```
You are a Child LLM specialized in generating Tetris-playing code. You have
been given a parent program and specific mutation directive from the Root LLM.

Parent Program (Generation {gen}, Score: {score}):
{parent_code}

Parent Performance:
{parent_metrics}

Mutation Directive:
Strategy: {strategy}
Guidance: {guidance}
Constraints: {constraints}

Context:
{context}

Your task:
Generate an improved version of the parent program following the mutation
directive. Focus on the specific improvements requested while maintaining
code quality and correctness.

Your code must implement the TetrisPlayer interface:
- select_action(game_state) -> action

Return your generated code and a brief explanation of changes.
```

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

### Cost Control
- **Token Budgets**: Limit per generation and total
- **Model Selection**: Use cheaper models for Child LLMs when possible
- **Caching**: Reuse evaluations for identical code
- **Early Stopping**: Halt evolution if no improvement after N generations

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
