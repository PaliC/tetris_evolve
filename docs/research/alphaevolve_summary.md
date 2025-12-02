# AlphaEvolve Technical Summary

## Overview

AlphaEvolve is an evolutionary coding agent developed by Google DeepMind that combines large language models with evolutionary algorithms to discover novel algorithms. It extends FunSearch by evolving entire codebases rather than single functions.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AlphaEvolve System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐        │
│  │   Prompt    │ ───► │     LLM     │ ───► │  Evaluator  │        │
│  │   Sampler   │      │  Ensemble   │      │    Pool     │        │
│  └──────▲──────┘      └─────────────┘      └──────┬──────┘        │
│         │                                          │               │
│         │         ┌─────────────────┐              │               │
│         └─────────│    Program      │◄─────────────┘               │
│                   │    Database     │                              │
│                   └─────────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Prompt Sampler

**Purpose**: Constructs rich prompts by combining:
- Problem description and constraints
- Previous high-performing programs from the database
- Evaluation results and metrics
- System instructions and meta-prompts

**Key Features**:
- Stochastic formatting with externally defined distributions
- Customizable prompt templates
- Can include evolutionary history for context
- Supports meta-prompt evolution (the prompt itself can evolve)

**Sampling Strategy**:
- Balances exploitation (sampling best programs) with exploration (diverse programs)
- Uses temperature-based selection from program database
- Can sample from different islands for diversity

### 2. LLM Ensemble

**Purpose**: Generates code modifications (diffs) to create new program variants.

**Architecture**:
- **Fast Model (Gemini Flash)**: High throughput, lower quality
  - Maximizes idea exploration per unit time
  - Handles breadth of search
- **Quality Model (Gemini Pro)**: Lower throughput, higher quality
  - Provides breakthrough suggestions
  - Handles depth of search

**Output Format**:
- Produces code diffs rather than full programs
- Specifies line replacements: `{old_code} → {new_code}`
- Enables handling of large codebases efficiently

### 3. Program Database

**Purpose**: Stores discovered solutions with quality assessments, managing exploration vs. exploitation.

**Algorithm**: Hybrid of MAP-Elites and Island Model

**MAP-Elites Integration**:
- Programs are characterized by behavioral descriptors (features)
- Database maintains a grid of cells, each holding best program for that feature combination
- Encourages diversity by preserving programs that are "best at something"

**Island Model**:
```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Island  │   │ Island  │   │ Island  │   │ Island  │   │ Island  │
│    1    │◄─►│    2    │◄─►│    3    │◄─►│    4    │◄─►│    5    │
│(pop: N) │   │(pop: N) │   │(pop: N) │   │(pop: N) │   │(pop: N) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     │             │             │             │             │
     └─────────────┴──────┬──────┴─────────────┴─────────────┘
                          │
                    Migration Events
                    (periodic exchange)
```

- Multiple subpopulations evolve independently
- Periodic migration exchanges individuals between islands
- Prevents premature convergence
- Maintains genetic diversity

**Database Operations**:
1. **Insert**: Add new program with evaluation metrics
2. **Sample**: Select programs for prompt construction
3. **Prune**: Remove low-performing programs to maintain size
4. **Migrate**: Exchange programs between islands

### 4. Evaluator Pool

**Purpose**: Tests generated programs and assigns quality scores.

**Characteristics**:
- Automated evaluation metrics
- Robust verification for correctness
- Returns structured metric dictionaries
- Supports parallel evaluation

**Evaluation Types**:
- Functional correctness (does it run?)
- Performance metrics (how well does it work?)
- Efficiency metrics (time/space complexity)
- Custom domain-specific metrics

## Evolutionary Loop

```python
# Pseudocode for AlphaEvolve main loop
def evolve(initial_program, evaluator, config):
    database = ProgramDatabase(config)
    database.insert(initial_program, evaluator(initial_program))

    for generation in range(config.max_generations):
        # 1. Sample programs for prompt
        parent_programs = database.sample(
            num_parents=config.num_parents,
            strategy=config.sampling_strategy
        )

        # 2. Construct prompt
        prompt = prompt_sampler.construct(
            problem_description=config.problem,
            parent_programs=parent_programs,
            evaluation_history=database.get_history()
        )

        # 3. Generate mutations via LLM ensemble
        for model in llm_ensemble:
            diff = model.generate(prompt)
            new_program = apply_diff(parent_programs[0], diff)

            # 4. Evaluate new program
            if new_program.is_valid():
                metrics = evaluator(new_program)

                # 5. Insert into database
                database.insert(new_program, metrics)

        # 6. Periodic migration between islands
        if generation % config.migration_interval == 0:
            database.migrate()

    return database.get_best()
```

## Code Evolution Markers

Programs mark sections for modification:

```python
def algorithm(input):
    # Fixed preprocessing
    data = preprocess(input)

    # EVOLVE-BLOCK-START
    # This section will be modified by LLM
    result = default_algorithm(data)
    # EVOLVE-BLOCK-END

    # Fixed postprocessing
    return postprocess(result)
```

## Key Innovations

1. **Full Codebase Evolution**: Evolves entire programs, not just single functions
2. **Diff-Based Modification**: LLMs output changes, not complete programs
3. **Ensemble Diversity**: Combines fast exploration with deep reasoning
4. **Hybrid Database**: MAP-Elites + Island model for diversity maintenance
5. **Meta-Prompt Evolution**: The prompts themselves can evolve

## Relevance to Our Project

For our LLM Tetris optimizer, we adapt:

| AlphaEvolve Concept | Our Adaptation |
|---------------------|----------------|
| Program Database | Trial storage with generation hierarchy |
| LLM Ensemble | Root LLM + Child LLMs with cost tracking |
| Prompt Sampler | Root LLM's context from previous generations |
| Evaluator | PufferLib Tetris game evaluation |
| Island Model | Replaced by RLM-style selection |

**Key Difference**: Instead of algorithmic selection (MAP-Elites), we use an LLM (Root LM) to decide which trials advance to the next generation, combining evolutionary structure with LLM reasoning.

## References

- [AlphaEvolve Blog Post](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [AlphaEvolve Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
- [OpenEvolve Implementation](https://huggingface.co/blog/codelion/openevolve)
