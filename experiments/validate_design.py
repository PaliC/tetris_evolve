"""
Design Validation Experiments

Run these experiments to verify the design decisions are technically feasible.
Usage: python experiments/validate_design.py
"""

import io
import sys
import contextlib
import traceback
from typing import Any, Callable


def experiment_1_sandbox_execution():
    """
    Validate: Safe code execution in a sandbox environment.

    Tests:
    1. Basic code execution works
    2. Output capture works
    3. Variable persistence works
    4. Unsafe operations are blocked
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Sandbox Code Execution")
    print("="*60)

    # Define safe builtins
    SAFE_BUILTINS = {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'print': print,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'True': True,
        'False': False,
        'None': None,
    }

    class SimpleSandbox:
        def __init__(self):
            self.variables = {}

        def execute(self, code: str) -> dict:
            """Execute code and return result."""
            # Create restricted namespace with NO default builtins
            namespace = {
                '__builtins__': SAFE_BUILTINS,  # Override builtins completely
                **self.variables,
            }

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_capture):
                    with contextlib.redirect_stderr(stderr_capture):
                        exec(code, namespace)

                # Update variables (exclude builtins and special names)
                self.variables = {
                    k: v for k, v in namespace.items()
                    if k != '__builtins__' and not k.startswith('_')
                }

                return {
                    'success': True,
                    'stdout': stdout_capture.getvalue(),
                    'stderr': stderr_capture.getvalue(),
                    'variables': list(self.variables.keys())
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f"{type(e).__name__}: {str(e)}",
                    'stdout': stdout_capture.getvalue(),
                    'stderr': stderr_capture.getvalue()
                }

    sandbox = SimpleSandbox()

    # Test 1: Basic execution
    print("\nTest 1: Basic execution")
    result = sandbox.execute("x = 5 + 3")
    print(f"  Code: x = 5 + 3")
    print(f"  Result: {result}")
    assert result['success'], "Basic execution failed"
    assert 'x' in result['variables'], "Variable not persisted"
    print("  ✓ PASSED")

    # Test 2: Output capture
    print("\nTest 2: Output capture")
    result = sandbox.execute("print('Hello, sandbox!')")
    print(f"  Code: print('Hello, sandbox!')")
    print(f"  Stdout: {repr(result['stdout'])}")
    assert "Hello, sandbox!" in result['stdout'], "Output not captured"
    print("  ✓ PASSED")

    # Test 3: Variable persistence
    print("\nTest 3: Variable persistence")
    sandbox.execute("counter = 0")
    sandbox.execute("counter += 1")
    sandbox.execute("counter += 1")
    result = sandbox.execute("print(f'Counter: {counter}')")
    print(f"  After 3 increments: {result['stdout'].strip()}")
    assert "Counter: 2" in result['stdout'], "Variable persistence failed"
    print("  ✓ PASSED")

    # Test 4: Unsafe operations blocked
    print("\nTest 4: Unsafe operations blocked")
    result = sandbox.execute("import os; os.system('ls')")
    print(f"  Code: import os")
    print(f"  Error: {result.get('error', 'None')}")
    assert not result['success'], "Unsafe import should fail"
    print("  ✓ PASSED (import blocked)")

    # Test 5: Function definition
    print("\nTest 5: Function definition and calling")
    sandbox.execute("""
def select_action(observation, info):
    return 0  # Always move left
""")
    result = sandbox.execute("action = select_action(None, {}); print(f'Action: {action}')")
    print(f"  Result: {result['stdout'].strip()}")
    assert "Action: 0" in result['stdout'], "Function execution failed"
    print("  ✓ PASSED")

    print("\n✓ All sandbox tests passed!")
    return True


def experiment_2_code_extraction():
    """
    Validate: Extracting code blocks from LLM responses.

    Tests:
    1. Extract single code block
    2. Extract multiple code blocks
    3. Handle code blocks without language specifier
    4. Extract specific function (select_action)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Code Block Extraction")
    print("="*60)

    import re

    def extract_code_blocks(text: str) -> list[str]:
        """Extract all code blocks from markdown-formatted text."""
        # Pattern matches ```python or ``` followed by code
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]

    def extract_select_action(code: str) -> str | None:
        """Extract the select_action function from code."""
        # Pattern to match function definition
        pattern = r'(def select_action\s*\([^)]*\)\s*(?:->.*?)?:\s*(?:""".*?"""|\'\'\'.*?\'\'\')?\s*.*?)(?=\ndef |\Z)'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # Test 1: Single code block
    print("\nTest 1: Single code block extraction")
    response = '''
Here's my implementation:

```python
def select_action(observation, info):
    return 5  # Hard drop
```

This will quickly drop pieces.
'''
    blocks = extract_code_blocks(response)
    print(f"  Found {len(blocks)} code block(s)")
    assert len(blocks) == 1, "Should find 1 code block"
    assert "select_action" in blocks[0], "Should contain function"
    print("  ✓ PASSED")

    # Test 2: Multiple code blocks
    print("\nTest 2: Multiple code blocks")
    response = '''
First, let me check something:

```python
result = spawn_child_llm("test")
```

Now evaluate:

```python
eval_result = evaluate_program(result['code'])
print(eval_result)
```
'''
    blocks = extract_code_blocks(response)
    print(f"  Found {len(blocks)} code block(s)")
    assert len(blocks) == 2, "Should find 2 code blocks"
    print("  ✓ PASSED")

    # Test 3: Code block without language specifier
    print("\nTest 3: Code block without language specifier")
    response = '''
Execute this:

```
x = 42
print(x)
```
'''
    blocks = extract_code_blocks(response)
    print(f"  Found {len(blocks)} code block(s)")
    assert len(blocks) == 1, "Should find block without language"
    print("  ✓ PASSED")

    # Test 4: Extract select_action function
    print("\nTest 4: Extract select_action function")
    code = '''
import numpy as np

def helper(x):
    return x * 2

def select_action(observation, info):
    """Select the best action."""
    # Check if we should move left
    if info.get('current_piece') == 1:
        return 0
    return 5

def another_function():
    pass
'''
    func = extract_select_action(code)
    print(f"  Extracted function length: {len(func) if func else 0} chars")
    assert func is not None, "Should extract function"
    assert "def select_action" in func, "Should contain definition"
    assert "another_function" not in func, "Should not include next function"
    print("  ✓ PASSED")

    print("\n✓ All code extraction tests passed!")
    return True


def experiment_3_repl_function_injection():
    """
    Validate: Injecting custom functions into REPL namespace.

    Tests:
    1. Functions can be injected
    2. Functions can access external state
    3. Functions can return structured data
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: REPL Function Injection")
    print("="*60)

    # Simulated external state
    state = {
        'trials': [],
        'generation': 0,
        'terminated': False
    }

    # Define REPL functions that modify state
    def spawn_child_llm(prompt: str, parent_id: str = None) -> dict:
        """Spawn a child LLM (mock)."""
        trial_id = f"trial_{len(state['trials']):03d}"
        trial = {
            'trial_id': trial_id,
            'prompt': prompt,
            'parent_id': parent_id,
            'code': f"def select_action(obs, info): return {len(state['trials']) % 7}",
            'success': True
        }
        state['trials'].append(trial)
        return trial

    def evaluate_program(code: str, num_games: int = 10) -> dict:
        """Evaluate program (mock)."""
        return {
            'success': True,
            'metrics': {
                'mean_score': 100 + len(code),
                'max_score': 200 + len(code)
            }
        }

    def advance_generation(selected_ids: list, reasoning: str) -> int:
        """Advance to next generation."""
        state['generation'] += 1
        return state['generation']

    def terminate_evolution(reason: str) -> dict:
        """Terminate the evolution."""
        state['terminated'] = True
        return {
            'total_generations': state['generation'],
            'total_trials': len(state['trials']),
            'reason': reason
        }

    # Create REPL with injected functions
    repl_functions = {
        'spawn_child_llm': spawn_child_llm,
        'evaluate_program': evaluate_program,
        'advance_generation': advance_generation,
        'terminate_evolution': terminate_evolution,
    }

    SAFE_BUILTINS = {
        'print': print, 'len': len, 'range': range,
        'list': list, 'dict': dict, 'str': str,
        'True': True, 'False': False, 'None': None,
    }

    class REPLWithFunctions:
        def __init__(self, functions):
            self.functions = functions
            self.variables = {}

        def execute(self, code: str) -> dict:
            namespace = {
                **SAFE_BUILTINS,
                **self.functions,
                **self.variables,
            }

            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(code, namespace)

                self.variables = {
                    k: v for k, v in namespace.items()
                    if k not in SAFE_BUILTINS
                    and k not in self.functions
                    and not k.startswith('_')
                }

                return {
                    'success': True,
                    'stdout': stdout.getvalue(),
                    'terminated': state['terminated']
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'stdout': stdout.getvalue()
                }

    repl = REPLWithFunctions(repl_functions)

    # Test 1: Spawn child LLM
    print("\nTest 1: Spawn child LLM")
    result = repl.execute("""
trial1 = spawn_child_llm("Create a greedy Tetris agent")
print(f"Spawned: {trial1['trial_id']}")
""")
    print(f"  Output: {result['stdout'].strip()}")
    assert result['success'], "Spawn should succeed"
    assert len(state['trials']) == 1, "Trial should be recorded"
    print("  ✓ PASSED")

    # Test 2: Evaluate program
    print("\nTest 2: Evaluate program")
    result = repl.execute("""
metrics = evaluate_program(trial1['code'], num_games=5)
print(f"Mean score: {metrics['metrics']['mean_score']}")
""")
    print(f"  Output: {result['stdout'].strip()}")
    assert result['success'], "Evaluate should succeed"
    print("  ✓ PASSED")

    # Test 3: Multiple trials and generation advance
    print("\nTest 3: Multiple trials and generation advance")
    result = repl.execute("""
trial2 = spawn_child_llm("Create a random agent")
trial3 = spawn_child_llm("Create a heuristic agent")
new_gen = advance_generation([trial1['trial_id'], trial2['trial_id']], "Best performers")
print(f"Advanced to generation {new_gen}")
""")
    print(f"  Output: {result['stdout'].strip()}")
    assert state['generation'] == 1, "Generation should advance"
    print("  ✓ PASSED")

    # Test 4: Terminate evolution
    print("\nTest 4: Terminate evolution")
    result = repl.execute("""
summary = terminate_evolution("Testing complete")
print(f"Total trials: {summary['total_trials']}")
""")
    print(f"  Output: {result['stdout'].strip()}")
    assert result['terminated'], "Should be terminated"
    assert state['terminated'], "State should reflect termination"
    print("  ✓ PASSED")

    print("\n✓ All REPL function injection tests passed!")
    return True


def experiment_4_tetris_gymnasium():
    """
    Validate: Tetris-Gymnasium integration.

    Tests:
    1. Environment creation
    2. Basic game loop
    3. Custom agent integration
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Tetris-Gymnasium Integration")
    print("="*60)

    try:
        import gymnasium as gym
    except ImportError:
        print("  ⚠ gymnasium not installed, skipping")
        print("  Install with: pip install gymnasium")
        return None

    try:
        # Try to import tetris_gymnasium
        import tetris_gymnasium
        HAS_TETRIS = True
    except ImportError:
        print("  ⚠ tetris_gymnasium not installed")
        print("  Install with: pip install tetris-gymnasium")
        HAS_TETRIS = False

    if not HAS_TETRIS:
        print("\n  Testing with CartPole as proxy environment...")

        # Test 1: Environment creation
        print("\nTest 1: Environment creation (CartPole proxy)")
        env = gym.make("CartPole-v1")
        print(f"  Created environment")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print("  ✓ PASSED")

        # Test 2: Basic game loop
        print("\nTest 2: Basic game loop")
        obs, info = env.reset(seed=42)
        print(f"  Initial observation shape: {obs.shape}")
        total_reward = 0
        steps = 0
        terminated = False
        while not terminated and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"  Game lasted {steps} steps, total reward: {total_reward}")
        env.close()
        print("  ✓ PASSED")

        # Test 3: Custom agent
        print("\nTest 3: Custom agent integration")

        agent_code = '''
def select_action(observation, info):
    # Simple heuristic: go left if pole leaning left
    if observation[2] < 0:
        return 0
    return 1
'''
        namespace = {}
        exec(agent_code, namespace)
        agent = namespace['select_action']

        env = gym.make("CartPole-v1")
        obs, info = env.reset(seed=42)
        total_reward = 0
        steps = 0
        terminated = False
        while not terminated and steps < 200:
            action = agent(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"  Custom agent: {steps} steps, reward: {total_reward}")
        env.close()
        print("  ✓ PASSED")

        print("\n✓ Proxy environment tests passed!")
        print("  Note: Install tetris-gymnasium for actual Tetris testing")
        return True

    # Actual Tetris tests
    print("\nTest 1: Tetris environment creation")
    env = gym.make("tetris_gymnasium/Tetris")
    print(f"  Created Tetris environment")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("  ✓ PASSED")

    print("\nTest 2: Basic Tetris game loop")
    obs, info = env.reset(seed=42)
    print(f"  Initial observation shape: {obs.shape}")
    total_reward = 0
    steps = 0
    terminated = False
    while not terminated and steps < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    print(f"  Game lasted {steps} steps")
    env.close()
    print("  ✓ PASSED")

    print("\nTest 3: Custom Tetris agent")
    agent_code = '''
def select_action(observation, info):
    # Simple strategy: prefer hard drops
    import random
    if random.random() < 0.3:
        return 5  # hard_drop
    return random.randint(0, 6)
'''
    namespace = {'__builtins__': __builtins__}
    exec(agent_code, namespace)
    agent = namespace['select_action']

    env = gym.make("tetris_gymnasium/Tetris")
    obs, info = env.reset(seed=42)
    total_reward = 0
    steps = 0
    terminated = False
    while not terminated and steps < 500:
        action = agent(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    print(f"  Custom agent: {steps} steps, reward: {total_reward}")
    env.close()
    print("  ✓ PASSED")

    print("\n✓ All Tetris-Gymnasium tests passed!")
    return True


def experiment_5_llm_client_mock():
    """
    Validate: LLM client structure and response parsing.

    Tests:
    1. Response structure
    2. Token usage tracking
    3. Cost calculation
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: LLM Client Structure")
    print("="*60)

    from dataclasses import dataclass

    @dataclass
    class TokenUsage:
        input_tokens: int
        output_tokens: int

    @dataclass
    class LLMResponse:
        content: str
        usage: TokenUsage

    class MockLLMClient:
        """Mock LLM client for testing."""

        def __init__(self, model: str, responses: list[str] = None):
            self.model = model
            self.responses = responses or []
            self.call_count = 0

        def chat(self, messages: list[dict]) -> tuple[LLMResponse, TokenUsage]:
            """Simulate chat completion."""
            # Calculate input tokens (rough estimate)
            input_text = " ".join(m.get('content', '') for m in messages)
            input_tokens = len(input_text.split()) * 1.3  # Rough token estimate

            # Get response
            if self.call_count < len(self.responses):
                content = self.responses[self.call_count]
            else:
                content = "```python\nresult = 'default'\n```"

            output_tokens = len(content.split()) * 1.3
            self.call_count += 1

            usage = TokenUsage(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens)
            )

            return LLMResponse(content=content, usage=usage), usage

    # Test 1: Basic response
    print("\nTest 1: Basic response structure")
    client = MockLLMClient(
        model="test-model",
        responses=["Here's my answer:\n```python\nx = 42\n```"]
    )
    response, usage = client.chat([{"role": "user", "content": "Write code"}])
    print(f"  Response content length: {len(response.content)}")
    print(f"  Input tokens: {usage.input_tokens}")
    print(f"  Output tokens: {usage.output_tokens}")
    assert response.content, "Should have content"
    assert usage.input_tokens > 0, "Should have input tokens"
    print("  ✓ PASSED")

    # Test 2: Cost calculation
    print("\nTest 2: Cost calculation")

    def calculate_cost(usage: TokenUsage, input_cost: float, output_cost: float) -> float:
        """Calculate cost in dollars."""
        return (usage.input_tokens / 1000 * input_cost +
                usage.output_tokens / 1000 * output_cost)

    cost = calculate_cost(usage, input_cost=0.003, output_cost=0.015)
    print(f"  Cost for {usage.input_tokens} in + {usage.output_tokens} out: ${cost:.6f}")
    assert cost > 0, "Cost should be positive"
    print("  ✓ PASSED")

    # Test 3: Multi-turn conversation
    print("\nTest 3: Multi-turn conversation")
    client = MockLLMClient(
        model="test-model",
        responses=[
            "```python\ntrial = spawn_child_llm('test')\n```",
            "```python\nresult = evaluate_program(trial['code'])\n```",
            "```python\nterminate_evolution('done')\n```"
        ]
    )

    total_cost = 0
    messages = [{"role": "user", "content": "Start evolution"}]

    for i in range(3):
        response, usage = client.chat(messages)
        cost = calculate_cost(usage, 0.003, 0.015)
        total_cost += cost
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": f"Result: success"})

    print(f"  Total cost over 3 turns: ${total_cost:.6f}")
    assert client.call_count == 3, "Should have 3 calls"
    print("  ✓ PASSED")

    print("\n✓ All LLM client tests passed!")
    return True


def run_all_experiments():
    """Run all validation experiments."""
    print("="*60)
    print("DESIGN VALIDATION EXPERIMENTS")
    print("="*60)
    print("\nRunning experiments to validate design feasibility...")

    results = {}

    experiments = [
        ("Sandbox Execution", experiment_1_sandbox_execution),
        ("Code Extraction", experiment_2_code_extraction),
        ("REPL Function Injection", experiment_3_repl_function_injection),
        ("Tetris Integration", experiment_4_tetris_gymnasium),
        ("LLM Client Mock", experiment_5_llm_client_mock),
    ]

    for name, func in experiments:
        try:
            result = func()
            results[name] = "PASSED" if result else "SKIPPED"
        except Exception as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results[name] = "FAILED"

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results.items():
        symbol = "✓" if status == "PASSED" else ("⚠" if status == "SKIPPED" else "✗")
        print(f"  {symbol} {name}: {status}")

    passed = sum(1 for s in results.values() if s == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")

    if all(s in ("PASSED", "SKIPPED") for s in results.values()):
        print("\n✓ Design validation successful!")
        return True
    else:
        print("\n✗ Some experiments failed - review design")
        return False


if __name__ == "__main__":
    success = run_all_experiments()
    sys.exit(0 if success else 1)
