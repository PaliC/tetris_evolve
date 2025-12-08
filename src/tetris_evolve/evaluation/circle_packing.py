"""
Circle Packing Evaluator

Evaluates circle packing programs that attempt to pack n circles into a unit square
while maximizing the sum of their radii.

Based on the evaluator from:
https://github.com/algorithmicsuperintelligence/openevolve/blob/main/examples/circle_packing/evaluator.py
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# Default target: AlphaEvolve result for n=26
DEFAULT_TARGET = 2.635
DEFAULT_N_CIRCLES = 26


@dataclass
class PackingResult:
    """Result of evaluating a circle packing program."""

    valid: bool
    sum_radii: float
    target_ratio: float
    combined_score: float
    eval_time: float
    error: str | None = None
    centers: np.ndarray | None = None
    radii: np.ndarray | None = None


def validate_packing(
    centers: np.ndarray, radii: np.ndarray, n_circles: int = DEFAULT_N_CIRCLES
) -> tuple[bool, str | None]:
    """
    Validate that circles don't overlap and are inside the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n,) with radius of each circle
        n_circles: Expected number of circles

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check shapes
    if centers.shape != (n_circles, 2):
        return False, f"Invalid centers shape: {centers.shape}, expected ({n_circles}, 2)"

    if radii.shape != (n_circles,):
        return False, f"Invalid radii shape: {radii.shape}, expected ({n_circles},)"

    # Check for NaN values
    if np.isnan(centers).any():
        return False, "NaN values in centers"

    if np.isnan(radii).any():
        return False, "NaN values in radii"

    # Check for negative radii
    for i in range(n_circles):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"

    # Check if circles are inside the unit square
    tolerance = 1e-6
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tolerance or x + r > 1 + tolerance:
            return False, f"Circle {i} at ({x}, {y}) with radius {r} extends outside x-bounds"
        if y - r < -tolerance or y + r > 1 + tolerance:
            return False, f"Circle {i} at ({x}, {y}) with radius {r} extends outside y-bounds"

    # Check for overlaps
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = radii[i] + radii[j] - tolerance
            if dist < min_dist:
                return (
                    False,
                    f"Circles {i} and {j} overlap: distance={dist:.6f}, required={min_dist:.6f}",
                )

    return True, None


def run_code_with_timeout(
    code: str,
    timeout_seconds: int = 30,
    n_circles: int = DEFAULT_N_CIRCLES,  # noqa: ARG001
    python_executable: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, str | None]:
    """
    Run circle packing code in a separate process with timeout.

    Args:
        code: Python code that defines construct_packing() or run_packing()
        timeout_seconds: Maximum execution time
        n_circles: Expected number of circles
        python_executable: Path to Python executable to use. If None, uses sys.executable.
                          Use this to run code in a custom uv environment with restricted packages.
                          Example: ".venv-restricted/bin/python"

    Returns:
        Tuple of (centers, radii, sum_radii, error_message)
    """
    # Use custom Python executable if provided, otherwise use current interpreter
    python_cmd = python_executable if python_executable else sys.executable
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as code_file:
        code_file.write(code)
        code_path = code_file.name

    results_path = f"{code_path}.results"

    # Create runner script
    runner_script = f'''
import sys
import numpy as np
import pickle
import traceback

try:
    # Execute the code
    with open("{code_path}", "r") as f:
        code = f.read()

    namespace = {{"np": np, "numpy": np}}
    exec(code, namespace)

    # Try run_packing first, then construct_packing
    if "run_packing" in namespace:
        centers, radii, sum_radii = namespace["run_packing"]()
    elif "construct_packing" in namespace:
        centers, radii, sum_radii = namespace["construct_packing"]()
    else:
        raise ValueError("Code must define run_packing() or construct_packing()")

    # Convert to numpy arrays
    centers = np.array(centers)
    radii = np.array(radii)
    sum_radii = float(sum_radii)

    results = {{
        "centers": centers,
        "radii": radii,
        "sum_radii": sum_radii,
        "error": None
    }}
except Exception as e:
    results = {{
        "centers": None,
        "radii": None,
        "sum_radii": None,
        "error": f"{{type(e).__name__}}: {{str(e)}}"
    }}

with open("{results_path}", "wb") as f:
    pickle.dump(results, f)
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as runner_file:
        runner_file.write(runner_script)
        runner_path = runner_file.name

    try:
        # Run the script with timeout using specified Python executable
        process = subprocess.Popen(
            [python_cmd, runner_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)

            if process.returncode != 0:
                error_msg = (
                    stderr.decode() if stderr else f"Process exited with code {process.returncode}"
                )
                return None, None, None, error_msg

            # Load results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                if results["error"]:
                    return None, None, None, results["error"]

                return (
                    results["centers"],
                    results["radii"],
                    results["sum_radii"],
                    None,
                )
            else:
                return None, None, None, "Results file not created"

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return None, None, None, f"Timeout after {timeout_seconds}s"

    finally:
        # Cleanup
        import contextlib

        for path in [code_path, runner_path, results_path]:
            if os.path.exists(path):
                with contextlib.suppress(Exception):
                    os.unlink(path)


def evaluate_code(
    code: str,
    target: float = DEFAULT_TARGET,
    n_circles: int = DEFAULT_N_CIRCLES,
    timeout_seconds: int = 30,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate circle packing code.

    The code must define either:
    - run_packing() -> (centers, radii, sum_radii)
    - construct_packing() -> (centers, radii, sum_radii)

    Args:
        code: Python code defining the packing function
        target: Target sum of radii (for computing ratio)
        n_circles: Expected number of circles
        timeout_seconds: Maximum execution time
        python_executable: Path to Python executable to use. If None, uses sys.executable.
                          Use this to run code in a custom uv environment with restricted packages.

    Returns:
        Dictionary with evaluation metrics:
        - valid: bool - whether the packing is geometrically valid
        - sum_radii: float - sum of all radii (0 if invalid)
        - target_ratio: float - sum_radii / target (0 if invalid)
        - combined_score: float - target_ratio if valid, else 0
        - eval_time: float - evaluation time in seconds
        - error: Optional[str] - error message if any
    """
    start_time = time.time()

    # Run the code
    centers, radii, reported_sum, error = run_code_with_timeout(
        code,
        timeout_seconds=timeout_seconds,
        n_circles=n_circles,
        python_executable=python_executable,
    )

    if error or centers is None or radii is None or reported_sum is None:
        return {
            "valid": False,
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "combined_score": 0.0,
            "eval_time": time.time() - start_time,
            "error": error or "Invalid result from code execution",
        }

    # Validate the packing
    valid, validation_error = validate_packing(centers, radii, n_circles)

    if not valid:
        return {
            "valid": False,
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "combined_score": 0.0,
            "eval_time": time.time() - start_time,
            "error": validation_error,
        }

    # Compute metrics
    sum_radii = float(np.sum(radii))
    target_ratio = sum_radii / target
    combined_score = target_ratio  # Only valid packings reach here

    # Verify reported sum matches computed sum
    if abs(sum_radii - reported_sum) > 1e-6:
        # Warning but not an error
        pass

    return {
        "valid": True,
        "sum_radii": sum_radii,
        "target_ratio": target_ratio,
        "combined_score": combined_score,
        "eval_time": time.time() - start_time,
        "error": None,
    }


class CirclePackingEvaluator:
    """
    Evaluator for circle packing programs.

    This is the main evaluator class used by the evolution system.
    """

    def __init__(
        self,
        target: float = DEFAULT_TARGET,
        n_circles: int = DEFAULT_N_CIRCLES,
        timeout_seconds: int = 30,
        python_executable: str | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            target: Target sum of radii for computing ratio
            n_circles: Number of circles to pack
            timeout_seconds: Maximum time for evaluation
            python_executable: Path to Python executable to use for running child code.
                              If None, uses the current Python interpreter (sys.executable).
                              Use this to run child LM code in a restricted uv environment.

                              Example - create a restricted environment:
                                  uv venv .venv-restricted
                                  uv pip install numpy scipy --python .venv-restricted/bin/python

                              Then set python_executable: ".venv-restricted/bin/python"
                              This ensures child code cannot import matplotlib or other packages.
        """
        self.target = target
        self.n_circles = n_circles
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable

    def evaluate(self, code: str) -> dict[str, Any]:
        """
        Evaluate circle packing code.

        Args:
            code: Python code defining run_packing() or construct_packing()

        Returns:
            Dictionary with metrics: valid, sum_radii, target_ratio, combined_score, eval_time, error
        """
        return evaluate_code(
            code,
            target=self.target,
            n_circles=self.n_circles,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
        )


# Example programs for testing
SIMPLE_GRID_PROGRAM = '''
import numpy as np

def construct_packing():
    """Simple grid-based packing."""
    n = 26
    centers = []

    # 5x5 grid = 25, plus 1 more
    grid_size = 5
    spacing = 1.0 / (grid_size + 1)

    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx < n:
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                centers.append([x, y])
                idx += 1

    # Add one more circle
    centers.append([0.5, 0.5])

    centers = np.array(centers[:n])

    # Compute radii
    radii = np.ones(n) * (spacing / 2 - 0.01)

    # Limit by boundaries
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1-x, 1-y)
        radii[i] = min(radii[i], max_r)

    # Limit by neighbors
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j]) * 0.99
                radii[i] *= scale
                radii[j] *= scale

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

def run_packing():
    return construct_packing()
'''

CONCENTRIC_RINGS_PROGRAM = '''
import numpy as np

def construct_packing():
    """Concentric rings packing - the initial program from OpenEvolve."""
    n = 26
    centers = np.zeros((n, 2))

    # Center circle
    centers[0] = [0.5, 0.5]

    # Inner ring: 8 circles
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Outer ring: 16 circles
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Clip to unit square
    centers = np.clip(centers, 0.01, 0.99)

    # One more circle in corner
    centers[25] = [0.1, 0.1]

    # Compute max radii
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii

def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by boundaries
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)

    # Limit by neighbors
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii

def run_packing():
    return construct_packing()
'''

BROKEN_PROGRAM = """
def run_packing():
    return undefined_variable
"""

NO_FUNCTION_PROGRAM = """
x = 5
"""


if __name__ == "__main__":
    # Test the evaluator
    evaluator = CirclePackingEvaluator()

    print("Testing Circle Packing Evaluator")
    print("=" * 50)

    print("\n1. Simple Grid Program:")
    result = evaluator.evaluate(SIMPLE_GRID_PROGRAM)
    print(f"   Valid: {result['valid']}")
    print(f"   Sum of radii: {result['sum_radii']:.4f}")
    print(f"   Target ratio: {result['target_ratio']:.4f}")
    print(f"   Time: {result['eval_time']:.3f}s")

    print("\n2. Concentric Rings Program:")
    result = evaluator.evaluate(CONCENTRIC_RINGS_PROGRAM)
    print(f"   Valid: {result['valid']}")
    print(f"   Sum of radii: {result['sum_radii']:.4f}")
    print(f"   Target ratio: {result['target_ratio']:.4f}")
    print(f"   Time: {result['eval_time']:.3f}s")

    print("\n3. Broken Program:")
    result = evaluator.evaluate(BROKEN_PROGRAM)
    print(f"   Valid: {result['valid']}")
    print(f"   Error: {result['error']}")

    print("\n4. No Function Program:")
    result = evaluator.evaluate(NO_FUNCTION_PROGRAM)
    print(f"   Valid: {result['valid']}")
    print(f"   Error: {result['error']}")
