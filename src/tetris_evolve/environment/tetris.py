"""
Tetris environment configuration.

This module implements the EnvironmentConfig interface for Tetris,
providing Tetris-specific configuration, descriptions, and metric extraction.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym

from .base import EnvironmentConfig


class TetrisConfig(EnvironmentConfig):
    """
    Tetris-specific environment configuration.

    Provides configuration for the custom Tetris environment implemented
    in tetris_env.py, including observation/action descriptions for LLM
    prompts and metric extraction.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Tetris configuration.

        Args:
            width: Board width (default 10)
            height: Board height (default 20)
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.width = width
        self.height = height
        self.render_mode = render_mode

    def get_env_id(self) -> str:
        """Return the Tetris environment identifier."""
        return "TetrisEvolve-v0"

    def get_env_kwargs(self) -> Dict[str, Any]:
        """Return Tetris environment parameters."""
        kwargs = {
            "width": self.width,
            "height": self.height,
        }
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode
        return kwargs

    def get_observation_description(self) -> str:
        """Return detailed observation description for LLM."""
        obs_size = (self.height * self.width) + (4 * 4) + (4 * 4) + 2
        return f"""Observation is a flat numpy array of {obs_size} float32 values:

1. Board State ({self.height}x{self.width} = {self.height * self.width} values):
   - Flattened row-by-row from top to bottom
   - 0.0 = empty cell, 1.0 = filled cell
   - Index 0 is top-left, index {self.width - 1} is top-right

2. Current Piece (4x4 = 16 values):
   - 4x4 binary matrix representing the current tetromino
   - Piece types: I, O, T, S, Z, J, L
   - 1.0 = part of piece, 0.0 = empty

3. Next Piece (4x4 = 16 values):
   - 4x4 binary matrix representing the upcoming tetromino
   - Same format as current piece

4. Current Position (2 values):
   - position[0] = piece_x / board_width (normalized x position)
   - position[1] = piece_y / board_height (normalized y position)

To decode the board: board = obs[0:{self.height * self.width}].reshape({self.height}, {self.width})
"""

    def get_action_description(self) -> str:
        """Return detailed action description for LLM."""
        return """Actions are discrete integers from 0 to 5:

0: Move Left
   - Shifts the current piece one column to the left
   - No effect if blocked by wall or existing pieces

1: Move Right
   - Shifts the current piece one column to the right
   - No effect if blocked by wall or existing pieces

2: Rotate Clockwise
   - Rotates the piece 90 degrees clockwise
   - Includes wall kick logic (shifts piece if rotation blocked)

3: Rotate Counter-Clockwise
   - Rotates the piece 90 degrees counter-clockwise
   - Includes wall kick logic

4: Soft Drop
   - Moves the piece down one row immediately
   - Locks piece if it cannot move down further
   - Small reward bonus (+0.01)

5: Hard Drop
   - Instantly drops the piece to the lowest valid position
   - Immediately locks the piece
   - Reward bonus based on drop distance (+0.02 per row)

Note: Gravity automatically drops the piece every 30 steps.
"""

    def get_reward_description(self) -> str:
        """Return reward structure description for LLM."""
        return """Reward structure:

1. Line Clear Scoring (standard Tetris):
   - 1 line:  +40 points
   - 2 lines: +100 points
   - 3 lines: +300 points
   - 4 lines (Tetris): +1200 points

2. Drop Rewards:
   - Soft drop: +0.01 per drop
   - Hard drop: +0.02 per row dropped

3. Survival Reward:
   - +0.1 * current_score per step (encourages building score)

4. Game Over:
   - Episode ends when new piece cannot spawn
   - No explicit penalty, but episode terminates

Goals for optimization:
- Maximize lines cleared (especially Tetrises)
- Minimize holes (empty cells below filled cells)
- Keep the stack low
- Survive as long as possible
"""

    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """Extract Tetris-specific metrics from episode info."""
        return {
            "score": float(info.get("score", 0)),
            "lines_cleared": float(info.get("lines_cleared", 0)),
            "pieces_placed": float(info.get("pieces_placed", 0)),
        }

    def get_player_interface_template(self) -> str:
        """Return the player interface template for evolved code."""
        return '''"""
Tetris Player - Evolved Agent

This class implements a Tetris-playing agent. The LLM should modify the
code within EVOLVE-BLOCK markers to improve performance.
"""
import numpy as np
from typing import Dict, Any


class TetrisPlayer:
    """
    Tetris player that selects actions based on the current game state.

    The player receives observations and must return an action (0-5).
    """

    def __init__(self):
        """Initialize the player. Can store state between moves."""
        self.board_width = 10
        self.board_height = 20
        self.move_count = 0

    def select_action(self, observation: np.ndarray) -> int:
        """
        Select an action given the current observation.

        Args:
            observation: Flat numpy array with:
                - Board state (200 values)
                - Current piece (16 values)
                - Next piece (16 values)
                - Position (2 values, normalized)

        Returns:
            Action integer (0-5):
                0=left, 1=right, 2=rotate_cw, 3=rotate_ccw, 4=soft_drop, 5=hard_drop
        """
        self.move_count += 1

        # Decode observation
        board = observation[:200].reshape(20, 10)
        current_piece = observation[200:216].reshape(4, 4)
        next_piece = observation[216:232].reshape(4, 4)
        pos_x = observation[232] * 10  # Denormalize
        pos_y = observation[233] * 20  # Denormalize

        # EVOLVE-BLOCK-START: decision_logic
        # This is the main decision logic to be evolved
        # Default: simple heuristic-based action selection

        # Calculate board statistics
        heights = self._get_column_heights(board)
        holes = self._count_holes(board)
        max_height = max(heights) if heights else 0

        # Simple decision tree (to be improved by evolution)
        if max_height > 15:
            # Emergency: drop pieces quickly
            return 5  # Hard drop
        elif holes > 5:
            # Try to fill holes by moving to emptier areas
            left_height = sum(heights[:5])
            right_height = sum(heights[5:])
            if left_height < right_height:
                return 0  # Move left
            else:
                return 1  # Move right
        else:
            # Default: hard drop
            return 5
        # EVOLVE-BLOCK-END: decision_logic

    def _get_column_heights(self, board: np.ndarray) -> list:
        """Get the height of each column."""
        heights = []
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col]:
                    heights.append(board.shape[0] - row)
                    break
            else:
                heights.append(0)
        return heights

    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes (empty cells with filled cells above)."""
        holes = 0
        for col in range(board.shape[1]):
            found_block = False
            for row in range(board.shape[0]):
                if board[row, col]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def reset(self):
        """Reset player state for a new game."""
        self.move_count = 0
'''

    def create_env(self, **override_kwargs) -> gym.Env:
        """
        Create a Tetris environment instance.

        This method imports and creates the custom TetrisEnv directly,
        since it may not be registered with Gymnasium.
        """
        kwargs = self.get_env_kwargs().copy()
        kwargs.update(override_kwargs)

        # Import the custom Tetris environment
        # First try from the package, then from root
        try:
            from tetris_env import TetrisEnv
        except ImportError:
            # Try adding root directory to path
            root_dir = Path(__file__).parent.parent.parent.parent.parent
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))
            from tetris_env import TetrisEnv

        return TetrisEnv(**kwargs)

    def get_metrics_spec(self) -> Dict[str, Dict[str, Any]]:
        """Return specification for Tetris metrics."""
        return {
            "score": {
                "type": "float",
                "description": "Tetris game score based on lines cleared",
                "aggregation": "mean",
                "higher_is_better": True,
            },
            "lines_cleared": {
                "type": "float",
                "description": "Total number of lines cleared",
                "aggregation": "mean",
                "higher_is_better": True,
            },
            "pieces_placed": {
                "type": "float",
                "description": "Number of tetromino pieces placed",
                "aggregation": "mean",
                "higher_is_better": True,
            },
        }
