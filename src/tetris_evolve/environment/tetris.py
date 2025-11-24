"""
Tetris environment configuration.

This module provides the EnvironmentConfig implementation for Tetris,
defining observations, actions, rewards, and metrics specific to Tetris gameplay.
"""

from typing import Dict, Any
from tetris_evolve.environment.base import EnvironmentConfig


class TetrisConfig(EnvironmentConfig):
    """
    Configuration for Tetris environment.

    This implementation provides all necessary metadata for LLMs to understand
    and evolve Tetris-playing programs.

    Example:
        >>> config = TetrisConfig()
        >>> wrapper = GenericEnvironmentWrapper(config)
        >>> # Use wrapper to run Tetris episodes
    """

    def __init__(
        self,
        render_mode: str = None,
        max_episode_steps: int = 1000,
        **kwargs,
    ):
        """
        Initialize Tetris configuration.

        Args:
            render_mode: Rendering mode (None, "human", "rgb_array")
            max_episode_steps: Maximum steps per episode
            **kwargs: Additional environment-specific parameters
        """
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.extra_kwargs = kwargs

    def get_env_id(self) -> str:
        """
        Return Tetris environment ID.

        Returns:
            str: Gymnasium Tetris environment identifier

        Note:
            Using a standard Tetris environment. If specific Tetris environment
            is not available, this may need to be adjusted to available envs.
        """
        return "ALE/Tetris-v5"

    def get_env_kwargs(self) -> Dict[str, Any]:
        """
        Return Tetris environment configuration.

        Returns:
            Dict with render_mode, max_episode_steps, and any extra kwargs
        """
        kwargs = {
            "render_mode": self.render_mode,
        }
        if self.max_episode_steps is not None:
            kwargs["max_episode_steps"] = self.max_episode_steps

        # Add any extra kwargs
        kwargs.update(self.extra_kwargs)

        return kwargs

    def get_observation_description(self) -> str:
        """
        Return description of Tetris observation space.

        Returns:
            Detailed description of what observations look like
        """
        return """
Tetris Observation Space:

The observation is typically a 2D or 3D array representing the game state:

For visual environments (Atari):
- Shape: (210, 160, 3) RGB image, or (210, 160) grayscale
- Values: Pixel intensities (0-255)
- The screen shows the Tetris board, current piece, and score

For symbolic/state-based environments:
- Board state: 2D array (typically 20 rows × 10 columns)
  - 0: empty cell
  - 1-7: occupied by specific tetromino type
- Current piece: Type and orientation
- Next piece: Type of upcoming piece
- Game statistics: Lines cleared, level, score

The player function receives this observation and must decide which action to take.
"""

    def get_action_description(self) -> str:
        """
        Return description of Tetris action space.

        Returns:
            Description of available actions
        """
        return """
Tetris Action Space:

Actions control the movement and rotation of the falling tetromino piece.
Typically discrete actions from the following set:

Common action mappings:
- 0: NOOP (no operation, piece falls naturally)
- 1: FIRE (start game / hard drop)
- 2: UP (rotate clockwise)
- 3: RIGHT (move piece right)
- 4: LEFT (move piece left)
- 5: DOWN (soft drop / move down faster)

Some environments may have additional actions:
- Rotate counterclockwise
- Hold piece (swap with hold slot)

The goal is to position and rotate pieces optimally to clear lines
and avoid stacking too high.
"""

    def get_reward_description(self) -> str:
        """
        Return description of Tetris reward structure.

        Returns:
            Description of how rewards are calculated
        """
        return """
Tetris Reward Structure:

Rewards are typically based on:

1. Lines Cleared (primary reward):
   - Single line: +40 to +100 points
   - Double (2 lines): +100 to +300 points
   - Triple (3 lines): +300 to +500 points
   - Tetris (4 lines): +800 to +1200 points (bonus for clearing 4 at once)

2. Piece Placement:
   - Small reward (+1) for successfully placing each piece
   - May receive negative reward for creating holes or bad positions

3. Game Over:
   - Episode ends when pieces stack to the top
   - No additional reward/penalty at termination

4. Survival Time:
   - Implicit reward for surviving longer (more pieces = more chances to score)

The cumulative reward is the total score achieved during the episode.
Goal: Maximize total reward by clearing many lines, especially multiple lines at once.
"""

    def extract_episode_metrics(self, info: Dict) -> Dict[str, float]:
        """
        Extract Tetris-specific metrics from episode info.

        Args:
            info: Episode info dict from environment

        Returns:
            Dict containing Tetris-specific metrics like lines cleared,
            pieces placed, etc.
        """
        metrics = {}

        # Extract common Tetris metrics if available
        if "lines_cleared" in info:
            metrics["lines_cleared"] = float(info["lines_cleared"])

        if "pieces_placed" in info:
            metrics["pieces_placed"] = float(info["pieces_placed"])

        if "max_height" in info:
            metrics["max_height"] = float(info["max_height"])

        if "holes" in info:
            metrics["holes"] = float(info["holes"])

        if "level" in info:
            metrics["level"] = float(info["level"])

        # For Atari environments, extract episode info if present
        if "episode" in info:
            episode_info = info["episode"]
            if "r" in episode_info:  # Atari episode return
                metrics["episode_return"] = float(episode_info["r"])
            if "l" in episode_info:  # Atari episode length
                metrics["episode_length"] = float(episode_info["l"])

        # Calculate derived metrics if we have the data
        if "lines_cleared" in metrics and "pieces_placed" in metrics:
            if metrics["pieces_placed"] > 0:
                metrics["efficiency"] = metrics["lines_cleared"] / metrics["pieces_placed"]
            else:
                metrics["efficiency"] = 0.0

        return metrics

    def get_player_interface_template(self) -> str:
        """
        Return code template for Tetris player.

        Returns:
            Python code template showing expected player interface
        """
        return '''
def select_action(observation, info):
    """
    Select the next action for the Tetris game.

    Args:
        observation: Game state observation
            - For visual: RGB/grayscale image array (210, 160, 3) or (210, 160)
            - For state-based: Dict with 'board', 'current_piece', 'next_piece'
        info: Additional game information
            - May contain: lines_cleared, level, score, etc.

    Returns:
        int: Action to take (0-5)
            0: NOOP
            1: FIRE/Hard drop
            2: UP/Rotate
            3: RIGHT
            4: LEFT
            5: DOWN/Soft drop

    Example simple strategy:
        # Random action
        import random
        return random.randint(0, 5)

    Example heuristic strategy:
        # Analyze board and choose action to minimize holes
        # and maximize line clears
        board = observation.get('board') if isinstance(observation, dict) else None
        if board is not None:
            # Your heuristic logic here
            action = calculate_best_action(board)
            return action
        return 0  # Default: NOOP
    """
    # Your evolved code here
    # Default implementation: random action
    import random
    return random.randint(0, 5)
'''
