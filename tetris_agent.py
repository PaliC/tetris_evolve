"""
AlphaEvolve Interface for Tetris
This file contains the code that will be evolved by AlphaEvolve
"""
import numpy as np
from typing import Dict, Any, Tuple


# EVOLVE-BLOCK-START
def compute_heuristics(board: np.ndarray, current_piece: np.ndarray,
                       next_piece: np.ndarray, position: Tuple[int, int]) -> Dict[str, float]:
    """
    Compute heuristic features for the board state.
    This function can be evolved by AlphaEvolve to discover better features.

    Args:
        board: (height, width) binary array of the board state
        current_piece: (4, 4) binary array of current tetromino
        next_piece: (4, 4) binary array of next tetromino
        position: (x, y) current piece position

    Returns:
        Dictionary of heuristic values
    """
    height, width = board.shape

    # Basic heuristics (these will be evolved)
    # FIXED: Correctly calculate aggregate height
    # Height = (total_rows - row_index) for each filled cell
    # Row 0 is at the top, so a block at row 0 has height = total_rows
    column_heights = []
    for col in range(width):
        for row in range(height):
            if board[row, col]:
                column_heights.append(height - row)
                break
        else:
            column_heights.append(0)
    aggregate_height = sum(column_heights)
    complete_lines = np.sum(np.all(board, axis=1))
    holes = count_holes(board)
    bumpiness = compute_bumpiness(board)
    
    return {
        'aggregate_height': aggregate_height,
        'complete_lines': complete_lines,
        'holes': holes,
        'bumpiness': bumpiness,
    }


def count_holes(board: np.ndarray) -> int:
    """Count holes (empty cells with filled cells above)"""
    holes = 0
    height, width = board.shape
    
    for col in range(width):
        block_found = False
        for row in range(height):
            if board[row, col]:
                block_found = True
            elif block_found and not board[row, col]:
                holes += 1
    
    return holes


def compute_bumpiness(board: np.ndarray) -> float:
    """Compute bumpiness (variation in column heights)"""
    height, width = board.shape
    column_heights = []
    
    for col in range(width):
        for row in range(height):
            if board[row, col]:
                column_heights.append(height - row)
                break
        else:
            column_heights.append(0)
    
    bumpiness = sum(abs(column_heights[i] - column_heights[i+1]) 
                    for i in range(len(column_heights) - 1))
    
    return bumpiness


def decide_action(observation: np.ndarray, width: int = 10, height: int = 20) -> int:
    """
    Decide which action to take based on observation.
    This is the main function that AlphaEvolve will evolve.
    
    Args:
        observation: Flattened observation from environment
        width: Board width
        height: Board height
    
    Returns:
        Action index (0-5)
    """
    # Parse observation
    board_size = height * width
    piece_size = 4 * 4
    
    board = observation[:board_size].reshape(height, width)
    current_piece = observation[board_size:board_size + piece_size].reshape(4, 4)
    next_piece = observation[board_size + piece_size:board_size + 2*piece_size].reshape(4, 4)
    position = observation[-2:]
    
    # Denormalize position
    piece_x = int(position[0] * width)
    piece_y = int(position[1] * height)
    
    # Compute heuristics
    heuristics = compute_heuristics(board, current_piece, next_piece, (piece_x, piece_y))
    
    # Simple decision logic (will be evolved)
    # For now: prioritize clearing lines and avoiding holes
    if heuristics['complete_lines'] > 0:
        return 5  # Hard drop if we can clear lines
    elif heuristics['holes'] > 3:
        return 2  # Try rotating to avoid creating holes
    elif piece_x > width // 2:
        return 0  # Move left if on right side
    else:
        return 1  # Move right otherwise


def evaluate_position(board: np.ndarray, piece: np.ndarray, x: int, y: int) -> float:
    """
    Evaluate how good a particular piece placement would be.
    This can be evolved to discover better evaluation strategies.
    
    Args:
        board: Current board state
        piece: Piece to place
        x, y: Position to evaluate
    
    Returns:
        Score (higher is better)
    """
    # Simulate placing piece
    test_board = board.copy()
    
    for i in range(4):
        for j in range(4):
            if piece[i, j]:
                board_x = x + j
                board_y = y + i
                if 0 <= board_y < board.shape[0] and 0 <= board_x < board.shape[1]:
                    test_board[board_y, board_x] = 1
    
    # Compute score based on heuristics
    heuristics = compute_heuristics(test_board, piece, piece, (x, y))
    
    # Weighted combination (these weights will be evolved)
    score = (
        -0.5 * heuristics['aggregate_height'] +
        1.0 * heuristics['complete_lines'] +
        -0.7 * heuristics['holes'] +
        -0.3 * heuristics['bumpiness']
    )
    
    return score
# EVOLVE-BLOCK-END


class EvolvedTetrisAgent:
    """
    Wrapper class for the evolved Tetris agent.
    This provides a clean interface for running games.
    """
    
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.action_history = []
    
    def get_action(self, observation: np.ndarray) -> int:
        """Get action from evolved decision logic"""
        action = decide_action(observation, self.width, self.height)
        self.action_history.append(action)
        return action
    
    def reset(self):
        """Reset agent state"""
        self.action_history = []


if __name__ == "__main__":
    # Test the agent
    from tetris_env import TetrisEnv
    
    env = TetrisEnv(render_mode='human')
    agent = EvolvedTetrisAgent()
    
    obs, info = env.reset()
    agent.reset()
    
    total_reward = 0
    
    for step in range(500):
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            env.render()
        
        if done:
            print("\nGame Over!")
            print(f"Total Steps: {step}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Score: {info['score']}")
            print(f"Lines: {info['lines_cleared']}")
            print(f"Pieces: {info['pieces_placed']}")
            break
