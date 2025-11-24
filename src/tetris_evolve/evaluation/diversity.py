"""
Diversity metrics for evolution.

Provides functions to measure and preserve diversity in the
population to prevent premature convergence.
"""
from typing import Any, Dict, List, Tuple
import hashlib
from collections import Counter


def _tokenize_code(code: str) -> List[str]:
    """Simple tokenization of Python code."""
    # Remove comments and normalize whitespace
    lines = []
    for line in code.split('\n'):
        # Remove comments
        if '#' in line:
            line = line[:line.index('#')]
        line = line.strip()
        if line:
            lines.append(line)

    # Split into tokens (simple approach)
    tokens = []
    for line in lines:
        # Split on common delimiters
        parts = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
        parts = parts.replace(':', ' ').replace('[', ' ').replace(']', ' ')
        parts = parts.replace('{', ' ').replace('}', ' ').replace('=', ' ')
        tokens.extend(parts.split())

    return tokens


def _jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    set1 = set(tokens1)
    set2 = set(tokens2)

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def code_similarity(code1: str, code2: str) -> float:
    """
    Compute similarity between two code samples.

    Args:
        code1: First code sample
        code2: Second code sample

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    tokens1 = _tokenize_code(code1)
    tokens2 = _tokenize_code(code2)
    return _jaccard_similarity(tokens1, tokens2)


def compute_code_diversity(codes: List[str]) -> float:
    """
    Compute overall diversity of a set of code samples.

    Higher values indicate more diversity.

    Args:
        codes: List of code samples

    Returns:
        Diversity score between 0.0 and 1.0
    """
    if len(codes) < 2:
        return 1.0  # Single sample is maximally diverse

    # Compute pairwise dissimilarities
    total_dissimilarity = 0.0
    count = 0

    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            similarity = code_similarity(codes[i], codes[j])
            total_dissimilarity += (1.0 - similarity)
            count += 1

    # Average dissimilarity
    return total_dissimilarity / count if count > 0 else 0.0


def select_diverse(
    candidates: List[Dict[str, Any]],
    n: int,
    diversity_weight: float = 0.3,
    score_key: str = "score",
    code_key: str = "code",
) -> List[Dict[str, Any]]:
    """
    Select n candidates balancing score and diversity.

    Uses a greedy algorithm that iteratively selects candidates
    that maximize: score * (1 - diversity_weight) + diversity_contribution * diversity_weight

    Args:
        candidates: List of candidate dicts with score and code
        n: Number of candidates to select
        diversity_weight: Weight for diversity (0-1)
        score_key: Key for score in candidate dict
        code_key: Key for code in candidate dict

    Returns:
        List of n selected candidates
    """
    if len(candidates) <= n:
        return candidates

    # Normalize scores
    scores = [c[score_key] for c in candidates]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    def normalized_score(c):
        return (c[score_key] - min_score) / score_range

    selected = []
    remaining = list(range(len(candidates)))

    # Greedy selection
    for _ in range(n):
        best_idx = None
        best_value = -float('inf')

        for idx in remaining:
            candidate = candidates[idx]

            # Score component
            score_component = normalized_score(candidate) * (1 - diversity_weight)

            # Diversity component (average dissimilarity to selected)
            if selected:
                dissimilarities = [
                    1.0 - code_similarity(candidate[code_key], candidates[s][code_key])
                    for s in selected
                ]
                diversity_component = sum(dissimilarities) / len(dissimilarities)
            else:
                diversity_component = 1.0  # First selection gets max diversity

            diversity_component *= diversity_weight

            total_value = score_component + diversity_component

            if total_value > best_value:
                best_value = total_value
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def compute_population_stats(
    programs: List[Dict[str, Any]],
    code_key: str = "code",
) -> Dict[str, Any]:
    """
    Compute diversity statistics for a population.

    Args:
        programs: List of program dicts with code
        code_key: Key for code in program dict

    Returns:
        Dictionary with diversity statistics
    """
    if not programs:
        return {"diversity": 0.0, "num_unique": 0, "clusters": 0}

    codes = [p[code_key] for p in programs]

    # Compute overall diversity
    diversity = compute_code_diversity(codes)

    # Count unique structures (by hash)
    hashes = set()
    for code in codes:
        tokens = _tokenize_code(code)
        # Hash the sorted tokens for structure comparison
        structure_hash = hashlib.md5(' '.join(sorted(set(tokens))).encode()).hexdigest()[:8]
        hashes.add(structure_hash)

    return {
        "diversity": diversity,
        "num_programs": len(programs),
        "num_unique_structures": len(hashes),
        "structure_ratio": len(hashes) / len(programs),
    }
