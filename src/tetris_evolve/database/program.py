"""
Program database for tracking evolved programs.

This module provides data structures and storage for programs across
generations, including metrics, lineage, and mutation history.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class ProgramMetrics:
    """Metrics collected from evaluating a program."""
    avg_score: float
    std_score: float
    avg_lines_cleared: float
    games_played: int
    max_score: float = 0.0
    min_score: float = 0.0
    avg_survival_time: float = 0.0
    std_survival_time: float = 0.0
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "avg_score": self.avg_score,
            "std_score": self.std_score,
            "avg_lines_cleared": self.avg_lines_cleared,
            "games_played": self.games_played,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "avg_survival_time": self.avg_survival_time,
            "std_survival_time": self.std_survival_time,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProgramMetrics":
        """Create from dictionary."""
        return cls(
            avg_score=d.get("avg_score", 0.0),
            std_score=d.get("std_score", 0.0),
            avg_lines_cleared=d.get("avg_lines_cleared", 0.0),
            games_played=d.get("games_played", 0),
            max_score=d.get("max_score", 0.0),
            min_score=d.get("min_score", 0.0),
            avg_survival_time=d.get("avg_survival_time", 0.0),
            std_survival_time=d.get("std_survival_time", 0.0),
            success_rate=d.get("success_rate", 1.0),
        )


@dataclass
class MutationInfo:
    """Information about how a program was mutated from its parent."""
    strategy: str  # e.g., "exploitation", "exploration", "crossover"
    focus_area: Optional[str] = None  # e.g., "hole_management", "lookahead"
    rllm_id: Optional[str] = None
    guidance: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    rllm_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "focus_area": self.focus_area,
            "rllm_id": self.rllm_id,
            "guidance": self.guidance,
            "constraints": self.constraints,
            "rllm_explanation": self.rllm_explanation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MutationInfo":
        """Create from dictionary."""
        return cls(
            strategy=d.get("strategy", "unknown"),
            focus_area=d.get("focus_area"),
            rllm_id=d.get("rllm_id"),
            guidance=d.get("guidance"),
            constraints=d.get("constraints", []),
            rllm_explanation=d.get("rllm_explanation"),
        )


@dataclass
class Program:
    """A program in the evolution database."""
    program_id: str
    generation: int
    code: str
    parent_ids: List[str] = field(default_factory=list)
    metrics: Optional[ProgramMetrics] = None
    mutation_info: Optional[MutationInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # FIXED: Use timezone-aware datetime instead of deprecated utcnow()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "program_id": self.program_id,
            "generation": self.generation,
            "code": self.code,
            "parent_ids": self.parent_ids,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "mutation_info": self.mutation_info.to_dict() if self.mutation_info else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Program":
        """Create from dictionary."""
        metrics = None
        if d.get("metrics"):
            metrics = ProgramMetrics.from_dict(d["metrics"])

        mutation_info = None
        if d.get("mutation_info"):
            mutation_info = MutationInfo.from_dict(d["mutation_info"])

        return cls(
            program_id=d["program_id"],
            generation=d["generation"],
            code=d["code"],
            parent_ids=d.get("parent_ids", []),
            metrics=metrics,
            mutation_info=mutation_info,
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


class ProgramDatabase:
    """
    In-memory database for program storage with file persistence.

    Stores programs across generations with support for:
    - CRUD operations
    - Lineage tracking
    - Metric queries
    - File-based persistence
    """

    def __init__(self):
        """Initialize an empty database."""
        self._programs: Dict[str, Program] = {}
        self._next_id: int = 1

    def add_program(self, program: Program) -> None:
        """
        Add a program to the database.

        Args:
            program: Program to add
        """
        self._programs[program.program_id] = program

    def get_program(self, program_id: str) -> Optional[Program]:
        """
        Get a program by ID.

        Args:
            program_id: ID of the program

        Returns:
            Program if found, None otherwise
        """
        return self._programs.get(program_id)

    def get_all_programs(self) -> List[Program]:
        """
        Get all programs in the database.

        Returns:
            List of all programs
        """
        return list(self._programs.values())

    def get_programs_by_generation(self, generation: int) -> List[Program]:
        """
        Get all programs from a specific generation.

        Args:
            generation: Generation number

        Returns:
            List of programs from that generation
        """
        return [p for p in self._programs.values() if p.generation == generation]

    def update_metrics(self, program_id: str, metrics: ProgramMetrics) -> None:
        """
        Update metrics for a program.

        Args:
            program_id: ID of the program
            metrics: New metrics
        """
        if program_id in self._programs:
            self._programs[program_id].metrics = metrics

    def get_top_programs(
        self,
        generation: Optional[int] = None,
        n: int = 10,
        metric: str = "avg_score"
    ) -> List[Program]:
        """
        Get top N programs by a metric.

        Args:
            generation: Optional generation to filter by
            n: Number of programs to return
            metric: Metric to sort by

        Returns:
            List of top programs sorted by metric (descending)
        """
        programs = self._programs.values()

        if generation is not None:
            programs = [p for p in programs if p.generation == generation]

        # Filter to only programs with metrics
        programs_with_metrics = [p for p in programs if p.metrics is not None]

        # Sort by metric
        def get_metric(p: Program) -> float:
            if p.metrics is None:
                return 0.0
            return getattr(p.metrics, metric, 0.0)

        sorted_programs = sorted(programs_with_metrics, key=get_metric, reverse=True)

        return sorted_programs[:n]

    def get_current_generation(self) -> int:
        """
        Get the highest generation number in the database.

        Returns:
            Current generation number (0 if database is empty)
        """
        if not self._programs:
            return 0
        return max(p.generation for p in self._programs.values())

    def generate_program_id(self) -> str:
        """
        Generate a unique program ID.

        Returns:
            New unique program ID
        """
        program_id = f"prog_{self._next_id:05d}"
        self._next_id += 1
        return program_id

    def get_children(self, program_id: str) -> List[Program]:
        """
        Get all programs that have this program as a parent.

        Args:
            program_id: ID of the parent program

        Returns:
            List of child programs
        """
        return [
            p for p in self._programs.values()
            if program_id in p.parent_ids
        ]

    def get_lineage(self, program_id: str) -> List[Program]:
        """
        Trace the full lineage of a program back to its ancestors.

        Args:
            program_id: ID of the program

        Returns:
            List of programs from the given program back to its oldest ancestor
        """
        lineage = []
        visited = set()
        current_ids = [program_id]

        while current_ids:
            current_id = current_ids.pop(0)
            if current_id in visited:
                continue

            visited.add(current_id)
            program = self.get_program(current_id)
            if program:
                lineage.append(program)
                current_ids.extend(program.parent_ids)

        return lineage

    def save(self, path: Path) -> None:
        """
        Save the database to disk.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save index
        index = {
            "next_id": self._next_id,
            "program_count": len(self._programs),
        }
        with open(path / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        # Save programs
        programs_data = [p.to_dict() for p in self._programs.values()]
        with open(path / "programs.json", "w") as f:
            json.dump(programs_data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ProgramDatabase":
        """
        Load a database from disk.

        Args:
            path: Directory to load from

        Returns:
            Loaded database (empty if path doesn't exist)
        """
        path = Path(path)
        db = cls()

        if not path.exists():
            return db

        # Load index
        index_path = path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
                db._next_id = index.get("next_id", 1)

        # Load programs
        programs_path = path / "programs.json"
        if programs_path.exists():
            with open(programs_path) as f:
                programs_data = json.load(f)
                for d in programs_data:
                    program = Program.from_dict(d)
                    db._programs[program.program_id] = program

        return db

    def __len__(self) -> int:
        """Return number of programs in database."""
        return len(self._programs)
