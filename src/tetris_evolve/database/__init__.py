"""
Database module: Program database and persistence.

This module manages the in-memory program database and file-based persistence for
storing program history, metrics, and evolutionary lineage.

Usage:
    from tetris_evolve.database import Program, ProgramDatabase, ProgramMetrics

    db = ProgramDatabase()
    program = Program(
        program_id=db.generate_program_id(),
        generation=0,
        code="def select_action(obs): return 5",
    )
    db.add_program(program)
"""
from .program import (
    Program,
    ProgramDatabase,
    ProgramMetrics,
    MutationInfo,
)

__all__ = [
    "Program",
    "ProgramDatabase",
    "ProgramMetrics",
    "MutationInfo",
]
