"""
Tests for the program database.

Following TDD: These tests define the expected behavior of Program and
ProgramDatabase classes.
"""
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from tetris_evolve.database import (
    Program,
    ProgramDatabase,
    ProgramMetrics,
    MutationInfo,
)


class TestProgram:
    """Tests for the Program dataclass."""

    def test_program_creation(self):
        """Should create a Program with required fields."""
        program = Program(
            program_id="prog_001",
            generation=0,
            code="def test(): pass",
        )

        assert program.program_id == "prog_001"
        assert program.generation == 0
        assert program.code == "def test(): pass"

    def test_program_with_metrics(self):
        """Should create a Program with metrics."""
        metrics = ProgramMetrics(
            avg_score=100.0,
            std_score=10.0,
            avg_lines_cleared=5.0,
            games_played=10,
        )

        program = Program(
            program_id="prog_001",
            generation=0,
            code="def test(): pass",
            metrics=metrics,
        )

        assert program.metrics.avg_score == 100.0
        assert program.metrics.games_played == 10

    def test_program_with_parent_ids(self):
        """Should track parent IDs for lineage."""
        program = Program(
            program_id="prog_002",
            generation=1,
            code="def test(): pass",
            parent_ids=["prog_001"],
        )

        assert program.parent_ids == ["prog_001"]

    def test_program_with_mutation_info(self):
        """Should track mutation information."""
        mutation_info = MutationInfo(
            strategy="exploitation",
            focus_area="hole_management",
            rllm_id="rllm_001",
        )

        program = Program(
            program_id="prog_002",
            generation=1,
            code="def test(): pass",
            mutation_info=mutation_info,
        )

        assert program.mutation_info.strategy == "exploitation"
        assert program.mutation_info.focus_area == "hole_management"

    def test_program_to_dict(self):
        """Should convert to dictionary for serialization."""
        program = Program(
            program_id="prog_001",
            generation=0,
            code="def test(): pass",
        )

        d = program.to_dict()

        assert isinstance(d, dict)
        assert d["program_id"] == "prog_001"
        assert d["generation"] == 0
        assert d["code"] == "def test(): pass"

    def test_program_from_dict(self):
        """Should create Program from dictionary."""
        d = {
            "program_id": "prog_001",
            "generation": 0,
            "code": "def test(): pass",
            "parent_ids": [],
            "metrics": None,
            "mutation_info": None,
            "metadata": {},
            "created_at": "2025-01-23T14:30:00",
        }

        program = Program.from_dict(d)

        assert program.program_id == "prog_001"
        assert program.generation == 0

    def test_program_has_created_at(self):
        """Should have a created_at timestamp."""
        program = Program(
            program_id="prog_001",
            generation=0,
            code="def test(): pass",
        )

        assert program.created_at is not None
        # Should be parseable as datetime
        if isinstance(program.created_at, str):
            datetime.fromisoformat(program.created_at)


class TestProgramMetrics:
    """Tests for ProgramMetrics dataclass."""

    def test_metrics_creation(self):
        """Should create metrics with all fields."""
        metrics = ProgramMetrics(
            avg_score=100.0,
            std_score=10.0,
            avg_lines_cleared=5.0,
            games_played=10,
        )

        assert metrics.avg_score == 100.0
        assert metrics.std_score == 10.0
        assert metrics.games_played == 10

    def test_metrics_to_dict(self):
        """Should convert to dictionary."""
        metrics = ProgramMetrics(
            avg_score=100.0,
            std_score=10.0,
            avg_lines_cleared=5.0,
            games_played=10,
        )

        d = metrics.to_dict()

        assert d["avg_score"] == 100.0
        assert d["games_played"] == 10

    def test_metrics_from_dict(self):
        """Should create from dictionary."""
        d = {
            "avg_score": 100.0,
            "std_score": 10.0,
            "avg_lines_cleared": 5.0,
            "games_played": 10,
        }

        metrics = ProgramMetrics.from_dict(d)

        assert metrics.avg_score == 100.0


class TestProgramDatabase:
    """Tests for ProgramDatabase."""

    @pytest.fixture
    def db(self):
        """Create a fresh database."""
        return ProgramDatabase()

    @pytest.fixture
    def sample_program(self):
        """Create a sample program."""
        return Program(
            program_id="prog_001",
            generation=0,
            code="def test(): pass",
        )

    def test_add_program(self, db, sample_program):
        """Should add a program to the database."""
        db.add_program(sample_program)

        assert db.get_program("prog_001") == sample_program

    def test_get_nonexistent_program(self, db):
        """Should return None for nonexistent program."""
        assert db.get_program("nonexistent") is None

    def test_get_programs_by_generation(self, db):
        """Should retrieve all programs from a generation."""
        p1 = Program(program_id="p1", generation=0, code="")
        p2 = Program(program_id="p2", generation=0, code="")
        p3 = Program(program_id="p3", generation=1, code="")

        db.add_program(p1)
        db.add_program(p2)
        db.add_program(p3)

        gen0_programs = db.get_programs_by_generation(0)

        assert len(gen0_programs) == 2
        assert all(p.generation == 0 for p in gen0_programs)

    def test_get_all_programs(self, db):
        """Should retrieve all programs."""
        p1 = Program(program_id="p1", generation=0, code="")
        p2 = Program(program_id="p2", generation=1, code="")

        db.add_program(p1)
        db.add_program(p2)

        all_programs = db.get_all_programs()

        assert len(all_programs) == 2

    def test_update_program_metrics(self, db, sample_program):
        """Should update program metrics."""
        db.add_program(sample_program)

        metrics = ProgramMetrics(
            avg_score=100.0,
            std_score=10.0,
            avg_lines_cleared=5.0,
            games_played=10,
        )

        db.update_metrics("prog_001", metrics)

        program = db.get_program("prog_001")
        assert program.metrics.avg_score == 100.0

    def test_get_top_programs(self, db):
        """Should retrieve top programs by score."""
        p1 = Program(
            program_id="p1", generation=0, code="",
            metrics=ProgramMetrics(avg_score=100.0, std_score=0, avg_lines_cleared=0, games_played=10)
        )
        p2 = Program(
            program_id="p2", generation=0, code="",
            metrics=ProgramMetrics(avg_score=200.0, std_score=0, avg_lines_cleared=0, games_played=10)
        )
        p3 = Program(
            program_id="p3", generation=0, code="",
            metrics=ProgramMetrics(avg_score=150.0, std_score=0, avg_lines_cleared=0, games_played=10)
        )

        db.add_program(p1)
        db.add_program(p2)
        db.add_program(p3)

        top_programs = db.get_top_programs(generation=0, n=2)

        assert len(top_programs) == 2
        assert top_programs[0].program_id == "p2"  # Highest score
        assert top_programs[1].program_id == "p3"

    def test_get_current_generation(self, db):
        """Should track current generation."""
        assert db.get_current_generation() == 0

        p1 = Program(program_id="p1", generation=0, code="")
        p2 = Program(program_id="p2", generation=2, code="")

        db.add_program(p1)
        db.add_program(p2)

        assert db.get_current_generation() == 2

    def test_generate_program_id(self, db):
        """Should generate unique program IDs."""
        id1 = db.generate_program_id()
        id2 = db.generate_program_id()

        assert id1 != id2
        assert id1.startswith("prog_")
        assert id2.startswith("prog_")


class TestProgramDatabasePersistence:
    """Tests for database persistence."""

    def test_save_and_load(self):
        """Should save and load database from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "programs"

            # Create and save
            db1 = ProgramDatabase()
            p1 = Program(program_id="p1", generation=0, code="test code")
            db1.add_program(p1)
            db1.save(db_path)

            # Load into new instance
            db2 = ProgramDatabase.load(db_path)

            assert db2.get_program("p1") is not None
            assert db2.get_program("p1").code == "test code"

    def test_save_creates_directory(self):
        """Should create directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "programs"

            db = ProgramDatabase()
            db.add_program(Program(program_id="p1", generation=0, code=""))
            db.save(db_path)

            assert db_path.exists()

    def test_load_nonexistent_returns_empty(self):
        """Should return empty database if path doesn't exist."""
        db = ProgramDatabase.load(Path("/nonexistent/path"))

        assert len(db.get_all_programs()) == 0


class TestProgramDatabaseLineage:
    """Tests for lineage tracking."""

    @pytest.fixture
    def db(self):
        return ProgramDatabase()

    def test_get_children(self, db):
        """Should find all children of a program."""
        parent = Program(program_id="parent", generation=0, code="")
        child1 = Program(program_id="child1", generation=1, code="", parent_ids=["parent"])
        child2 = Program(program_id="child2", generation=1, code="", parent_ids=["parent"])
        unrelated = Program(program_id="unrelated", generation=1, code="", parent_ids=[])

        db.add_program(parent)
        db.add_program(child1)
        db.add_program(child2)
        db.add_program(unrelated)

        children = db.get_children("parent")

        assert len(children) == 2
        assert all(c.parent_ids == ["parent"] for c in children)

    def test_get_lineage(self, db):
        """Should trace lineage back to original ancestors."""
        p0 = Program(program_id="p0", generation=0, code="")
        p1 = Program(program_id="p1", generation=1, code="", parent_ids=["p0"])
        p2 = Program(program_id="p2", generation=2, code="", parent_ids=["p1"])

        db.add_program(p0)
        db.add_program(p1)
        db.add_program(p2)

        lineage = db.get_lineage("p2")

        assert len(lineage) == 3
        assert lineage[0].program_id == "p2"
        assert lineage[-1].program_id == "p0"
