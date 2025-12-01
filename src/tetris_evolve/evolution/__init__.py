"""Evolution module for main controller."""

__all__ = []

# Import controller if available
try:
    from .controller import EvolutionController, EvolutionConfig, EvolutionResult
    __all__.extend(["EvolutionController", "EvolutionConfig", "EvolutionResult"])
except ImportError:
    pass
