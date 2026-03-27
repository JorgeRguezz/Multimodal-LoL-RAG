__all__ = ["KnowledgeBuilder"]


def __getattr__(name: str):
    if name == "KnowledgeBuilder":
        from .builder import KnowledgeBuilder

        return KnowledgeBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
