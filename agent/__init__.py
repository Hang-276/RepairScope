__all__ = ["Agent", "TeacherForcingAgent", "PromptAgent", "construct_agent"]


def __getattr__(name):
    """Keep browser/LLM dependencies off the structured RepairScope import path."""
    if name in __all__:
        from importlib import import_module
        value = getattr(import_module("agent.agent"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
