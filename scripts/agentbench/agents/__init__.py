from __future__ import annotations

from .hermes import HermesAgentAdapter
from .openclaw import OpenClawAgentAdapter


AGENT_REGISTRY = {
    "hermes": HermesAgentAdapter,
    "openclaw": OpenClawAgentAdapter,
}


def create_agent(name: str, config: dict):
    try:
        cls = AGENT_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent {name!r}. Supported: {sorted(AGENT_REGISTRY)}") from exc
    return cls(config)
