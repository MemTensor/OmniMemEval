"""Central registry of supported memory framework libraries.

Add new libraries here — all evaluation scripts import SUPPORTED_LIBS
from this module, so a single change propagates everywhere.

When adding a new lib:
  1. Add a Client class in scripts/client_factory/<name>_client.py
  2. Register it in _LIB_CLIENT_REGISTRY below  (module_name, class_name)
"""

from importlib import import_module

DEFAULT_LIB = "memos"

# ── Client factory ───────────────────────────────────────────────────────────
# Lazy registry: maps lib name -> (module_name, class_name)
# Clients are only imported when actually requested, so a missing dependency
# for one product never breaks the others.
_LIB_CLIENT_REGISTRY = {
    "memos":       ("memos_client",       "MemosClient"),
    "everos":      ("everos_client",      "EverosClient"),
    "mem0":        ("mem0_client",        "Mem0Client"),
    "supermemory": ("supermemory_client", "SupermemoryClient"),
    "zep":         ("zep_client",         "ZepClient"),
    "letta":       ("letta_client",       "LettaClient"),
    "hindsight":   ("hindsight_client",  "HindsightClient"),
    "graphiti":    ("graphiti_client",   "GraphitiClient"),
    "cognee":      ("cognee_client",      "CogneeClient"),
    "viking":      ("viking_client",      "VikingClient"),
    "memori":      ("memori_client",      "MemoriClient"),
    "memmachine":  ("memmachine_client",  "MemMachineClient"),
    "memorylake":  ("memorylake_client",  "MemoryLakeClient"),
    "backboard":   ("backboard_client",   "BackboardClient"),
    "mem9":        ("mem9_client",        "Mem9Client"),
}

SUPPORTED_LIBS = list(_LIB_CLIENT_REGISTRY.keys())


def _load_client_class(lib_name: str):
    """Import and return the client class for *lib_name* on demand."""
    entry = _LIB_CLIENT_REGISTRY.get(lib_name)
    if entry is None:
        raise ValueError(
            f"Unknown lib: {lib_name!r}. Supported: {SUPPORTED_LIBS}"
        )
    module_name, class_name = entry
    try:
        module = import_module(f".{module_name}", package=__package__)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import client module for {lib_name!r}: {exc}. "
            f"Make sure its dependencies are installed."
        ) from exc
    return getattr(module, class_name)


def create_client(lib_name: str):
    """Create a memory framework client by lib name.

    Centralizes the lib->client mapping so that ingestion/search scripts
    don't need their own if/elif chains for client creation.
    """
    cls = _load_client_class(lib_name)
    return cls()