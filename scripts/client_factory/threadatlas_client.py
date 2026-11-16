"""ThreadAtlas, in-process, one local store per user.

ThreadAtlas reads each session into dated facts tied to the words they were read from, and answers a
search with a packet of those records: a header that says how to read them, then up to top_k - 1
bundles, each a dated record with the source passage it came from.

Each add() call is one session. A session ThreadAtlas has already read (byte for byte, at the same
minute) is taken from its store; any other is read on the spot. Either way it is written into that
user's own store, and search runs on that store alone.

Configurable env vars:
    THREADATLAS_REPO        - the ThreadAtlas checkout to import (required; never modified)
    THREADATLAS_STATE_DIR   - root for per-user folders (required)
    THREADATLAS_STORE_DIR   - the pipeline's store (threadatlas.sqlite, threadatlas.npy)
    THREADATLAS_STEP0_DIR   - the pipeline's Step 0 export (documents.jsonl, units.jsonl)
    THREADATLAS_SOURCE      - the benchmark file the store's chats were read from
    THREADATLAS_READ_MODEL  - the model that reads new sessions (default gpt-5.6-luna)
    OPENAI_API_KEY          - for reading sessions the store has not read
"""

import sys

from .base_client import require_env


class ThreadAtlasClient:
    def __init__(self):
        repo = require_env("THREADATLAS_REPO")
        require_env("THREADATLAS_STATE_DIR")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import threadatlas_memory
        self._memory = threadatlas_memory

    def add(self, messages, user_id, *args, **kwargs):
        if not messages:
            return
        minute = str(messages[0].get("chat_time") or "")[:16]
        turns = []
        for message in messages:
            turns.append({"role": message["role"], "content": message["content"]})
        self._memory.add(turns, user_id, minute)

    def search(self, query, user_id, top_k=20, **kwargs):
        return self._memory.search(query, user_id, top_k)

    def delete(self, user_id):
        self._memory.delete(user_id)
