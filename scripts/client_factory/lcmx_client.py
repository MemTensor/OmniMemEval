import hashlib
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from .base_client import env_int, env_str, require_env

_DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Match FastEmbed 0.8.0 TextEmbedding.embed's internal default grouping.
_EMBED_BATCH_SIZE = 256

# One warmed embedder per process (FastEmbed loads an ONNX model; ingestion
# runs several client instances on worker threads).
_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()
_USER_LOCKS = {}
_USER_LOCKS_GUARD = threading.Lock()
_WARNED = set()


def _validate_user_id(user_id):
    rule = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}"
    if not isinstance(user_id, str) or re.fullmatch(rule, user_id) is None:
        raise ValueError(f"user_id must fullmatch {rule}")
    return user_id


def _user_key(user_id):
    """Keep accepted IDs distinct even on case-insensitive filesystems."""
    user_id = _validate_user_id(user_id)
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
    return f"{user_id[:24].lower()}-{digest}"


def _user_lock(user_id):
    key = _user_key(user_id)
    with _USER_LOCKS_GUARD:
        return _USER_LOCKS.setdefault(key, threading.Lock())


class _LockedEmbedder:
    """Serialize calls into the shared embedder across worker threads."""

    def __init__(self, provider):
        self._provider = provider
        self._lock = threading.Lock()

    @property
    def model_id(self):
        return self._provider.model_id

    @property
    def dim(self):
        return int(self._provider.dim)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def embed_documents(self, texts):
        with self._lock:
            return self._provider.embed_documents(texts)

    def embed_query(self, text):
        with self._lock:
            return self._provider.embed_query(text)

    def embed_document_batches(self, texts):
        with self._lock:
            method = getattr(self._provider, "embed_document_batches", None)
            if callable(method):
                yield from method(texts)
            else:
                yield SimpleNamespace(vectors=self._provider.embed_documents(texts))


class LcmxClient:
    """LCM-X (hermes-lcm) in-process client for OmniMemEval.

    Runs a headless LCM-X store per ``user_id`` under
    ``$LCMX_DB_ROOT/<user_key>/`` using the same construction as the LCM-X
    memorybench bridge (benchmarks/qa-harness/src/providers/hermes-lcm/bridge):
    lossless message rows, conversational chunk embeddings, one deterministic
    session summary per ``add`` call, and the production ``lcm_recall`` tool
    for search. No LLM is used inside LCM-X.

    Each ``add`` call is one LCM-X session. Every message keeps its role and
    its ``chat_time`` as the host event time (``observed_at``), which
    ``lcm_recall`` returns as each message hit's ``event_time``.

    Configurable env vars:
        LCMX_REPO            – LCM-X checkout to import (required; never modified)
        LCMX_DB_ROOT         – root for per-user stores (required)
        LCMX_CONFIG          – "fleet" (default): fastembed embeddings on;
                               "off": full-text only
        LCMX_EMBED_PROVIDER  – embedding provider for fleet (default: fastembed)
        LCMX_EMBED_MODEL     – embedding model (default: BAAI/bge-small-en-v1.5)
        LCMX_EMBED_CACHE     – optional SQLite text->vector cache shared with the
                               LCM-X LongMemEval harness
        LCMX_SEARCH_SCOPE_BIAS – optional lcm_recall scope_bias override
        LCMX_CHUNK_POLICY    – LCM-X chunk policy: "conversational" (default; skips
                               turns under 40 tokens) or "full" (every message)
        LCMX_DELIVERY        – "full" (default): full stored message per hit;
                               "snippet": lcm_recall's 300-char snippet
    """

    def __init__(self):
        repo = Path(require_env("LCMX_REPO")).expanduser().resolve()
        if not repo.is_dir():
            raise RuntimeError(f"LCMX_REPO does not exist: {repo}")
        self._db_root = Path(require_env("LCMX_DB_ROOT")).expanduser().resolve()
        self._db_root.mkdir(parents=True, exist_ok=True)

        mode = (env_str("LCMX_CONFIG", "fleet") or "fleet").lower()
        if mode not in ("fleet", "off"):
            raise ValueError(f"LCMX_CONFIG must be 'fleet' or 'off', got {mode!r}")
        self._embeddings = mode == "fleet"
        self._provider_name = (env_str("LCMX_EMBED_PROVIDER", "fastembed") or "fastembed").lower()
        self._model = env_str("LCMX_EMBED_MODEL", _DEFAULT_FASTEMBED_MODEL)
        self._scope_bias = env_str("LCMX_SEARCH_SCOPE_BIAS")
        self._top_k_cap = env_int("LCMX_TOPK_CAP", 25, min_value=1)
        self._delivery = (env_str("LCMX_DELIVERY", "full") or "full").lower()
        self._chunk_policy = (env_str("LCMX_CHUNK_POLICY", "conversational") or "conversational").lower()
        if self._chunk_policy not in ("conversational", "full"):
            raise ValueError(f"LCMX_CHUNK_POLICY must be 'conversational' or 'full', got {self._chunk_policy!r}")
        if self._delivery not in ("full", "snippet"):
            raise ValueError(f"LCMX_DELIVERY must be 'full' or 'snippet', got {self._delivery!r}")

        # Import LCM-X exactly the way its own LongMemEval harness does.
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        cache = env_str("LCMX_EMBED_CACHE")
        if cache:
            os.environ.setdefault("LCM_LONGMEMEVAL_EMBED_CACHE", cache)
        from benchmarking.longmemeval import (
            _ensure_hermes_lcm_package,
            deterministic_session_summary,
            resolve_harness_provider,
        )

        _ensure_hermes_lcm_package()
        self._summary = deterministic_session_summary
        self._embedder = None
        self._dim = 0
        if self._embeddings:
            global _EMBEDDER
            with _EMBEDDER_LOCK:
                if _EMBEDDER is None:
                    _EMBEDDER = _LockedEmbedder(
                        resolve_harness_provider(self._provider_name, self._model)
                    )
            self._embedder = _EMBEDDER
            self._dim = self._embedder.dim
            self._model = self._embedder.model_id

    # -- store helpers --------------------------------------------------------

    def _user_dir(self, user_id):
        return self._db_root / _user_key(user_id)

    def _db_path(self, user_id):
        return self._user_dir(user_id) / "lcm.db"

    def _sessions_path(self, user_id):
        # session_id -> {"order", "time"}: the session order (recency prior, as
        # in the bridge) and the session event time for summary hits, which
        # carry no per-message observed_at.
        return self._user_dir(user_id) / "sessions.json"

    def _load_sessions(self, user_id):
        path = self._sessions_path(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _config(self, user_id):
        from hermes_lcm.config import LCMConfig

        if not self._embeddings:
            return LCMConfig(database_path=str(self._db_path(user_id)),
                             embeddings_enabled=False)
        return LCMConfig(
            database_path=str(self._db_path(user_id)),
            embeddings_enabled=True,
            embedding_provider=self._provider_name,
            embedding_model=self._model,
        )

    def _open(self, user_id):
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        self._user_dir(user_id).mkdir(parents=True, exist_ok=True)
        config = self._config(user_id)
        db = str(self._db_path(user_id))
        store = MessageStore(db, ingest_protection_config=config)
        dag = SummaryDAG(db)
        vector_store = VectorStore(db, config=config)
        return config, store, dag, vector_store

    # -- add ------------------------------------------------------------------

    def add(self, messages, user_id, *args, **kwargs):
        """Ingest one session of messages losslessly.

        Args:
            messages: ``[{"role", "content", "chat_time"?}, ...]``.
            user_id: Store owner; one LCM-X database per user.
            session_key / session_id / conv_id: Optional session identifier
                (positional third argument also accepted). Without one, the
                session is named ``<user_id>-s<order>``.
            timestamp: Optional ISO-8601 event time used when a message has
                no ``chat_time``.
        """
        user_id = _validate_user_id(user_id)
        if not messages:
            return
        session_key = (
            kwargs.get("session_key")
            or kwargs.get("session_id")
            or kwargs.get("conv_id")
            or (args[0] if args and isinstance(args[0], str) else None)
        )
        default_time = kwargs.get("timestamp")
        if isinstance(default_time, (int, float)) and not isinstance(default_time, bool):
            from datetime import datetime, timezone

            default_time = datetime.fromtimestamp(default_time, tz=timezone.utc).isoformat()

        from hermes_lcm.chunking import iter_message_chunks
        from hermes_lcm.dag import SummaryNode
        from hermes_lcm.vector_store import EmbeddingIdentity

        with _user_lock(user_id):
            sessions = self._load_sessions(user_id)
            order = len(sessions) + 1
            session_id = str(session_key) if session_key else f"{user_id}-s{order}"
            if not session_key and session_id in sessions:
                session_id = f"{session_id}#{order}"

            rows_in = []
            for m in messages:
                event_time = m.get("chat_time") or default_time
                row = {
                    "role": str(m.get("role", "user")),
                    "content": str(m.get("content", "")),
                }
                if event_time:
                    row["timestamp"] = event_time
                rows_in.append(row)
            session_time = next((r["timestamp"] for r in rows_in if r.get("timestamp")), None)

            _config, store, dag, vector_store = self._open(user_id)
            try:
                existing, nodes = [], []
                if session_key:
                    existing = store.get_session_messages(session_id, limit=len(rows_in) + 1)
                    nodes = dag.get_session_nodes(session_id, depth=0, limit=2)
                    def row_hash(rows):
                        pairs = [(r["role"], r["content"]) for r in rows]
                        return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).digest()
                    if (existing or nodes) and (
                        row_hash(existing) != row_hash(rows_in) or len(nodes) != 1
                    ):
                        raise ValueError("session rows or depth-0 node mismatch")
                store_ids = [r["store_id"] for r in existing] if existing else store.append_batch(
                    session_id, rows_in, source="benchmark", conversation_id=session_id
                )
                summary_text = self._summary(
                    [{"role": r["role"], "content": r["content"]} for r in rows_in]
                )
                node_id = nodes[0].node_id if nodes else dag.add_node(
                    SummaryNode(
                        session_id=session_id,
                        depth=0,
                        summary=summary_text,
                        token_count=len(summary_text.split()),
                        source_token_count=sum(len(r["content"].split()) for r in rows_in),
                        source_type="messages",
                        created_at=float(order),
                    )
                )
                if self._embeddings:
                    vector_store.register_profile(self._model, self._provider_name, self._dim)
                    identity = vector_store.capture_identity(
                        self._model, provider=self._provider_name
                    )
                    vector_store.register_profile(
                        self._model, self._provider_name, self._dim, task="chunk"
                    )
                    chunk_identity = EmbeddingIdentity.canonical(
                        self._provider_name, self._model, "", self._dim,
                        "float32", "little", "chunk",
                    )
                    rows = [
                        {"store_id": sid, "role": r["role"], "content": r["content"]}
                        for sid, r in zip(store_ids, rows_in)
                    ]
                    chunks = list(iter_message_chunks(rows, policy=self._chunk_policy))
                    for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
                        batch = chunks[start:start + _EMBED_BATCH_SIZE]
                        vectors = [v for result in self._embedder.embed_document_batches(
                            [c.text for c in batch]
                        ) for v in result.vectors]
                        if len(vectors) != len(batch):
                            raise ValueError("chunk embedding count mismatch")
                        for chunk, vector in zip(batch, vectors):
                            vector_store.record_chunk_embedding(
                                chunk.chunk_id, self._model, vector,
                                store_id=chunk.store_id, chunk_index=chunk.chunk_index,
                                char_start=chunk.char_start, char_end=chunk.char_end,
                                token_estimate=chunk.token_estimate, identity=chunk_identity,
                            )
                    summary_vector = self._embedder.embed_documents([summary_text])[0]
                    vector_store.record_embedding(
                        str(node_id), "summary", self._model, summary_vector, identity=identity
                    )
            finally:
                vector_store.close()
                dag.close()
                store.close()

            sessions[session_id] = {"order": order, "time": session_time}
            self._sessions_path(user_id).write_text(json.dumps(sessions), encoding="utf-8")

    # -- search ---------------------------------------------------------------

    def search(self, query, user_id, top_k):
        """Run ``lcm_recall`` and return one ``[event_time] text`` line per hit."""
        user_id = _validate_user_id(user_id)
        if not self._db_path(user_id).exists():
            return ""
        import hermes_lcm.tools as lcm_tools

        config, store, dag, vector_store = self._open(user_id)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _hermes_home=str(self._user_dir(user_id)),
                # Disjoint from every ingested session so the scope prior never
                # lifts an evidence session.
                current_session_id=f"__lcmx_omnimemeval_probe__{user_id}",
            )
            if self._embeddings:
                cache_key = (self._provider_name, str(self._embedder.model_id).strip())
                engine._lcm_embedding_provider_cache = (cache_key, self._embedder)
            args = {"query": query, "limit": min(int(top_k), self._top_k_cap)}
            if self._scope_bias:
                args["scope_bias"] = float(self._scope_bias)
            payload = json.loads(lcm_tools.lcm_recall(args, engine=engine))
            # Recall snippets are cut at 300 chars; deliver the full stored
            # message for message hits (LCMX_DELIVERY=snippet restores the cut).
            full_rows = {}
            if self._delivery == "full":
                ids = [h.get("store_id") for h in payload.get("hits", [])
                       if isinstance(h.get("store_id"), int)]
                full_rows = store.get_batch(ids)
        finally:
            vector_store.close()
            dag.close()
            store.close()

        if "error" in payload:
            raise RuntimeError(f"lcm_recall error: {payload['error']}")
        reason = payload.get("degraded_reason")
        if payload.get("degraded") and self._embeddings and reason not in _WARNED:
            # Logged once per distinct reason; short turns (< 40 tokens) carry
            # no chunk vectors by LCM-X design, so "chunk vectors are
            # unavailable" is expected on short-message stores.
            _WARNED.add(reason)
            print(f"[lcmx] degraded recall ({user_id}): {reason}", file=sys.stderr)

        sessions = self._load_sessions(user_id)
        lines = []
        for hit in payload.get("hits", [])[:top_k]:
            row = full_rows.get(hit.get("store_id")) if hit.get("kind") != "summary" else None
            if row and row.get("content"):
                text = " ".join(str(row["content"]).split())
            else:
                # Drop the FTS match-highlight markers; keep the stored text.
                snippet = str(hit.get("snippet") or "").replace(">>>", "").replace("<<<", "")
                text = " ".join(snippet.split())
            if not text:
                continue
            event_time = hit.get("event_time") or (
                sessions.get(str(hit.get("session_id"))) or {}
            ).get("time")
            prefix = f"[{event_time}] " if event_time else ""
            if hit.get("kind") == "summary":
                prefix += "(session summary) "
            lines.append(prefix + text)
        return "\n".join(lines)

    # -- delete ---------------------------------------------------------------

    def delete_all(self, user_id):
        user_id = _validate_user_id(user_id)
        with _user_lock(user_id):
            shutil.rmtree(self._user_dir(user_id), ignore_errors=True)

    def delete(self, user_id):
        user_id = _validate_user_id(user_id)
        self.delete_all(user_id)
