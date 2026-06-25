"""Shared resource initialization for evaluation scripts.

The default network endpoints are the official upstream services. Users in
restricted networks can opt into mirrors through environment variables:

* ``HF_ENDPOINT`` for Hugging Face model downloads.
* ``OMNIMEMEVAL_NLTK_INDEX_URL`` for a custom NLTK index.xml.
* ``OMNIMEMEVAL_NLTK_GITHUB_PROXY`` for proxying NLTK package URLs.
"""

import os
import time

_sentence_model = None
_SENTENCE_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


_GITHUB_RAW = "https://raw.githubusercontent.com/"
_DEFAULT_NLTK_INDEX_URL = (
    _GITHUB_RAW + "nltk/nltk_data/gh-pages/index.xml"
)


def _with_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else value + "/"


def _download_nltk_resource(name):
    """Download a single NLTK package using official or configured endpoints."""
    import nltk

    index_url = os.environ.get("OMNIMEMEVAL_NLTK_INDEX_URL", "").strip()
    github_proxy = os.environ.get("OMNIMEMEVAL_NLTK_GITHUB_PROXY", "").strip()

    if not index_url and not github_proxy:
        nltk.download(name)
        return

    import nltk.downloader

    dl = nltk.downloader.Downloader(
        server_index_url=index_url or _DEFAULT_NLTK_INDEX_URL
    )
    if github_proxy:
        proxy = _with_trailing_slash(github_proxy)
        for pkg in dl.packages():
            if pkg.url and pkg.url.startswith(_GITHUB_RAW):
                pkg.url = proxy + pkg.url
    dl.download(name)


def ensure_nltk_resources():
    """Check and download required NLTK resources (wordnet + punkt_tab).
    """
    import nltk

    resources = [
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"  [NLTK] {name}: found in local cache")
        except LookupError:
            print(f"  [NLTK] {name}: missing, downloading...")
            _download_nltk_resource(name)
            print(f"  [NLTK] {name}: download complete")


def _check_model_cached(model_name: str) -> bool:
    """Check whether a HuggingFace model is already cached locally."""
    hf_home = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    hub_dir = os.path.join(
        hf_home,
        "hub",
        f"models--{model_name.replace('/', '--')}",
        "snapshots",
    )
    return os.path.isdir(hub_dir) and len(os.listdir(hub_dir)) > 0


def ensure_sentence_model():
    """Load SentenceTransformer model, downloading if needed.

    Returns the model instance, or None on failure.
    Uses the official Hugging Face endpoint unless ``HF_ENDPOINT`` is set.
    """
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model

    from sentence_transformers import SentenceTransformer

    is_cached = _check_model_cached(_SENTENCE_MODEL_NAME)

    if is_cached:
        print(
            f"  [SentenceTransformer] {_SENTENCE_MODEL_NAME}: "
            "found in local cache, loading...",
            flush=True,
        )
    else:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        print(
            f"  [SentenceTransformer] {_SENTENCE_MODEL_NAME}: "
            f"not cached, downloading from {endpoint} "
            "(about 1.2 GB)...",
            flush=True,
        )

    try:
        t0 = time.time()
        if is_cached:
            import huggingface_hub
            old_offline = huggingface_hub.constants.HF_HUB_OFFLINE
            huggingface_hub.constants.HF_HUB_OFFLINE = True
        try:
            _sentence_model = SentenceTransformer(
                _SENTENCE_MODEL_NAME,
                **({"local_files_only": True} if is_cached else {}),
            )
        finally:
            if is_cached:
                huggingface_hub.constants.HF_HUB_OFFLINE = old_offline
        elapsed = time.time() - t0
        print(f"  [SentenceTransformer] model loaded ({elapsed:.1f}s)", flush=True)
    except Exception as e:
        print(f"  [SentenceTransformer] model load failed: {e}", flush=True)
        _sentence_model = None

    return _sentence_model


def init_eval_resources():
    """One-call initialization for all eval resources.

    Only downloads NLTK data here. The SentenceTransformer model
    (~1.2 GB) is lazy-loaded on first use in calculate_semantic_similarity()
    so users who don't need ``--options semantic`` can skip the download.
    """
    print("=" * 60)
    print("Initializing evaluation resources...")
    print("=" * 60)
    ensure_nltk_resources()
    print("=" * 60)
    print("Evaluation resources initialized")
    print("=" * 60)
    print()
