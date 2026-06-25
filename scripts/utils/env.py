"""Centralized env file loading for OmniMemEval.

Reads OMNIMEMEVAL_ENV_FILE env var (set by shell scripts via --env flag)
to determine which env file to load.

Usage in any Python script:
    from utils.env import load_env
    load_env()
"""

import os
import sys

from dotenv import load_dotenv


def load_env():
    """Load environment variables from the configured env file.

    Requires OMNIMEMEVAL_ENV_FILE to be set (by shell --env flag).
    """
    env_file = os.getenv("OMNIMEMEVAL_ENV_FILE")
    if not env_file:
        print("Error: OMNIMEMEVAL_ENV_FILE not set. Use --env <file> when running the eval script.", file=sys.stderr)
        sys.exit(1)
    load_dotenv(env_file, override=True)
