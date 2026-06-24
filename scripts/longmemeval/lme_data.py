import copy
from pathlib import Path

import pandas as pd


LME_DATA_PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")
DISALLOWED_SPECIAL_TOKENS = ("<|endoftext|>",)


def sanitize_lme_message_content(content):
    """Normalize LME message text before it reaches any memory client."""
    if content is None:
        return ""

    text = str(content)
    for token in DISALLOWED_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def _sanitize_session(session):
    sanitized = []
    changed = 0
    for message in session:
        cleaned_message = copy.deepcopy(message)
        original = cleaned_message.get("content")
        cleaned = sanitize_lme_message_content(original)
        if cleaned != original:
            changed += 1
        cleaned_message["content"] = cleaned
        sanitized.append(cleaned_message)
    return sanitized, changed


def sanitize_lme_dataframe(df):
    """Return a copy of the LME dataframe with known bad tokens removed."""
    sanitized_df = df.copy(deep=True)
    cleaned_messages = 0

    for row_idx, sessions in sanitized_df["haystack_sessions"].items():
        sanitized_sessions = []
        for session in sessions:
            sanitized_session, changed = _sanitize_session(session)
            sanitized_sessions.append(sanitized_session)
            cleaned_messages += changed
        sanitized_df.at[row_idx, "haystack_sessions"] = sanitized_sessions

    return sanitized_df, cleaned_messages


def load_lme_dataframe(path=LME_DATA_PATH, *, verbose=True):
    """Load LongMemEval data through the shared sanitization path."""
    df = pd.read_json(path)
    df, cleaned_messages = sanitize_lme_dataframe(df)

    if verbose:
        print(f"📚 Loaded LongMemeval dataset from {path}")
        if cleaned_messages:
            print(f"🧼 Sanitized {cleaned_messages} LME message(s) with disallowed special token(s)")

    return df
