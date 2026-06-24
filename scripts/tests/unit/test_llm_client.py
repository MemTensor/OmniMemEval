import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.llm_client import (  # noqa: E402
    LLMRetryConfig,
    _TrackedAsyncCompletions,
    _TrackedSyncCompletions,
    get_llm_retry_config,
)
from utils.token_tracker import TokenTracker  # noqa: E402


class DummyTracker:
    def record(self, *_args, **_kwargs):
        pass


class RetryableError(Exception):
    status_code = 500


class NonRetryableError(Exception):
    status_code = 400


class FakeResponse:
    usage = None
    model = "fake-model"


class FakeUsage:
    prompt_tokens = 11
    completion_tokens = 3
    total_tokens = 14


class FakePartialUsage:
    prompt_tokens = None
    completion_tokens = 3
    total_tokens = 3


class FakeUsageResponse:
    usage = FakeUsage()
    model = "fake-model"


class FakePartialUsageResponse:
    usage = FakePartialUsage()
    model = "fake-model"


class FakeSyncCompletions:
    def __init__(self, errors, response=None):
        self.errors = list(errors)
        self.calls = 0
        self.response = response or FakeResponse()

    def create(self, *args, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.response


class FakeAsyncCompletions:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    async def create(self, *args, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return FakeResponse()


def _config(max_retries=2):
    return LLMRetryConfig(
        module="ANSWER",
        max_retries=max_retries,
        retry_base_seconds=0.0,
        retry_max_seconds=0.0,
        timeout_seconds=10.0,
    )


class TestLLMClientRetry(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_sync_completion_retries_transient_errors(self):
        completions = FakeSyncCompletions([RetryableError("server error")])
        tracked = _TrackedSyncCompletions(completions, DummyTracker(), _config())

        self.assertIsInstance(tracked.create(model="m", messages=[]), FakeResponse)
        self.assertEqual(completions.calls, 2)

    def test_sync_completion_does_not_retry_non_transient_errors(self):
        completions = FakeSyncCompletions([NonRetryableError("bad request")])
        tracked = _TrackedSyncCompletions(completions, DummyTracker(), _config())

        with self.assertRaises(NonRetryableError):
            tracked.create(model="m", messages=[])
        self.assertEqual(completions.calls, 1)

    def test_sync_completion_records_provider_usage_when_available(self):
        tracker = TokenTracker()
        completions = FakeSyncCompletions([], response=FakeUsageResponse())
        tracked = _TrackedSyncCompletions(completions, tracker, _config())

        tracked.create(model="m", messages=[{"role": "user", "content": "hello"}])
        answer = tracker.summary()["modules"]["ANSWER"]

        self.assertEqual(answer["call_count"], 1)
        self.assertEqual(answer["prompt_tokens"], 11)
        self.assertEqual(answer["usage_reported_call_count"], 1)
        self.assertEqual(answer["estimated_prompt_call_count"], 0)

    def test_sync_completion_estimates_prompt_tokens_when_usage_is_partial(self):
        tracker = TokenTracker()
        completions = FakeSyncCompletions([], response=FakePartialUsageResponse())
        tracked = _TrackedSyncCompletions(completions, tracker, _config())

        tracked.create(model="m", messages=[{"role": "user", "content": "hello memory context"}])
        answer = tracker.summary()["modules"]["ANSWER"]

        self.assertEqual(answer["call_count"], 1)
        self.assertGreater(answer["prompt_tokens"], 0)
        self.assertEqual(answer["completion_tokens"], 3)
        self.assertEqual(answer["usage_reported_call_count"], 1)
        self.assertEqual(answer["estimated_prompt_call_count"], 1)
        self.assertEqual(answer["estimated_prompt_tokens"], answer["prompt_tokens"])

    def test_sync_completion_estimates_prompt_tokens_when_usage_missing(self):
        tracker = TokenTracker()
        completions = FakeSyncCompletions([])
        tracked = _TrackedSyncCompletions(completions, tracker, _config())

        tracked.create(
            model="m",
            messages=[
                {"role": "system", "content": "Answer with facts."},
                {"role": "user", "content": "Question plus retrieved memories."},
            ],
        )
        answer = tracker.summary()["modules"]["ANSWER"]

        self.assertEqual(answer["call_count"], 1)
        self.assertGreater(answer["prompt_tokens"], 0)
        self.assertEqual(answer["usage_reported_call_count"], 0)
        self.assertEqual(answer["estimated_prompt_call_count"], 1)
        self.assertEqual(answer["estimated_prompt_tokens"], answer["prompt_tokens"])

    def test_async_completion_retries_transient_errors(self):
        async def run():
            completions = FakeAsyncCompletions([RetryableError("timeout")])
            tracked = _TrackedAsyncCompletions(completions, DummyTracker(), _config())
            response = await tracked.create(model="m", messages=[])
            return response, completions.calls

        response, calls = asyncio.run(run())
        self.assertIsInstance(response, FakeResponse)
        self.assertEqual(calls, 2)

    def test_module_specific_retry_config_overrides_global_defaults(self):
        os.environ["LLM_MAX_RETRIES"] = "3"
        os.environ["ANSWER_MAX_RETRIES"] = "1"
        os.environ["LLM_TIMEOUT_SECONDS"] = "120"
        os.environ["ANSWER_TIMEOUT_SECONDS"] = "30"

        config = get_llm_retry_config("ANSWER")

        self.assertEqual(config.max_retries, 1)
        self.assertEqual(config.timeout_seconds, 30.0)

        eval_config = get_llm_retry_config("EVAL")
        self.assertEqual(eval_config.max_retries, 3)
        self.assertEqual(eval_config.timeout_seconds, 120.0)


if __name__ == "__main__":
    unittest.main()
