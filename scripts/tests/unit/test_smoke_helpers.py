"""Offline tests for the smoke client runner helpers."""

from datetime import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.integration.smoke_clients import (
    check_time_in_result,
    get_delete_fn,
    has_search_result,
)


class TestSmokeClientHelpers(unittest.TestCase):
    def test_check_time_in_result_matches_iso_date(self):
        found, detail = check_time_in_result(
            "Memory created on 2024-02-03 during the session.",
            [datetime(2024, 2, 3)],
        )

        self.assertTrue(found)
        self.assertIn("2024-02-03", detail)

    def test_check_time_in_result_reports_missing_date(self):
        found, detail = check_time_in_result("No date here.", [datetime(2024, 2, 3)])

        self.assertFalse(found)
        self.assertIn("2024-02-03", detail)

    def test_has_search_result_handles_common_shapes(self):
        self.assertFalse(has_search_result(""))
        self.assertFalse(has_search_result("Conversation memories:\n\n"))
        self.assertFalse(has_search_result({}))
        self.assertFalse(has_search_result([]))
        self.assertTrue(has_search_result("Alice went to Paris."))
        self.assertTrue(has_search_result({"memories": ["Alice went to Paris."]}))
        self.assertTrue(has_search_result(["Alice went to Paris."]))

    def test_get_delete_fn_prefers_delete(self):
        class Client:
            def delete(self, user_id):
                return f"delete:{user_id}"

            def delete_user(self, user_id):
                return f"delete_user:{user_id}"

        delete_fn = get_delete_fn(Client())

        self.assertEqual(delete_fn("u1"), "delete:u1")

    def test_get_delete_fn_supports_delete_all(self):
        class Client:
            def delete_all(self, user_id):
                return f"delete_all:{user_id}"

        delete_fn = get_delete_fn(Client())

        self.assertEqual(delete_fn("u1"), "delete_all:u1")


if __name__ == "__main__":
    unittest.main()
