"""Unit tests for the mem9 client helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.mem9_client import Mem9Client


class TestMem9Sanitization(unittest.TestCase):
    def test_loopback_url_is_neutralized(self):
        text = "Run it at http://localhost:5000 and test the endpoint."
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            "Run it at local address and test the endpoint.",
        )

    def test_ipv4_loopback_url_is_neutralized(self):
        text = "Debug on https://127.0.0.1:8000/api but keep the path."
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            "Debug on local address but keep the path.",
        )

    def test_public_url_is_neutralized(self):
        text = "Docs live at https://api.mem9.ai/healthz."
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            "Docs live at url.",
        )

    def test_html_tags_are_neutralized_without_tag_names(self):
        text = '<script>alert("x")</script><div class="card">ok</div>'
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            ' html markup alert("x") html markup html markup ok html markup ',
        )

    def test_code_fences_are_neutralized(self):
        text = "```html\n<div>`value`</div>\n```"
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            " code block html\n html markup 'value' html markup \n code block ",
        )

    def test_waf_tokens_are_neutralized(self):
        text = "Use javascript onclick and SVG script tags."
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            "Use js term event handler and vector graphic code term tags.",
        )

    def test_email_is_neutralized(self):
        text = "Contact root@example.com for details."
        self.assertEqual(
            Mem9Client._sanitize_content(text),
            "Contact email address for details.",
        )

    def test_none_content_becomes_empty_string(self):
        self.assertEqual(Mem9Client._sanitize_content(None), "")


if __name__ == "__main__":
    unittest.main()
