"""Regression: LoCoMo image captions reach the Zep graph and Graphiti ingestion.

LoCoMo stores the description of a shared photo under the singular key
``blip_caption`` (1,226 of 5,882 turns in locomo10.json; no turn carries a
plural ``blip_captions``). Zep graph mode and Graphiti read the plural key,
so the description never reached either store. These tests use one turn in
the dataset's own shape and check that its caption is ingested.
"""

import unittest
from unittest.mock import Mock

# One LoCoMo turn with a photo, in the shape of locomo10.json.
PHOTO_TURN = {
    "speaker": "Melanie",
    "dia_id": "D1:5",
    "text": "Our latest work from last weekend!",
    "img_url": ["https://example.invalid/photo.jpg"],
    "blip_caption": "a photo of a sunset with a palm tree",
    "query": "sunset palm tree",
}
PLAIN_TURN = {"speaker": "Caroline", "dia_id": "D1:6", "text": "Lovely."}
SESSION = [PHOTO_TURN, PLAIN_TURN]

META = {
    "session_date": "1:56 pm on 8 May, 2023",
    "conv_idx": 0,
    "session_key": "session_1",
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "speaker_a_user_id": "u_a",
    "speaker_b_user_id": "u_b",
}
CAPTION_TEXT = " (description of attached image: a photo of a sunset with a palm tree)"


class TestLocomoImageCaptions(unittest.TestCase):
    def test_zep_graph_mode_keeps_caption(self):
        from scripts.locomo.locomo_ingestion import ingest_session

        client = Mock()
        ingest_session(client, SESSION, "zep", dict(META, group_id="g_0"))
        data = [c.kwargs["data"] for c in client.sdk_graph_add.call_args_list]
        self.assertEqual(
            data,
            [
                "Melanie: Our latest work from last weekend!" + CAPTION_TEXT,
                "Caroline: Lovely.",
            ],
        )

    def test_graphiti_transcript_keeps_caption(self):
        from scripts.locomo.locomo_ingestion import ingest_session

        client = Mock()
        ingest_session(client, SESSION, "graphiti", dict(META))
        transcripts = {c.kwargs["raw_content"] for c in client.add.call_args_list}
        self.assertEqual(
            transcripts,
            {
                "Melanie: Our latest work from last weekend!"
                + CAPTION_TEXT
                + "\nCaroline: Lovely."
            },
        )


if __name__ == "__main__":
    unittest.main()
