"""Database configuration must be usable without a running service."""

import os
import unittest
from unittest.mock import patch

from core.database import DatabaseSettings


class DatabaseSettingsTests(unittest.TestCase):
    def test_reads_database_and_embedding_configuration(self):
        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://demo:example@localhost:5432/echomind",
            "EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5",
            "EMBEDDING_DIMENSION": "512",
        }, clear=True):
            settings = DatabaseSettings.from_env()
        self.assertEqual(settings.database_url, "postgresql://demo:example@localhost:5432/echomind")
        self.assertEqual(settings.embedding_model, "BAAI/bge-small-zh-v1.5")
        self.assertEqual(settings.embedding_dimension, 512)

    def test_rejects_zero_embedding_dimension(self):
        with patch.dict(os.environ, {"EMBEDDING_DIMENSION": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "EMBEDDING_DIMENSION"):
                DatabaseSettings.from_env()


if __name__ == "__main__":
    unittest.main()
