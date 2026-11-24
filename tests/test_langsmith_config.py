import os
import unittest
from unittest.mock import patch

from app.configs.config import Settings, langsmithconfig


class LangSmithConfigTests(unittest.TestCase):
    def test_missing_api_key_clears_environment_without_failing(self):
        config = Settings(
            ENABLE_OPENAI=True,
            OPENAI_API_KEY="test-key",
            LANGSMITH_API_KEY=None,
            LANGSMITH_PROJECT="test-project",
            LANGSMITH_TRACING=False,
        )

        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "stale-key"}):
            langsmithconfig(config)

            self.assertNotIn("LANGSMITH_API_KEY", os.environ)
            self.assertEqual(os.environ["LANGSMITH_PROJECT"], "test-project")
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")


if __name__ == "__main__":
    unittest.main()
