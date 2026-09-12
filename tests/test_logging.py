import unittest
from pathlib import Path

from ttea.config import Settings
from ttea.logging import get_logger, setup_logging


class LoggingTests(unittest.TestCase):
    def test_logging_creates_and_persists_message(self):
        log_path = setup_logging(
            Settings(log_directory=Path("logs") / "test_runtime_logs", log_level=20)
        )
        logger = get_logger("test_logging")
        message = "persistent logging test message"

        logger.info(message)

        self.assertTrue(log_path.parent.is_dir())
        self.assertTrue(log_path.is_file())
        self.assertIn(message, log_path.read_text(encoding="utf-8"))
