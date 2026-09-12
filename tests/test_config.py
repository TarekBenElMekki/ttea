import logging
import os
import unittest

from ttea.config import load_settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.previous_level = os.environ.pop("TTEA_LOG_LEVEL", None)
        self.previous_directory = os.environ.pop("TTEA_LOG_DIRECTORY", None)

    def tearDown(self):
        if self.previous_level is not None:
            os.environ["TTEA_LOG_LEVEL"] = self.previous_level
        if self.previous_directory is not None:
            os.environ["TTEA_LOG_DIRECTORY"] = self.previous_directory

    def test_default_settings(self):
        settings = load_settings()

        self.assertEqual(settings.log_level, logging.INFO)
        self.assertEqual(str(settings.log_directory), "logs")

    def test_log_level_environment_override(self):
        os.environ["TTEA_LOG_LEVEL"] = "debug"

        self.assertEqual(load_settings().log_level, logging.DEBUG)
