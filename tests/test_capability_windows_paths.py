from __future__ import annotations

from pathlib import PureWindowsPath
import unittest

from ttea.tournament.capability import _atomic_temp_path


class CapabilityWindowsPathTests(unittest.TestCase):
    def test_atomic_temp_path_avoids_real_round2_windows_max_path_regression(self):
        target = (
            PureWindowsPath(
                r"C:\Users\tarek\Projects\ttea\data\tournaments"
                r"\capability-20260915T111817.722524Z-c7b9a4c6"
                r"\capability"
                r"\2e76af34c304a0fff12fa086b1b353c966baf27ba623e665cbc51d8cd9699095"
            )
            / ("technical-" + ("a" * 64) + ".json")
        )

        old_temp = target.with_name(
            "." + target.name + "." + ("b" * 32) + ".tmp"
        )
        new_temp = _atomic_temp_path(target)

        self.assertLess(len(str(target)), 260)
        self.assertGreater(len(str(old_temp)), 260)
        self.assertLess(len(str(new_temp)), 260)
        self.assertEqual(new_temp.parent, target.parent)
        self.assertTrue(new_temp.name.startswith(".tmp-"))
        self.assertEqual(len(new_temp.name), 37)
        self.assertNotIn(target.name, new_temp.name)


if __name__ == "__main__":
    unittest.main()
