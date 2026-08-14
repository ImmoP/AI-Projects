"""Basic package-structure regression tests."""

import importlib
import unittest

from spam_detector.paths import DATA_DIR, PROJECT_ROOT

MODULES = [
    "spam_detector.paths",
    "spam_detector.private_mailbox_config",
    "spam_detector.data_processing.extract",
    "spam_detector.data_processing.prepare_private_data",
    "spam_detector.data_processing.prepare_hf_data",
    "spam_detector.data_processing.split_private_data",
    "spam_detector.data_processing.build_combined_splits",
    "spam_detector.data_processing.clean_split_leakage",
    "spam_detector.inspection.inspect_private_data",
    "spam_detector.inspection.inspect_private_dates",
    "spam_detector.inspection.inspect_hf_data",
    "spam_detector.inspection.check_split_leakage",
    "spam_detector.model",
]


class PackageTests(unittest.TestCase):
    def test_data_dir_is_root_level_data_directory(self):
        self.assertEqual(DATA_DIR, PROJECT_ROOT / "data")
        self.assertTrue(DATA_DIR.is_dir())

    def test_all_modules_import(self):
        for module_name in MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
