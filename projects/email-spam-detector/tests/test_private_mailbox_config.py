"""Tests for the private mailbox TOML loader."""

import tempfile
import unittest
from pathlib import Path

import tomllib
from spam_detector.private_mailbox_config import load_private_mailboxes


class PrivateMailboxConfigTests(unittest.TestCase):
    def write_config(self, directory: Path, content: str) -> Path:
        config_path = directory / "mailboxes.local.toml"
        config_path.write_text(content, encoding="utf-8")
        return config_path

    def test_order_labels_and_path_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            relative_mailbox = root / "mailbox-a"
            relative_mailbox.touch()
            absolute_mailbox = root / "mailbox-b"
            absolute_mailbox.touch()

            config_path = self.write_config(
                root,
                "\n".join(
                    [
                        "[[mailboxes]]",
                        'path = "mailbox-a"',
                        'label = "ham"',
                        'source = "SOURCE_A"',
                        "",
                        "[[mailboxes]]",
                        f'path = "{absolute_mailbox.as_posix()}"',
                        'label = "spam"',
                        'source = "SOURCE_B"',
                    ]
                ),
            )

            mailboxes = load_private_mailboxes(config_path)

            self.assertEqual(
                [mailbox.source for mailbox in mailboxes],
                ["SOURCE_A", "SOURCE_B"],
            )
            self.assertEqual(
                [mailbox.numeric_label for mailbox in mailboxes],
                [0, 1],
            )
            self.assertEqual(mailboxes[0].path, relative_mailbox.resolve())
            self.assertEqual(mailboxes[1].path, absolute_mailbox.resolve())
            self.assertTrue(mailboxes[1].path.is_absolute())

    def test_duplicate_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mailbox = root / "mailbox"
            mailbox.touch()
            config_path = self.write_config(
                root,
                "\n".join(
                    [
                        "[[mailboxes]]",
                        'path = "mailbox"',
                        'label = "ham"',
                        'source = "SOURCE_A"',
                        "",
                        "[[mailboxes]]",
                        'path = "mailbox"',
                        'label = "spam"',
                        'source = "SOURCE_A"',
                    ]
                ),
            )

            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_private_mailboxes(config_path)

    def test_invalid_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mailbox = root / "mailbox"
            mailbox.touch()
            config_path = self.write_config(
                root,
                "\n".join(
                    [
                        "[[mailboxes]]",
                        'path = "mailbox"',
                        'label = "unknown"',
                        'source = "SOURCE_A"',
                    ]
                ),
            )

            with self.assertRaisesRegex(ValueError, "exactly"):
                load_private_mailboxes(config_path)

    def test_missing_required_fields_are_rejected(self):
        templates = {
            "path": [
                "[[mailboxes]]",
                'label = "ham"',
                'source = "SOURCE_A"',
            ],
            "label": [
                "[[mailboxes]]",
                'path = "mailbox"',
                'source = "SOURCE_A"',
            ],
            "source": [
                "[[mailboxes]]",
                'path = "mailbox"',
                'label = "ham"',
            ],
        }

        for missing_field, lines in templates.items():
            with self.subTest(missing_field=missing_field):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    config_path = self.write_config(root, "\n".join(lines))

                    with self.assertRaisesRegex(
                        ValueError,
                        missing_field,
                    ):
                        load_private_mailboxes(
                            config_path,
                            validate_paths=False,
                        )

    def test_nonexistent_path_is_rejected_at_runtime(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.write_config(
                root,
                "\n".join(
                    [
                        "[[mailboxes]]",
                        'path = "missing-mailbox"',
                        'label = "ham"',
                        'source = "SOURCE_A"',
                    ]
                ),
            )

            with self.assertRaises(FileNotFoundError):
                load_private_mailboxes(config_path)

    def test_example_configuration_contains_only_placeholders(self):
        example_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "private_mailboxes.example.toml"
        )
        with example_path.open("rb") as file:
            config = tomllib.load(file)

        self.assertTrue(config["mailboxes"])
        for mailbox in config["mailboxes"]:
            self.assertRegex(mailbox["path"], r"^<PRIVATE_[A-Z_]+>$")
            self.assertRegex(mailbox["source"], r"^<PRIVATE_[A-Z_]+>$")
            self.assertIn(mailbox["label"], {"ham", "spam"})


if __name__ == "__main__":
    unittest.main()
