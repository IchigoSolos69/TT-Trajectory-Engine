"""Filesystem fixtures only: no footage decoding, camera, model or training."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import public_dataset_source as source
else:
    import public_dataset_source as source


class PublicDatasetSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="public-source-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "原本 videos"
        self.root.mkdir()

    def file(self, name, contents=b"fixture bytes"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_known_names_use_exact_domain_role_paths(self):
        for name, relative in source.PUBLIC_SEQUENCE_LOCATIONS.items():
            with self.subTest(name=name):
                path = self.file(relative)
                self.assertEqual(source.resolve_public_dataset_video(name, self.root), path.resolve())
                self.assertEqual(source.resolve_public_dataset_video(name.upper()+".MP4", self.root), path.resolve())

    def test_default_root_is_independent_of_working_directory(self):
        path = self.file("training/world_cup/game_1.mp4")
        previous = Path.cwd()
        elsewhere = Path(self.temp.name) / "elsewhere"
        elsewhere.mkdir()
        try:
            os.chdir(elsewhere)
            with patch.object(source, "RAW_VIDEO_ROOT", self.root):
                self.assertEqual(source.resolve_public_dataset_video("game_1"), path.resolve())
        finally:
            os.chdir(previous)

    def test_flat_legacy_fallback_preserved(self):
        path = self.file("test_1.mp4")
        self.assertEqual(source.resolve_public_dataset_video("test_1", self.root), path.resolve())

    def test_unknown_public_name_only_checks_legacy_flat(self):
        path = self.file("game_3.mp4")
        self.assertEqual(source.resolve_public_dataset_video("game_3", self.root), path.resolve())
        self.file("seal/world_cup/test_3.mp4")
        with self.assertRaises(FileNotFoundError):
            source.resolve_public_dataset_video("test_3", self.root)

    def test_missing_implicit_and_explicit_sources_fail(self):
        with self.assertRaises(FileNotFoundError):
            source.resolve_public_dataset_video("game_1", self.root)
        with self.assertRaises(FileNotFoundError):
            source.resolve_public_dataset_video("game_1", self.root, explicit_path=self.root/"missing.mp4")

    def test_different_existing_copies_are_ambiguous(self):
        self.file("game_1.mp4", b"legacy bytes")
        self.file("training/world_cup/game_1.mp4", b"new bytes")
        with self.assertRaisesRegex(ValueError, "Ambiguous original"):
            source.resolve_public_dataset_video("game_1", self.root)

    def test_byte_identical_distinct_copies_still_need_explicit_choice(self):
        self.file("game_1.mp4")
        self.file("training/world_cup/game_1.mp4")
        with self.assertRaisesRegex(ValueError, "Ambiguous original"):
            source.resolve_public_dataset_video("game_1", self.root)

    def test_hardlink_alias_is_same_original_identity(self):
        canonical = self.file("training/world_cup/game_1.mp4")
        os.link(canonical, self.root / "game_1.mp4")
        self.assertEqual(source.resolve_public_dataset_video("game_1", self.root), canonical.resolve())

    def test_explicit_same_name_override_preserves_caller_choice(self):
        self.file("game_1.mp4", b"legacy bytes")
        self.file("training/world_cup/game_1.mp4", b"canonical bytes")
        explicit = self.file("other venue/game_1.mp4", b"chosen original")
        self.assertEqual(source.resolve_public_dataset_video("game_1", self.root, explicit_path=explicit), explicit.resolve())

    def test_invalid_basename_and_directory_are_rejected(self):
        for name in ("../game_1", "seal/world_cup/game_1.mp4", "game_1.mov", "game_0", "notes"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                source.resolve_public_dataset_video(name, self.root)
        (self.root / "game_1.mp4").mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            source.resolve_public_dataset_video("game_1", self.root)

    def test_no_seal_search_and_registry_iteration_is_available_files_only(self):
        path = self.file("validation/world_cup/test_1.mp4")
        self.file("seal/world_cup/game_2.mp4", b"sealed poison")
        with patch.object(Path, "rglob", side_effect=AssertionError("No recursive scan")), patch.object(Path, "glob", side_effect=AssertionError("No directory scan")):
            self.assertEqual(source.registered_public_dataset_videos(self.root), [path.resolve()])


if __name__ == "__main__":
    unittest.main()
