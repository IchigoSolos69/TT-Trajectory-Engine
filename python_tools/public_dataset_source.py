"""Resolve established public dataset videos without copying them or scanning seal.

The four known sequences retain their original roles. Moving a file into a
folder does not establish a new dataset split or label identity. Caller-supplied
paths are deliberate overrides; distinct implicit copies require such a choice.
"""
from pathlib import Path
import re


RAW_VIDEO_ROOT = Path(__file__).resolve().parent / "dataset" / "raw_videos"
PUBLIC_SEQUENCE_LOCATIONS = {
    "game_1": Path("training/world_cup/game_1.mp4"),
    "game_2": Path("training/world_cup/game_2.mp4"),
    "test_1": Path("validation/world_cup/test_1.mp4"),
    "test_2": Path("validation/world_cup/test_2.mp4"),
}


def _existing_file(path):
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Dataset video is not a regular file: {path}")
    return path.resolve(strict=True)


def resolve_public_dataset_video(sequence, raw_root=None, *, explicit_path=None):
    """Return an absolute original video path, or fail before decoder startup.

Known names check one canonical path and the legacy flat path. Other public
game/test sequence names retain flat-only compatibility. No recursive directory
scan, source decoding, hash calculation, or train/test reassignment occurs.
"""
    if explicit_path is not None:
        selected = _existing_file(Path(explicit_path).expanduser())
        if selected is None:
            raise FileNotFoundError(f"Explicit dataset video does not exist: {explicit_path}")
        return selected

    name = str(sequence).lower()
    if name.endswith(".mp4"):
        name = name[:-4]
    if not re.fullmatch(r"(?:game|test)_[1-9][0-9]*", name):
        raise ValueError("Use a public sequence basename or pass explicit_path")
    root = Path(raw_root if raw_root is not None else RAW_VIDEO_ROOT).expanduser().resolve()
    paths = []
    if name in PUBLIC_SEQUENCE_LOCATIONS:
        paths.append(root / PUBLIC_SEQUENCE_LOCATIONS[name])
    paths.append(root / f"{name}.mp4")
    existing = []
    for path in paths:
        selected = _existing_file(path)
        if selected is not None:
            if not selected.is_relative_to(root):
                raise ValueError(f"Implicit dataset path escapes raw_videos: {path}; pass explicit_path")
            existing.append(selected)
    if not existing:
        expected = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"Public dataset video {name} was not found; checked: {expected}")
    first = existing[0]
    if any(not first.samefile(candidate) for candidate in existing[1:]):
        raise ValueError(f"Ambiguous original video for {name}: {existing}; pass explicit_path")
    return first


def registered_public_dataset_videos(raw_root=None):
    """List available known sequences for the explicit public-data audit only."""
    result = []
    for name in PUBLIC_SEQUENCE_LOCATIONS:
        try:
            result.append(resolve_public_dataset_video(name, raw_root))
        except FileNotFoundError:
            # This iterator describes files present; the resolver remains strict.
            continue
    return result
