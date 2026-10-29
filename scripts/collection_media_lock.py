"""Serialize review media generation and diagnostic source cleanup."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def media_lock(output_dir: Path, *, blocking: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / '.collection_media.lock').open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
