"""Recoverable multi-file publication; direct byte comparisons, no fingerprints.

A pending journal blocks new publications. Recovery restores the previous batch.
An abandoned lock is never removed based merely on age.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile

JOURNAL = '.publication-journal.json'


def inherit_project_permissions(path):
    """Use destination-directory access rights before exposing a staged file."""
    if os.name != 'nt':
        return
    result = subprocess.run(['icacls.exe', str(path), '/inheritance:e'],
                            capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise OSError('Cannot enable project permission inheritance: ' + str(path))


def local(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Publication path outside project')
    return path


def capture(root, names):
    return {str(name): local(root, name).read_bytes() if local(root, name).exists() else None for name in names}


def assert_sources(root, expected):
    for name, value in expected.items():
        path = local(root, name)
        actual = path.read_bytes() if path.exists() else None
        if actual != value:
            raise ValueError('Source changed since review: ' + name)


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='.publication-', suffix='.tmp', dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if temporary.read_bytes() != data:
            raise ValueError('Temporary publication bytes differ')
        # mkstemp can create a protected owner-only ACL even in the destination
        # directory. Fix that ACL before replacement; a failure leaves the old
        # file intact, including during recovery.
        inherit_project_permissions(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(root, name='.monthly-pipeline.lock'):
    path = local(root, name)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, json.dumps({'pid': os.getpid()}).encode('utf-8'))
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def require_clean(root):
    if local(root, JOURNAL).exists():
        raise ValueError('Unfinished publication; restore previous batch before continuing')


def _restore(root, previous):
    for name, text in previous.items():
        path = local(root, name)
        if text is None:
            path.unlink(missing_ok=True)
        else:
            atomic(path, base64.b64decode(text, validate=True))


def prepare_rollback(root, names, expected):
    """Caller owns the lock; also supports legacy writers that need staged reads."""
    require_clean(root)
    assert_sources(root, expected)
    for name in names:
        if name not in expected:
            raise ValueError('Every output needs an expected previous value: ' + name)
        local(root, name)
    previous = {name: None if expected[name] is None else base64.b64encode(expected[name]).decode('ascii') for name in names}
    journal = local(root, JOURNAL)
    atomic(journal, json.dumps({'version': 1, 'previous': previous}, ensure_ascii=False).encode('utf-8'))
    return journal, previous


def commit_locked(root, outputs, expected, *, after_write=None):
    """Caller owns publication lock. Persist rollback data before the first write."""
    journal, previous = prepare_rollback(root, outputs, expected)
    try:
        for index, (name, data) in enumerate(outputs.items()):
            path = local(root, name)
            if data is None:
                path.unlink(missing_ok=True)
            else:
                atomic(path, data)
            if after_write: after_write(index)
        assert_sources(root, outputs)
    except Exception:
        _restore(root, previous)
        journal.unlink()
        raise
    # A hard termination before this point leaves rollback evidence; no new
    # publication or local HTTP serving may silently treat it as complete.
    journal.unlink()


def commit_files(root, outputs, expected, **kwargs):
    with exclusive_lock(root):
        commit_locked(root, outputs, expected, **kwargs)


def recover(root):
    with exclusive_lock(root):
        journal = local(root, JOURNAL)
        if not journal.exists(): return False
        record = json.loads(journal.read_text(encoding='utf-8'))
        if record.get('version') != 1 or not isinstance(record.get('previous'), dict):
            raise ValueError('Invalid publication recovery record')
        for name in record['previous']: local(root, name)
        _restore(root, record['previous'])
        journal.unlink()
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Restore an interrupted publication; first confirm no publisher owns its lock')
    parser.add_argument('--recover', action='store_true', required=True)
    parser.parse_args()
    print('RESTORED' if recover(Path(__file__).resolve().parents[1]) else 'NO_PENDING_PUBLICATION')
