# -*- coding: utf-8 -*-
"""Where tests put scratch files, and where the big built bundles live.

Seven test files hardcoded `D:/claude-tmp`, which exists on the Windows
machine and nowhere else, so those suites could not run on the Mac at
all. The three uses hiding behind that one string need three different
answers:

  - a directory the test CREATES      -> the system temp dir
  - a path the test only NAMES        -> anywhere valid, never written
  - the multi-GB bundles built once   -> a real location, or skip

`tempfile.gettempdir()` already honours TEMP/TMP, which on this machine
point at D: (C: was at 97%), so scratch work still lands on D: here
without the path being written down.
"""
import os
import tempfile


def scratch(*parts):
    """A path under the system temp dir.

    For directories a test creates, and for paths it merely names and
    must NOT write (a dry-run target, a 'does not exist' argument).
    """
    return os.path.join(tempfile.gettempdir(), *parts)


def artifacts():
    """Root of the big bundles built once and reused by several suites.

    These are real multi-GB artifacts, not scratch: the suites that need
    them already skip with a reason when they are absent, so the only
    job here is to find them without a machine's path in the source.

    CODELAB_TEST_ARTIFACTS wins. Failing that, the historical
    D:/claude-tmp is used WHEN IT ACTUALLY EXISTS, so the Windows
    machine keeps running the same suites it ran before this change;
    elsewhere the temp-based path simply will not contain them and the
    suites skip instead of dying on a path that cannot exist.
    """
    env = os.environ.get('CODELAB_TEST_ARTIFACTS')
    if env:
        return env
    if os.path.isdir('D:/claude-tmp'):
        return 'D:/claude-tmp'
    return os.path.join(tempfile.gettempdir(), 'codelab-test-artifacts')
