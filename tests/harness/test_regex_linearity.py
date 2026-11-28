"""Every regular expression `pydantic_ai_harness` ships scans repeated adversarial input in linear time.

`re` backtracks, so a pattern whose match attempts re-read the same input, from each offset of a run or once per
way of splitting it, takes quadratic or exponential time on input built to provoke it. A harness pattern often runs
on text the model or a tool produced, and it runs on the event loop.

`tests/_regex_call_sites.py` finds the patterns in the package source, so a new one is checked without being listed
here, and says which call each one runs with. Each call whose pattern only exists at runtime says where its pattern
comes from in `RUNTIME_PATTERNS`, and the built-in tables those calls compile are checked through `PATTERN_TABLES`.

Each pattern is then run by `tests/_regex_steps.py`, which counts the steps `re`'s backtracking takes rather than
timing it. `linearity_failure` says what it asserts, on inputs built from each pattern's own literals and repeats: a
pass shows those inputs scan in linear time, not that every input does. A failure names the repeated unit and the
step counts. The usual fix keeps a failed attempt from restarting inside the run it already read, with a lookbehind
such as `(?<![\\w-])` before the repeat, or bounds what the repeat can cross. `tests/test_regex_linearity.py` checks
`pydantic_ai` the same way, and tests the scan itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pydantic_ai_harness
from pydantic_ai_harness.artifact_scanning import DEFAULT_ARTIFACT_PATTERNS
from pydantic_ai_harness.guardrails.detectors import DEFAULT_PII_PATTERNS, DEFAULT_SECRET_PATTERNS
from pydantic_ai_harness.require_verification import DEFAULT_EDIT_COMMANDS, DEFAULT_VERIFICATION_COMMANDS
from tests._regex_call_sites import PackageScan
from tests._regex_steps import GROWTH_LIMIT, Method, StepCounter, linearity_failure

PATTERN_TABLES: dict[str, dict[str, tuple[str, int]]] = {
    'DEFAULT_SECRET_PATTERNS': {name: (pattern, 0) for name, pattern in DEFAULT_SECRET_PATTERNS.items()},
    'DEFAULT_PII_PATTERNS': {name: (pattern, 0) for name, pattern in DEFAULT_PII_PATTERNS.items()},
    'DEFAULT_ARTIFACT_PATTERNS': {
        name: (pattern, re.IGNORECASE) for name, (_, pattern) in DEFAULT_ARTIFACT_PATTERNS.items()
    },
    'DEFAULT_VERIFICATION_COMMANDS': {
        str(index): (pattern, 0) for index, pattern in enumerate(DEFAULT_VERIFICATION_COMMANDS)
    },
    'DEFAULT_EDIT_COMMANDS': {str(index): (pattern, 0) for index, pattern in enumerate(DEFAULT_EDIT_COMMANDS)},
}
"""Every `DEFAULT_*_PATTERNS` table the package defines, and the default command patterns `RequireVerification`
compiles, with the flags the package compiles them with."""

RUNTIME_PATTERNS: dict[str, tuple[str, ...]] = {
    'artifact_scanning/_scanner.py::ArtifactScanner.__post_init__': (
        'a `DEFAULT_ARTIFACT_PATTERNS` entry',
        "the user's `extra_patterns`",
    ),
    'browser_use/_toolset.py::BrowserUseToolset.__init__': ('an alternation of escaped secret placeholders',),
    'file_system/_toolset.py::FileSystemToolset._search_files': ("the model's search pattern",),
    'guardrails/detectors.py::_compile': (
        "a `DEFAULT_SECRET_PATTERNS` or `DEFAULT_PII_PATTERNS` entry, or the user's `extra`",
    ),
    'permissions/_capability.py::Permissions.__post_init__': ("the user's `ask_commands`",),
    'playwright/_toolset.py::_without_endpoint_credentials': (
        'an alternation of the escaped `cdp_url` credential values',
    ),
    'require_verification/_capability.py::RequireVerification.__post_init__': (
        "a `DEFAULT_VERIFICATION_COMMANDS` entry, or the user's `verification_commands`",
        "a `DEFAULT_EDIT_COMMANDS` entry, or the user's `edit_commands`",
    ),
}
"""Calls whose pattern only exists at runtime, keyed by `<file>::<function>`: where each call's pattern comes from, in
source order. A built-in table they compile is checked through `PATTERN_TABLES`; a caller's own pattern can't be."""


_SCAN = PackageScan(Path(pydantic_ai_harness.__file__).parent)


@pytest.mark.parametrize(
    ('pattern', 'flags', 'method'),
    [pytest.param(*p, id=label) for label, p in _SCAN.checked_patterns(PATTERN_TABLES).items()],
)
def test_scanning_is_linear(pattern: str, flags: int, method: Method):
    assert linearity_failure(pattern, flags, method) is None


@pytest.mark.parametrize(
    ('prefix', 'repeated', 'suffix'),
    [
        pytest.param('', 'black/', 'black x', id='tool-path'),
        pytest.param('uv run ', '-a/black ', 'x', id='runner-flags'),
    ],
)
def test_the_default_formatter_check_is_linear(prefix: str, repeated: str, suffix: str):
    """`DEFAULT_VERIFICATION_COMMANDS[0]` stays linear on two formatter commands `linearity_failure` doesn't build.

    A tool name repeated through a path or through runner flags gives the formatter alternative's prefix a place to
    end at each `/` or flag, and the `--check` lookahead must not re-read the rest of the step from each one.
    """
    counter = StepCounter(DEFAULT_VERIFICATION_COMMANDS[0])
    steps = [counter.scan(prefix + repeated * n + suffix, budget=1_000_000).steps for n in (32, 64, 128)]

    assert steps[2] - steps[1] <= GROWTH_LIMIT * (steps[1] - steps[0])


def test_every_runtime_pattern_says_where_it_comes_from():
    """Each call whose pattern only exists at runtime has an entry, and each entry still has its call."""
    assert _SCAN.unresolved() == {location: len(sources) for location, sources in RUNTIME_PATTERNS.items()}


def test_every_pattern_table_is_checked():
    """A new table reaches `re` through a call `RUNTIME_PATTERNS` already lists, so it has to be listed by name."""
    assert _SCAN.default_tables() == sorted(name for name in PATTERN_TABLES if name.endswith('_PATTERNS'))


def test_re_functions_are_only_called_directly():
    """A `from re import compile`, a renamed `re` or `re.compile` passed on would hide calls from the scan above."""
    assert _SCAN.hidden_re_uses() == []
