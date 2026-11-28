"""Every regular expression `pydantic_ai` ships scans repeated adversarial input in linear time.

`re` backtracks, so a pattern whose match attempts re-read the same input takes quadratic or exponential time on
input built to provoke it. Core patterns run on model output, tool arguments and message history, often on the event
loop.

The patterns are found and checked as `tests/harness/test_regex_linearity.py` says for `pydantic_ai_harness`: each is
run with the call the code makes, a `match` or `fullmatch` included. Each call whose pattern only exists at runtime
says where its pattern comes from in `RUNTIME_PATTERNS`, and the built-in tables those calls compile are checked
through `PATTERN_TABLES`. The tests after those test the scan itself: `tests/_regex_call_sites.py`, which finds the
calls, and `tests/_regex_steps.py`, which counts their steps.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from inline_snapshot import snapshot

import pydantic_ai
from pydantic_ai._framing import _BUILTIN_TAGS, _VOICE_TAG_OPENERS, _lookalike_frame, _tag_openers  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._griffe import _docstring_style_patterns  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._special_tokens import _JOINER_IN_TEXT, _SPECIAL_TOKEN_START, _TOKEN_TAIL_PREFIX  # pyright: ignore[reportPrivateUsage]

from ._regex_call_sites import CallSite, PackageScan, call_sites_in, hidden_re_uses
from ._regex_steps import Method, Shape, StepCounter, linearity_failure, shape

PATTERN_TABLES: dict[str, dict[str, tuple[str, int]]] = {
    '_docstring_style_patterns': {
        f'{style}:{replacement}': (pattern.format(replacement), re.IGNORECASE | re.MULTILINE)
        for pattern, replacements, style in _docstring_style_patterns
        for replacement in replacements
    },
    '_framing': {
        name: (compiled.pattern, compiled.flags)
        for name, compiled in {
            '_data_tag_openers': _tag_openers(*_BUILTIN_TAGS),
            '_VOICE_TAG_OPENERS': _VOICE_TAG_OPENERS,
            '_lookalike_frame': _lookalike_frame,
        }.items()
    },
    '_special_tokens': {
        name: (compiled.pattern, compiled.flags)
        for name, compiled in {
            '_SPECIAL_TOKEN_START': _SPECIAL_TOKEN_START,
            '_JOINER_IN_TEXT': _JOINER_IN_TEXT,
            '_TOKEN_TAIL_PREFIX': _TOKEN_TAIL_PREFIX,
        }.items()
    },
}
"""The built-in tables the runtime calls below compile: `_griffe`'s docstring style patterns, and the patterns
`_framing` and `_special_tokens` compile from their built-in tags and tokens, with the flags they compile them with.
`_data_tag_openers` is compiled here from the built-in tags alone, as it is before any `register_tag` call."""

RUNTIME_PATTERNS: dict[str, tuple[str, ...]] = {
    '_framing.py::_lookalike_pattern': (
        'an alternation of the escaped system-voice and data frame tags, and of any `add_system_voice_tag` added',
    ),
    '_framing.py::_tag_openers': (
        "an alternation of the escaped built-in tags, and of the user's `register_tag` tags",
    ),
    '_griffe.py::_infer_docstring_style': ('a `_docstring_style_patterns` entry, with each of its replacements',),
    '_special_tokens.py::_JOINER_IN_TEXT': ("an alternation built from the module's token table",),
    '_special_tokens.py::_SPECIAL_TOKEN_START': ("an alternation built from the module's token table",),
    '_special_tokens.py::_TOKEN_TAIL_PREFIX': ("an alternation built from the module's token table",),
}
"""Calls whose pattern only exists at runtime, keyed by `<file>::<function or module-level name>`: where each call's
pattern comes from, in source order. A built-in table they compile is checked through `PATTERN_TABLES`; a caller's
own pattern can't be."""

_SCAN = PackageScan(Path(pydantic_ai.__file__).parent)


@pytest.mark.parametrize(
    ('pattern', 'flags', 'method'),
    [pytest.param(*p, id=label) for label, p in _SCAN.checked_patterns(PATTERN_TABLES).items()],
)
def test_scanning_is_linear(pattern: str, flags: int, method: Method):
    assert linearity_failure(pattern, flags, method) is None


def test_every_runtime_pattern_says_where_it_comes_from():
    """Each call whose pattern only exists at runtime has an entry, and each entry still has its call."""
    assert _SCAN.unresolved() == {location: len(sources) for location, sources in RUNTIME_PATTERNS.items()}


def test_re_functions_are_only_called_directly():
    """A `from re import compile`, a renamed `re` or `re.compile` passed on would hide calls from the scan above."""
    assert _SCAN.hidden_re_uses() == []


_UNRESOLVED_SOURCE = """\
import re
from elsewhere import IMPORTED
T = {'a': 'x'}

def f(pattern, table, names):
    re.compile(pattern)
    [re.search(value, key) for key, value in table.items()]
    re.compile(IMPORTED)
    re.compile('|'.join(names))
    re.compile('|'.join(name for name in 'ab'))
    re.compile(f'{pattern!r}')
    re.compile(table()[0])
    re.compile(str(pattern))
    re.compile(T['missing'])
    re.compile(T['a' if pattern else 'b'])
    re.compile(T.pattern)
"""

_METHODS_SOURCE = """\
import re
_A = re.compile('a')
_B: re.Pattern[str] = re.compile('b')
_C = re.compile('c')
D = re.compile('d')
_E = re.compile('e')
_F = re.compile(_E.pattern + 'f')
_G = re.compile('g')
_H = re.compile('h')

def f(text):
    return _A.fullmatch(text) or _B.match(text, 1) or _B.fullmatch(text) or _C.match(text) or _C.sub('', text) or D.match(text) or _F.match(text) or use(_G) or _H.fullmatch(text, _H.groups)

re.fullmatch('i', 'x')
"""


@pytest.mark.parametrize(
    ('source', 'sites'),
    [
        pytest.param(
            "import re\n_A = re.compile(r'a+' 'b', re.I)\n",
            [CallSite('m.py::_A', [('a+b', re.IGNORECASE)])],
            id='module-constant',
        ),
        pytest.param(
            "import re\n_P = 'x'\n_N = ('a', 'b')\n_K = {'c': 1, 'd': 2}\n_A = re.compile(rf'{_P}(?:{\"|\".join(_N)})|' + '|'.join(_K))\n",
            [CallSite('m.py::_A', [('x(?:a|b)|c|d', 0)])],
            id='f-string-join-and-concatenation',
        ),
        pytest.param(
            "import re\nT: dict[str, tuple[str, str]] = {'j': ('t', 'b'), 'k': ('t', 'a*')}\n_A = re.compile(T['k'][1])\n",
            [CallSite('m.py::_A', [('a*', 0)])],
            id='table-entry',
        ),
        pytest.param(
            'import re\ndef f(command):\n    ps = [r"^a", r"^b"]\n    return any(re.match(p, command) for p in ps)\n',
            [CallSite('m.py::f', [('^a', 0), ('^b', 0)], ('match',))],
            id='comprehension-over-local-list',
        ),
        pytest.param(
            "import re\nclass C:\n    def f(self, word, strict):\n        return re.sub(rf'\\b{re.escape(word)}' if strict else re.escape(word), '', 'text', flags=re.I | re.M)\n",
            [CallSite('m.py::C.f', [('\\bx', re.I | re.M), ('x', re.I | re.M)])],
            id='escaped-runtime-text',
        ),
        pytest.param(
            "import re\n_A = re.compile(r'a(?!b)')\n_B = re.compile(rf'{_A.pattern}|c', re.I)\n",
            [CallSite('m.py::_A', [('a(?!b)', 0)]), CallSite('m.py::_B', [('a(?!b)|c', re.IGNORECASE)])],
            id='compiled-pattern',
        ),
        pytest.param(_UNRESOLVED_SOURCE, [CallSite('m.py::f', None)] * 11, id='runtime-patterns'),
        pytest.param(
            _METHODS_SOURCE,
            [
                CallSite('m.py::_A', [('a', 0)], ('fullmatch',)),
                CallSite('m.py::_B', [('b', 0)], ('fullmatch', 'match')),
                CallSite('m.py::_C', [('c', 0)], ('match', 'scan')),
                CallSite('m.py::D', [('d', 0)], ('match', 'scan')),
                CallSite('m.py::_E', [('e', 0)]),
                CallSite('m.py::_F', [('ef', 0)], ('match',)),
                CallSite('m.py::_G', [('g', 0)]),
                CallSite('m.py::_H', [('h', 0)], ('fullmatch', 'scan')),
                CallSite('m.py::', [('i', 0)], ('fullmatch',)),
            ],
            id='the-calls-each-pattern-runs-with',
        ),
    ],
)
def test_the_call_site_scan(source: str, sites: list[CallSite]):
    assert list(call_sites_in('m.py', ast.parse(source), source)) == sites


def test_a_pattern_another_module_names_runs_with_scan_too():
    """What another module does with a compiled pattern it imports isn't in this module, so any method may run it."""
    source = "import re\n_A = re.compile('a')\n_A.fullmatch('x')\n"

    assert list(call_sites_in('m.py', ast.parse(source), source, shared={'_A'})) == [
        CallSite('m.py::_A', [('a', 0)], ('fullmatch', 'scan'))
    ]


def test_a_parenthesized_re_is_scanned(tmp_path: Path):
    """The module filter looks for the name `re`, so a call through `(re)`, which formatting keeps, is still found."""
    (tmp_path / 'm.py').write_text("import re\n_A = (re).compile('a')\n")

    assert PackageScan(tmp_path).call_sites == [CallSite('m.py::_A', [('a', 0)])]


def test_the_hidden_use_scan():
    source = 'import re\nimport re as regexp\nfrom re import compile\nimport regex\n_re = re\nmap(re.compile, [])\nre.compile("a")\n'

    assert sorted(hidden_re_uses(ast.parse(source))) == snapshot(
        [
            '`re.compile` uncalled, at line 6',
            '`re` itself, at line 5',
            'from re import ...',
            'import re as regexp',
            'import regex',
        ]
    )


@pytest.mark.parametrize(
    ('pattern', 'failure'),
    [
        pytest.param(r'\w+@', snapshot("'0' * 36: over 1332 steps"), id='restart-in-a-run'),
        pytest.param(
            r'\bkey-[\w-]+\.',
            snapshot("'key-' * 6, 12, 24: [242, 914, 3554] steps"),
            id='restart-at-each-word-in-a-run',
        ),
        pytest.param(r'<.*?>', snapshot("'<' * 76: over 4928 steps"), id='lazy-to-a-missing-end'),
        pytest.param(r'\s*\s*x', snapshot("'\\t' * 19: over 1600 steps"), id='adjacent-repeats'),
        pytest.param(r'(a+)+b', snapshot("'a' * 18: over 1140 steps"), id='nested-repeats'),
        pytest.param(r'(?:a|a){0,20}b', snapshot("'a' * 37: over 13376 steps"), id='bounded-alternation'),
        # A one-character lookbehind only lets an attempt start after a character it admits, so its pump unit carries
        # one. `DEFAULT_VERIFICATION_COMMANDS[0]`, before its formatter alternative was split out, restarts at each
        # `black` of `  black  black`, and `blackblack` would never find that.
        pytest.param(
            r'(?<!\S)(?:(?:uv|poetry|pipenv|hatch|pdm)\s+run(?:\s+-\S+)*\s+)?(?:(?<!\S)(?:\S*/)?python3?\s+-m\s+)?'
            r'(?<!\S)(?:\S*/)?'
            r'(?:pytest|py\.test|unittest|tox|nox|mypy|pyright|basedpyright|ty\s+check|ruff\s+check'
            r'|(?:ruff\s+format|black|isort)(?=[^;&|\n]*\s--(?:check|diff)))\b',
            snapshot("'  black' * 10, 20, 40: [2358, 6973, 23103] steps"),
            id='restart-at-each-character-a-lookbehind-admits',
        ),
        pytest.param(
            r'(?<!\S)(?:ab)*(?:ab)*c',
            snapshot("'ab' * 11, 22, 44: [382, 1240, 4408] steps"),
            id='adjacent-repeats-after-a-lookbehind',
        ),
        pytest.param(r'(a*)\1b', snapshot("'a' * 18: over 1140 steps"), id='a-backreference-compares-each-split'),
    ],
)
def test_a_super_linear_pattern_fails(pattern: str, failure: str):
    """The check catches each way a backtracking pattern goes super-linear, with step counts that show it."""
    assert linearity_failure(pattern, 0) == failure


@pytest.mark.parametrize(
    ('pattern', 'method', 'failure'),
    [
        pytest.param(r'\w+@', 'match', None, id='one-attempt-reads-a-run-once'),
        pytest.param(
            r'(a+)+b',
            'match',
            snapshot("'a' * 18: over 1140 steps"),
            id='one-attempt-backtracks-through-nested-repeats',
        ),
        pytest.param(r'(?:a|aa)*', 'scan', None, id='a-scan-takes-the-first-match'),
        pytest.param(
            r'(?:a|aa)*',
            'fullmatch',
            snapshot("'a' * 18: over 912 steps"),
            id='a-full-match-backtracks-through-every-split',
        ),
    ],
)
def test_the_call_decides_what_backtracks(pattern: str, method: Method, failure: str | None):
    """`match` and `fullmatch` make one attempt where a scan makes one at each position, but a full match backtracks
    through every way to match the start of the text, where a scan takes the first."""
    assert linearity_failure(pattern, 0, method) == failure


@pytest.mark.parametrize(
    ('pattern', 'flags', 'text'),
    [
        pytest.param(r'ab|a[^b]', 0, 'abacad', id='branch-and-negated-literal'),
        pytest.param(r'a+', 0, 'aAa', id='case-sensitive'),
        pytest.param(r'[a-c]x|(?-i:[a-c])y|(?i:K)k', re.IGNORECASE, 'BXcx Ay ay kKKk', id='scoped-flags'),
        pytest.param(r'(?a:\w+)', 0, 'aé b', id='scoped-ascii'),
        pytest.param(r'.+', 0, 'a\nbc', id='any'),
        pytest.param(r'.+', re.DOTALL, 'a\nbc', id='dotall'),
        pytest.param(r'^a|a$|\Ab|b\Z', 0, 'ab\nab\n', id='anchors'),
        pytest.param(r'^a|a$', re.MULTILINE, 'ab\nba\na', id='multiline-anchors'),
        pytest.param(r'\bab\b', 0, 'ab cab ab', id='word-boundary'),
        pytest.param(r'(?<=a)b|(?<!c)d|x(?=y)|z(?!w)', 0, 'ab cd ad xy xz zw z', id='lookarounds'),
        pytest.param(r'a{2,3}?b|c*?d|e++e|f?+g', 0, 'aab aaab ccd eee fg', id='lazy-and-possessive'),
        pytest.param(r'(?>a+)b|(?:ab|a)+c|(?:x|)*y|(?:p|q)*?r|(?:m|n)++m', 0, 'aab ababc xxy pqr mnm', id='groups'),
        pytest.param(r'(?:ab){1,2}+', 0, 'ababab', id='possessive-group-at-its-bound'),
        pytest.param(r'x*', 0, 'axxb', id='empty-matches'),
        pytest.param(r'(?:a?){2,3}b', 0, 'b ab', id='empty-iterations'),
        pytest.param(r'(?:|a)*', 0, 'xaa', id='empty-first-alternative'),
        pytest.param(r'(?:ab)*?c|(?:|b)*?d', 0, 'ababc bd d', id='lazy-group-iterations'),
        pytest.param(r'(?:a?){2}+|(?:b?){0,3}+c', 0, 'x bbc', id='empty-possessive-iterations'),
        pytest.param(r'(["\'])(?:\\.|(?!\1).)*\1', 0, '"a\\"b" \'c"d\' "e', id='quoted-backreference'),
        pytest.param(r'(a)?b\1|(?:(c)|d)+\2', 0, 'bab aba b cdc dc', id='unset-and-repeated-groups'),
        pytest.param(
            r'(?:(?=(a))b|a)\1|(?:(?>(c))d|c)\2|(?:(e)*+f|e)\3', 0, 'aa cc ee', id='captures-a-failed-commit-undoes'
        ),
    ],
)
def test_the_step_counter_matches_like_re(pattern: str, flags: int, text: str):
    """The counter is only evidence if it backtracks the way `re` does: it finds the same matches.

    That holds for a scan of the whole text, and for a search, match or full match from each position in it.
    """
    counter, compiled = StepCounter(pattern, flags), re.compile(pattern, flags)
    assert counter.scan(text, budget=10_000).spans == [m.span() for m in compiled.finditer(text)]
    positions = range(len(text) + 1)
    assert [counter.search(text, 10_000, p).spans for p in positions] == [
        [m.span()] if (m := compiled.search(text, p)) else [] for p in positions
    ]
    assert [counter.match(text, 10_000, p).spans for p in positions] == [
        [m.span()] if (m := compiled.match(text, p)) else [] for p in positions
    ]
    assert [counter.fullmatch(text, 10_000, p).spans for p in positions] == [
        [m.span()] if (m := compiled.fullmatch(text, p)) else [] for p in positions
    ]


@pytest.mark.parametrize(
    ('pattern', 'flags', 'construct'),
    [
        pytest.param(r'(a)?(?(1)b|c)', 0, 'GROUPREF_EXISTS', id='conditional'),
        pytest.param(r'(a)\1', re.IGNORECASE, 'case-insensitive GROUPREF', id='case-insensitive-backreference'),
    ],
)
def test_the_step_counter_refuses_a_construct_it_does_not_model(pattern: str, flags: int, construct: str):
    with pytest.raises(NotImplementedError, match=construct):
        StepCounter(pattern, flags)


@pytest.mark.parametrize(
    ('pattern', 'expected'),
    [
        pytest.param(
            r'\beyJ[a-z_-]{2,}\.x',
            snapshot(
                Shape(units=['-', '.x', '_', 'a', 'eyJ', 'eyJ-', 'eyJ_', 'eyJa'], reach=7, unbounded=True, nodes=8)
            ),
            id='literal-before-a-repeat',
        ),
        pytest.param(
            r'\$\{?[A-Z]*(?:KEY|ID)',
            snapshot(Shape(units=['$', '$A', '$ID', '$KEY', '${', 'A', '{'], reach=6, unbounded=True, nodes=11)),
            id='optional-repeat-keeps-the-prefix',
        ),
        pytest.param(
            r'".*?"[^"]+[\w-]',
            snapshot(
                Shape(
                    units=['"', '""', '""0', '""A', '""a', '"0', '"A', '"a', '-', '0', 'A', 'a'],
                    reach=5,
                    unbounded=True,
                    nodes=7,
                )
            ),
            id='wildcards-and-categories',
        ),
        pytest.param(
            r'(?<!-)(?=\w)(?>ab|c)(?i:d){1,3}',
            snapshot(Shape(units=['-', 'a', 'aab', 'ab', 'ac', 'c', 'd'], reach=6, unbounded=False, nodes=12)),
            id='groups-and-lookarounds',
        ),
        pytest.param(
            r'(?<!\S)a(?<!\S)b',
            snapshot(Shape(units=[' a b', 'a', 'a b', 'ab'], reach=2, unbounded=False, nodes=6)),
            id='admitting-from-each-lookbehind-on',
        ),
    ],
)
def test_the_pump_units(pattern: str, expected: Shape):
    assert shape(pattern, 0) == expected
