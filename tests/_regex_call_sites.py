"""Find every call to a `re` function in a package's source, and the patterns and flags each call can compile.

The source is read rather than imported, so a new pattern is found without being listed. A call's pattern is worked
out from the literals, f-strings, `'|'.join(...)` calls, `.pattern` of a compiled pattern, and names bound to them in
its module or function. `re.escape(...)` of runtime text stands in as `x`: escaped text matches only itself, so the
pattern around it is what can backtrack. A call whose pattern only exists at runtime is found with no patterns, for
the package's test to say where its pattern comes from.

Each call site also says which `StepCounter` methods run its patterns the way the code does. A `re.match` or
`re.fullmatch` call is run with `match` or `fullmatch`. A pattern compiled into a module-level name is run with each
of those its module calls on the name, and with `scan` for any other use, or when the name is public or another
module of the package names it, since those uses aren't in the module. Every other call is run with `scan`, which
makes an attempt at every position, as `re.finditer` does. Code that repeats a `match` along the text makes those
attempts too, which a check of one `match` doesn't show.
"""

from __future__ import annotations

import ast
import bisect
import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Container, Iterator
from dataclasses import dataclass
from pathlib import Path

from ._regex_steps import Method

_FLAGS_ARGUMENT = {
    'compile': 1,
    'search': 2,
    'match': 2,
    'fullmatch': 2,
    'findall': 2,
    'finditer': 2,
    'split': 3,
    'sub': 4,
    'subn': 4,
}
"""Each `re` function that takes a pattern, and the position of its `flags` argument."""
_ESCAPED = 'x'
"""What `re.escape(...)` of runtime text stands in as."""


@dataclass(frozen=True)
class _ElementOf:
    """A comprehension target, bound to each element of a literal sequence in turn."""

    sequence: ast.expr


_Binding = ast.expr | _ElementOf | None
"""What a name is bound to in a scope; `None` when it can't be worked out statically, such as a parameter."""


class _Unresolved(Exception):
    """The expression's value isn't known until runtime."""


def _assigned(node: ast.AST) -> tuple[str, ast.expr] | None:
    """The name an assignment to one name binds, and the value it binds it to."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        return node.target.id, node.value
    return None


def _bindings(scope: ast.AST) -> dict[str, _Binding]:
    """The names a module or function binds in its own scope, not in the functions and classes nested in it."""
    bindings: dict[str, _Binding] = {}
    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ):
            if argument is not None:
                bindings[argument.arg] = None
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if (assignment := _assigned(node)) is not None:
            name, value = assignment
            bindings[name] = None if name in bindings else value
            pending.append(value)
            continue
        if isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            bindings[node.target.id] = _ElementOf(node.iter)
            pending.extend([node.iter, *node.ifs])
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings[node.id] = None
        pending.extend(ast.iter_child_nodes(node))
    return bindings


def _lookup(name: str, scopes: list[dict[str, _Binding]]) -> _Binding:
    for scope in scopes:
        if name in scope:
            return scope[name]
    raise _Unresolved(name)


def _literal_container(node: ast.expr, scopes: list[dict[str, _Binding]]) -> ast.expr:
    """The display a name refers to, following names bound to names."""
    while isinstance(node, ast.Name):
        binding = _lookup(node.id, scopes)
        if not isinstance(binding, ast.expr):
            raise _Unresolved(node.id)
        node = binding
    return node


def _elements(node: ast.expr, scopes: list[dict[str, _Binding]]) -> list[ast.expr]:
    """The elements of a tuple, list or set display, or a dict display's keys, as iterating it gives them."""
    container = _literal_container(node, scopes)
    if isinstance(container, ast.Dict):
        return [key for key in container.keys if key is not None]
    if isinstance(container, ast.Tuple | ast.List | ast.Set):
        return container.elts
    raise _Unresolved(ast.unparse(node))


def _subscript(node: ast.Subscript, scopes: list[dict[str, _Binding]]) -> ast.expr:
    """The element a subscript of a dict, tuple or list display picks."""
    container = _literal_container(node.value, scopes)
    if isinstance(container, ast.Subscript):
        container = _subscript(container, scopes)
    entries: dict[str | int, ast.expr]
    if isinstance(container, ast.Dict):
        entries = {
            key: value
            for key_node, value in zip(container.keys, container.values)
            if key_node
            for key in _fold(key_node, scopes)
        }
    elif isinstance(container, ast.Tuple | ast.List):
        entries = {index: element for index, element in enumerate(container.elts)}
    else:
        raise _Unresolved(ast.unparse(node))
    try:
        [key] = _fold(node.slice, scopes)
        return entries[key]
    except (ValueError, KeyError) as error:
        raise _Unresolved(ast.unparse(node)) from error


def _named(name: str, scopes: list[dict[str, _Binding]]) -> list[str | int]:
    """Every value the name `name` can take."""
    binding = _lookup(name, scopes)
    if isinstance(binding, _ElementOf):
        return [value for element in _elements(binding.sequence, scopes) for value in _fold(element, scopes)]
    if binding is None:
        raise _Unresolved(name)
    return _fold(binding, scopes)


def _compiled_pattern(node: ast.Attribute, name: str, scopes: list[dict[str, _Binding]]) -> ast.expr:
    """The pattern `name.pattern` reads: the one `name` was compiled from."""
    compiled = _lookup(name, scopes)
    if not (isinstance(compiled, ast.Call) and ast.unparse(compiled.func) == 're.compile'):
        raise _Unresolved(ast.unparse(node))
    pattern = _argument(compiled, 0, 'pattern')
    assert pattern is not None
    return pattern


def _strings(values: list[str | int]) -> list[str]:
    strings = [value for value in values if isinstance(value, str)]
    assert len(strings) == len(values), f'expected strings, got {values!r}'
    return strings


def _ints(values: list[str | int]) -> list[int]:
    ints = [value for value in values if isinstance(value, int)]
    assert len(ints) == len(values), f'expected flags, got {values!r}'
    return ints


def _fold(node: ast.expr, scopes: list[dict[str, _Binding]]) -> list[str | int]:
    """Every value `node` can take, worked out from the source alone."""
    match node:
        case ast.Constant(value=str() | int() as value):
            return [value]
        case ast.JoinedStr(values=values):
            texts = ['']
            for value in values:
                if isinstance(value, ast.FormattedValue) and (value.conversion != -1 or value.format_spec):
                    raise _Unresolved(ast.unparse(value))
                part = _strings(_fold(value.value if isinstance(value, ast.FormattedValue) else value, scopes))
                texts = [text + piece for text in texts for piece in part]
            return list(texts)
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            return [a + b for a in _strings(_fold(left, scopes)) for b in _strings(_fold(right, scopes))]
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            return [a | b for a in _ints(_fold(left, scopes)) for b in _ints(_fold(right, scopes))]
        case ast.Attribute(value=ast.Name(id='re'), attr=attribute) if isinstance(getattr(re, attribute, None), int):
            flag: int = getattr(re, attribute)
            return [flag]
        case ast.Attribute(value=ast.Name(id=name), attr='pattern'):
            return _fold(_compiled_pattern(node, name, scopes), scopes)
        case ast.Name(id=name):
            return _named(name, scopes)
        case ast.Subscript():
            return _fold(_subscript(node, scopes), scopes)
        case ast.Call(func=ast.Attribute(value=ast.Name(id='re'), attr='escape')):
            return [_ESCAPED]
        case ast.Call(func=ast.Attribute(value=separator, attr='join'), args=[sequence]):
            choices = [_strings(_fold(element, scopes)) for element in _elements(sequence, scopes)]
            joiners = _strings(_fold(separator, scopes))
            return [joiner.join(parts) for joiner in joiners for parts in itertools.product(*choices)]
        case ast.IfExp(body=body, orelse=orelse):
            return [*_fold(body, scopes), *_fold(orelse, scopes)]
        case _:
            raise _Unresolved(ast.unparse(node))


@dataclass(frozen=True)
class CallSite:
    location: str
    """`<file>::<qualified name>`: the function the call is in, or the name a module-level call is assigned to."""
    patterns: list[tuple[str, int]] | None
    """The patterns and flags the call can compile, or `None` if they only exist at runtime."""
    methods: tuple[Method, ...] = ('scan',)
    """The `StepCounter` methods that run the patterns the way the code does."""


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _patterns(call: ast.Call, function: str, scopes: list[dict[str, _Binding]]) -> list[tuple[str, int]] | None:
    pattern = _argument(call, 0, 'pattern')
    flags = _argument(call, _FLAGS_ARGUMENT[function], 'flags')
    assert pattern is not None
    try:
        patterns = _strings(_fold(pattern, scopes))
        flag_values = [0] if flags is None else _ints(_fold(flags, scopes))
    except _Unresolved:
        return None
    return [(text, flag) for text in patterns for flag in flag_values]


_METHODS: dict[str, Method] = {'match': 'match', 'fullmatch': 'fullmatch'}
"""The `re` functions and pattern methods that make one attempt, and the `StepCounter` method that runs each."""


def _uses(tree: ast.Module) -> dict[str, set[Method]]:
    """The methods each name's uses in the module may run a pattern compiled into it with.

    Reading `.pattern` builds another call's pattern, so it runs nothing. Any other use but a `match` or `fullmatch`
    call, such as passing the compiled pattern on, may run it with any method, so it counts as `scan`.
    """
    nodes = list(ast.walk(tree))
    attributes = {id(node.value): node for node in nodes if isinstance(node, ast.Attribute)}
    called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    uses: defaultdict[str, set[Method]] = defaultdict(set)
    for node in nodes:
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        attribute = attributes.get(id(node))
        if attribute is None:
            uses[node.id].add('scan')
        elif id(attribute) in called:
            uses[node.id].add(_METHODS.get(attribute.attr, 'scan'))
        elif attribute.attr != 'pattern':
            uses[node.id].add('scan')
    return uses


def call_sites_in(
    relative: str, tree: ast.Module, source: str, shared: Container[str] = frozenset()
) -> Iterator[CallSite]:
    """Every call to a `re` function that takes a pattern, in one module, whose source is `source`.

    `shared` holds the names other modules of the package use too, whose uses there this module doesn't show.
    """
    # A call names `re` on its first line, so a statement or expression on lines that never name it holds no call.
    # Skipping those keeps the scan cheap enough to run while a test module is collected.
    lines = sorted({source.count('\n', 0, match.start()) + 1 for match in re.finditer(r'\bre\b', source)})
    compiled_into: dict[int, str] = {}
    for statement in tree.body:
        if (assignment := _assigned(statement)) is not None:
            compiled_into[id(assignment[1])] = assignment[0]
    uses: dict[str, set[Method]] | None = None

    def may_hold_a_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.stmt | ast.expr) or node.end_lineno is None:
            return True
        start = node.lineno
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            start = min([start, *(decorator.lineno for decorator in node.decorator_list)])
        index = bisect.bisect_left(lines, start)
        return index < len(lines) and lines[index] <= node.end_lineno

    def visit(node: ast.AST, qualname: str, scopes: list[dict[str, _Binding]]) -> Iterator[CallSite]:
        nonlocal uses
        for child in ast.iter_child_nodes(node):
            if not may_hold_a_call(child):
                continue
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f'{qualname}.{child.name}' if qualname else child.name
                inner = scopes if isinstance(child, ast.ClassDef) else [_bindings(child), *scopes]
                yield from visit(child, name, inner)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == 're'
                and child.func.attr in _FLAGS_ARGUMENT
            ):
                function, name = child.func.attr, compiled_into.get(id(child))
                methods: tuple[Method, ...] = (_METHODS.get(function, 'scan'),)
                if function == 'compile' and name is not None:
                    uses = _uses(tree) if uses is None else uses
                    local = set(uses.get(name, ()))
                    if not name.startswith('_') or name in shared:
                        local.add('scan')
                    methods = tuple(sorted(local)) or methods
                yield CallSite(f'{relative}::{qualname}', _patterns(child, function, scopes), methods)
            assignment = _assigned(child)
            yield from visit(child, assignment[0] if assignment and not qualname else qualname, scopes)

    yield from visit(tree, '', [_bindings(tree)])


def hidden_re_uses(tree: ast.Module) -> list[str]:
    """Uses of `re` that `call_sites_in` can't see: imported names, a renamed or rebound `re`, a function passed on."""
    nodes = list(ast.walk(tree))
    attribute_values = {id(node.value) for node in nodes if isinstance(node, ast.Attribute)}
    called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    hidden: list[str] = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module in ('re', 'regex'):
            hidden.append(f'from {node.module} import ...')
        elif isinstance(node, ast.Import):
            hidden += [f'import {a.name} as {a.asname}' for a in node.names if a.name == 're' and a.asname]
            hidden += ['import regex' for a in node.names if a.name == 'regex']
        elif isinstance(node, ast.Name) and node.id == 're' and id(node) not in attribute_values:
            hidden.append(f'`re` itself, at line {node.lineno}')
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 're'
            and node.attr in _FLAGS_ARGUMENT
            and id(node) not in called
        ):
            hidden.append(f'`re.{node.attr}` uncalled, at line {node.lineno}')
    return hidden


@dataclass(frozen=True)
class _SharedNames:
    """The names more than one of `sources` uses, found by searching the sources for each name asked about."""

    sources: tuple[str, ...]

    def __contains__(self, name: object) -> bool:
        assert isinstance(name, str)
        word = re.compile(rf'\b{re.escape(name)}\b')
        return sum(1 for source in self.sources if name in source and word.search(source)) > 1


class PackageScan:
    """The `re` call sites in a package's source, read once, when the test module that checks them is collected."""

    def __init__(self, package: Path) -> None:
        self.sources = {
            path.relative_to(package).as_posix(): path.read_text(encoding='utf-8')
            for path in sorted(package.rglob('*.py'))
        }
        # A module that never names `re` can't call it, so it isn't parsed for call sites; `hidden_re_uses`
        # finds the other ways to reach `re`, when its test runs.
        shared = _SharedNames(tuple(self.sources.values()))
        self.call_sites = [
            site
            for relative, source in self.sources.items()
            if re.search(r'\bre\b', source)
            for site in call_sites_in(relative, ast.parse(source), source, shared)
        ]

    def checked_patterns(self, tables: dict[str, dict[str, tuple[str, int]]]) -> dict[str, tuple[str, int, Method]]:
        """Each pattern the call sites and `tables` compile, with a method that runs it, labelled where it is first found.

        A table's patterns are run with `scan`: the calls that compile them are the ones the package can't resolve.
        """
        checked: dict[str, tuple[str, int, Method]] = {}
        seen: Counter[str] = Counter()
        for site in self.call_sites:
            for (pattern, flags), method in itertools.product(site.patterns or (), site.methods):
                seen[site.location] += 1
                label = f'{site.location}#{seen[site.location]}' if seen[site.location] > 1 else site.location
                checked[label] = (pattern, flags, method)
        for table_name, table in tables.items():
            checked.update(
                {f'{table_name}[{name!r}]': (pattern, flags, 'scan') for name, (pattern, flags) in table.items()}
            )
        unique: dict[tuple[str, int, Method], str] = {}
        for label, pattern in checked.items():
            unique.setdefault(pattern, label)
        return {label: pattern for pattern, label in unique.items()}

    def unresolved(self) -> dict[str, int]:
        """How many calls at each location compile a pattern that only exists at runtime."""
        return dict(Counter(site.location for site in self.call_sites if site.patterns is None))

    def default_tables(self) -> list[str]:
        """The `DEFAULT_*_PATTERNS` tables the package defines."""
        return sorted(
            {
                name
                for source in self.sources.values()
                for name in re.findall(r'^(DEFAULT_\w+_PATTERNS)\b', source, re.M)
            }
        )

    def hidden_re_uses(self) -> list[tuple[str, str]]:
        """The uses of `re` in each module that its call sites don't show."""
        return [
            (relative, hidden)
            for relative, source in self.sources.items()
            if re.search(r'\b(?:re|regex)\b', source)
            for hidden in hidden_re_uses(ast.parse(source))
        ]
