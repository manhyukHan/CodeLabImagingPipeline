"""
Dead-definition audit: every function/method/class defined in the live
tree, checked against every textual reference to its name.

    python tools/audit_dead_code.py

Exists because run-based verification (smoke sweep, clicked flows) can
only vouch for code that RUNS: a function nothing calls is invisible to
it by construction, and this codebase is built by moving old attempts
aside and rebuilding -- exactly the process that strands orphans.
Confirmed real on 2026-08-19: six dead functions (one transitively-dead
pair among them) found on the first sweep, in a tree whose executed
paths were all green.

Two tiers, because the first version of this sweep counted docstring
mentions as references and a zero-caller method survived behind a single
"see calculate_distmap" line:

  DEAD:      the name appears nowhere outside its own definition.
  CODE-DEAD: the name appears only in comments/docstrings -- no code
             references it, but prose still points readers at it. Either
             the code or the prose is stale; both are findings.

Reference counting is textual (word-boundary), on the whole tree
including tests/ and tools/, so a definition whose only caller is a test
still counts as alive. Known limit: purely dynamic dispatch
(getattr/dir with no static mention of the name anywhere) would be
flagged -- if that ever produces a false positive, the fix is a code
comment naming the callee at the dispatch site, which is documentation
the dispatch needs anyway.

Skipped by name: dunders, and Qt virtual hooks the framework calls
(paintEvent, closeEvent, ...).

Exit code 1 when anything is flagged, so this can gate a commit.
"""
import ast
import collections
import os
import re
import sys
import tokenize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Definitions are audited in the LIVE tree only; tests/ and tools/ count
# as reference sources (a helper only a test calls is alive) but their
# own defs are not audited -- test functions are invoked by runners, and
# flagging every test_* would be pure noise.
LIVE = ['canvas', 'codelab_pipeline', 'ui', 'windows', 'config.py', 'main.py']
REF_ONLY = ['tools', 'tests']

QT_HOOKS = {
    'closeEvent', 'keyPressEvent', 'mousePressEvent', 'mouseMoveEvent',
    'mouseReleaseEvent', 'wheelEvent', 'resizeEvent', 'paintEvent',
    'eventFilter', 'showEvent', 'hideEvent', 'focusInEvent', 'focusOutEvent',
    'contextMenuEvent', 'dragEnterEvent', 'dropEvent', 'sizeHint',
    'minimumSizeHint', 'event', 'timerEvent', 'changeEvent', 'leaveEvent',
    'enterEvent',
}


def iter_py(paths):
    for p in paths:
        full = os.path.join(REPO, p)
        if full.endswith('.py'):
            yield full
        else:
            for root, dirs, files in os.walk(full):
                dirs[:] = [d for d in dirs if d != '__pycache__']
                for f in files:
                    if f.endswith('.py'):
                        yield os.path.join(root, f)


def definitions(path, source):
    """(kind, qualname, name, path, lineno) for every def/class."""
    out = []
    stack = []

    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            out.append(('class', '.'.join(stack + [node.name]), node.name, path, node.lineno))
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node):
            kind = 'method' if stack else 'function'
            out.append((kind, '.'.join(stack + [node.name]), node.name, path, node.lineno))
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    V().visit(ast.parse(source))
    return out


def code_only(source):
    """
    The source with comments and docstrings blanked (line structure
    preserved). Non-docstring string literals are deliberately KEPT: a
    string can be a real reference (getattr dispatch, a dispatch table),
    and dropping those would flag live code.
    """
    lines = source.splitlines(keepends=True)
    blank = set()
    # docstrings: the Expr-statement string opening a module/class/function body
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, 'body', [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                blank.update(range(body[0].lineno, body[0].end_lineno + 1))
    kept = []
    for i, line in enumerate(lines, 1):
        kept.append('\n' if i in blank else line)
    # comments, via tokenize so '#' inside strings survives
    try:
        toks = list(tokenize.generate_tokens(iter(kept).__next__))
    except tokenize.TokenizeError:
        return ''.join(kept)
    out_lines = [list(l) for l in kept]
    for tok in toks:
        if tok.type == tokenize.COMMENT and tok.start[0] == tok.end[0]:
            row = tok.start[0] - 1
            s, e = tok.start[1], tok.end[1]
            out_lines[row][s:e] = ' ' * (e - s)
    return ''.join(''.join(l) for l in out_lines)


def main():
    live_files = list(iter_py(LIVE))
    all_files = live_files + list(iter_py(REF_ONLY))
    full_src = {f: open(f).read() for f in all_files}
    code_src = {f: code_only(s) for f, s in full_src.items()}

    defs = []
    for f in live_files:
        defs.extend(definitions(f, full_src[f]))

    names = {d[2] for d in defs}
    n_full = collections.Counter()
    n_code = collections.Counter()
    for name in names:
        pat = re.compile(r'\b' + re.escape(name) + r'\b')
        for f in all_files:
            n_full[name] += len(pat.findall(full_src[f]))
            n_code[name] += len(pat.findall(code_src[f]))

    defcount = collections.Counter(d[2] for d in defs)
    dead, code_dead = [], []
    for kind, qual, name, f, ln in defs:
        if re.match(r'^__.*__$', name) or name in QT_HOOKS:
            continue
        if n_code[name] - defcount[name] > 0:
            continue                        # code references it: alive
        rel = os.path.relpath(f, REPO)
        entry = (rel, ln, kind, qual)
        if n_full[name] - defcount[name] > 0:
            code_dead.append(entry)         # prose mentions it, code doesn't
        else:
            dead.append(entry)

    total = len(defs)
    if not dead and not code_dead:
        print(f'{total} definitions audited: none dead.')
        return 0
    if dead:
        print(f'DEAD ({len(dead)}) -- no reference anywhere outside the definition:')
        for rel, ln, kind, qual in sorted(dead):
            print(f'  {rel}:{ln}  {kind}  {qual}')
    if code_dead:
        print(f'CODE-DEAD ({len(code_dead)}) -- mentioned only in comments/docstrings:')
        for rel, ln, kind, qual in sorted(code_dead):
            print(f'  {rel}:{ln}  {kind}  {qual}')
    print(f'\n{total} definitions audited; delete the finding or the stale prose pointing at it.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
