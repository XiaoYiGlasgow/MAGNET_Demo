"""Verify that a translated source file is logically identical to its original.

Rule: the two files must have identical ASTs, except that string constants whose ORIGINAL
value contains CJK characters may differ (comments, docstrings and log/print messages that
were translated). The translated value must not contain CJK characters. Comments are not part
of the AST, so they may change freely.

Usage:
  python check_translation_ast.py <original.py> <translated.py> [<original.py> <translated.py> ...]
Exit code 0 if all pairs pass.
"""
import ast
import re
import sys

CJK = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def compare(a, b, path, diffs):
    if type(a) is not type(b):
        diffs.append(f"{path}: node type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        if a.value != b.value:
            if not isinstance(b.value, str):
                diffs.append(f"{path}: str constant replaced by non-str")
            elif not CJK.search(a.value):
                diffs.append(f"{path}: non-CJK string changed: {a.value!r} -> {b.value!r}")
        if isinstance(b.value, str) and CJK.search(b.value):
            diffs.append(f"{path}: CJK left in string: {b.value!r}")
        return
    if isinstance(a, ast.AST):
        for field in a._fields:
            compare(getattr(a, field, None), getattr(b, field, None), f"{path}.{field}", diffs)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", diffs)
        return
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def check(orig_path, new_path):
    orig_src = open(orig_path, encoding="utf-8").read()
    new_src = open(new_path, encoding="utf-8").read()
    diffs = []
    compare(ast.parse(orig_src), ast.parse(new_src), "module", diffs)
    cjk_lines = [i + 1 for i, line in enumerate(new_src.splitlines()) if CJK.search(line)]
    if cjk_lines:
        diffs.append(f"CJK characters remain on lines {cjk_lines[:20]}")
    return diffs


def main():
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    ok = True
    for orig, new in zip(args[0::2], args[1::2]):
        diffs = check(orig, new)
        print(f"{'PASS' if not diffs else 'FAIL'}  {new}")
        for d in diffs[:30]:
            print("   ", d)
        ok &= not diffs
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
