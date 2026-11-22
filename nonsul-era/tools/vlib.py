"""검산 헬퍼 — verify 스크립트는 설정·재계산 몇 줄만 쓴다(정답 읽기·비교·보고·실행은 여기서).

예 (verify/AJOU-G24A.py):
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
    from vlib import *
    V = Verifier("AJOU-G24A", "items/AJOU/PAST")

    def p01():
        f = x**2/2 - 2
        V.check(1, "1-1 max", extrema(f, x, 2, 3)[1], V.ans(1, "1-1", "max"))
        V.check(1, "1-2", integral(sp.sin(x), x, 0, sp.pi), V.ans(1, "1-2"))

    if __name__ == "__main__": V.run(globals())

출력: 기본은 FAIL 줄 + SUMMARY 만(토큰 절약). 전체 보기: -v
규칙(제작지침 §6): 정답 하드코딩 금지 — 기대값은 항상 V.ans() 로 YAML 에서 읽는다.
"""
from __future__ import annotations

import itertools
import pathlib
import re
import sys

import sympy as sp
import yaml

x, y, t, n, k = sp.symbols("x y t n k", real=True)
KIT = pathlib.Path(__file__).resolve().parent.parent
__all__ = ["sp", "x", "y", "t", "n", "k", "Verifier", "same", "extrema", "integral", "limit",
           "series_sum", "roots_in", "search", "S"]
S = sp.sympify


def same(a, b, tol: float = 1e-10) -> bool:
    """기호적으로 같거나(simplify), 수치가 tol 이내면 True. 문자열은 sympify."""
    a = S(a) if isinstance(a, str) else a
    b = S(b) if isinstance(b, str) else b
    if isinstance(a, (list, tuple, set, sp.FiniteSet)) or isinstance(b, (list, tuple, set, sp.FiniteSet)):
        la, lb = sorted(map(S, a), key=sp.default_sort_key), sorted(map(S, b), key=sp.default_sort_key)
        return len(la) == len(lb) and all(same(p, q, tol) for p, q in zip(la, lb))
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return abs(complex(sp.N(a - b, 30))) < tol
    except Exception:  # noqa: BLE001
        return False


def extrema(expr, var, a, b):
    """닫힌구간 [a,b] 의 (최솟값, 최댓값) 정확값: 임계점 + 끝점. 미분 불가능한 점(절댓값·max 등)은
    후보에 안 들어가므로 그런 함수는 직접 후보를 넣어 계산할 것."""
    cand = {S(a), S(b)}
    try:
        for r in sp.solveset(sp.diff(expr, var), var, sp.Interval(a, b)):
            cand.add(r)
    except Exception:  # noqa: BLE001
        pass
    vals = [sp.simplify(expr.subs(var, c)) for c in cand]
    return min(vals, key=lambda v: float(sp.N(v))), max(vals, key=lambda v: float(sp.N(v)))


def integral(expr, var, a, b):
    return sp.simplify(sp.integrate(expr, (var, a, b)))


def limit(expr, var, pt, dir="+-"):
    return sp.simplify(sp.limit(expr, var, pt, dir) if dir != "+-" else sp.limit(expr, var, pt))


def series_sum(term, var, a, b):
    return sp.simplify(sp.summation(term, (var, a, b)))


def roots_in(expr, var, a, b):
    return sorted(sp.solveset(expr, var, sp.Interval(a, b)), key=lambda r: float(sp.N(r)))


def search(pred, *ranges, limit_fail: int = 1):
    """증명형 수치 반례 탐색: pred(*vals) 가 거짓인 조합 목록(최대 limit_fail개). 빈 목록이면 반례 없음."""
    bad = []
    for vals in itertools.product(*ranges):
        if not pred(*vals):
            bad.append(vals)
            if len(bad) >= limit_fail:
                break
    return bad


class Verifier:
    def __init__(self, prefix: str, items_dir: str, width: int = 2):
        self.prefix, self.width = prefix, width
        self.items = KIT / items_dir
        self.res: list[bool] = []
        self.verbose = "-v" in sys.argv
        self._cache: dict = {}

    def _item(self, no: int) -> dict:
        if no not in self._cache:
            f = self.items / f"{self.prefix}-{no:0{self.width}d}.yaml"
            self._cache[no] = yaml.safe_load(f.read_text(encoding="utf-8"))
        return self._cache[no]

    def ans(self, no: int, sub: str, key: str | None = None) -> str:
        """answer.value 를 읽는다. 'max = 5/2; min = 1' 꼴이면 key 로 한 값만."""
        v = str(next(s["answer"]["value"] for s in self._item(no)["subparts"] if str(s["id"]) == sub))
        if key is None:
            return v
        parts = dict(p.split("=", 1) for p in v.replace(" ", "").split(";") if "=" in p)
        return parts[key]

    def report(self, no: int, ok: bool, msg: str) -> None:
        self.res.append(bool(ok))
        if self.verbose or not ok:
            print(f"{self.prefix}-{no:0{self.width}d} {'PASS' if ok else 'FAIL'}  {msg}")

    def check(self, no: int, label: str, computed, expected, tol: float = 1e-10) -> bool:
        ok = same(computed, expected, tol)
        self.report(no, ok, f"{label}: 계산 {computed} / YAML {expected}")
        return ok

    def proof(self, no: int, label: str, counterexamples: list) -> None:
        self.report(no, not counterexamples, f"{label} [증명-수치확인] 반례 {counterexamples[:1] or '없음'}")

    def run(self, g: dict) -> None:
        for name in sorted((k for k in g if re.fullmatch(r"p\d+", k)), key=lambda s: int(s[1:])):
            try:
                g[name]()
            except Exception as e:  # noqa: BLE001
                self.report(int(name[1:]), False, f"{type(e).__name__}: {e}")
        npass = sum(self.res)
        print(f"SUMMARY {self.prefix}: {npass}/{len(self.res)} PASS")
        sys.exit(0 if self.res and npass == len(self.res) else 1)
