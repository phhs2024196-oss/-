#!/usr/bin/env python3
"""SJU-G19B (세종대 2019학년도 수시 논술고사[B]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SJU-G19B"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
E = sp.E


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str) -> dict:
    """'키=식; 키=식' 형태의 정답을 dict 로."""
    return {a.strip(): b.strip() for a, b in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s, **loc):
    return sp.sympify(s, locals={"e": sp.E, "x": x, "t": t, **loc})


def same(a, b, tol=1e-12):
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 if not (isinstance(a, float) or isinstance(b, float)) else abs(float(a) - float(b)) < tol
    except Exception:  # noqa: BLE001
        return abs(float(a) - float(b)) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _y_of_x(xv):
    """원 C 내부 점 (x,y) 에서 반직선 P→점 이 원과 만나는 점까지 거리 = x 인 y (0<y<1) — 이분법."""
    import mpmath as mp
    g = lambda yv: 1 - mp.sqrt(xv**2 + (1 - yv)**2) - xv
    return mp.findroot(g, (mp.mpf("1e-30"), 1 - mp.mpf("1e-30")), solver="bisect", tol=mp.mpf(10)**-40)


def p01():
    import mpmath as mp
    mp.mp.dps = 60
    X1 = sp.Symbol("x1")
    e1 = S(md_answer(1, "1"), x1=X1)
    ok1 = all(abs(_y_of_x(mp.mpf(v)) - mp.mpf(str(sp.N(e1.subs(X1, sp.Rational(v)), 50)))) < mp.mpf(10)**-30
              for v in ("0.01", "0.1", "0.3", "0.45"))
    r2, r3 = [], []
    for x1 in (mp.mpf("1e-4"), mp.mpf("1e-6"), mp.mpf("1e-8")):
        y1 = _y_of_x(x1)
        def phi(x2):
            y2 = _y_of_x(x2)
            return mp.sqrt((x1 - x2)**2 + (y1 - y2)**2) - (x1 + x2)
        x2 = mp.findroot(phi, (x1 * mp.mpf("1e-6"), x1 * (1 - mp.mpf("1e-12"))), solver="bisect", tol=mp.mpf(10)**-50)
        y2 = _y_of_x(x2)
        r2.append(y2 / x2); r3.append(y2 / y1)
    ok2 = abs(r2[-1] - mp.mpf(str(sp.N(S(md_answer(1, "2")), 50)))) < 1e-7
    ok3 = abs(r3[-1] - mp.mpf(str(sp.N(S(md_answer(1, "3")), 50)))) < 1e-7
    report(1, ok1 and ok2 and ok3, f"y2/x2→{mp.nstr(r2[-1], 10)} y2/y1→{mp.nstr(r3[-1], 12)} (기하 조건 직접 수치해)")


def _tet():
    O = sp.Matrix([0, 0, 0]); A = sp.Matrix([1, 0, 0]); B = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 2, 0])
    C = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 6, sp.sqrt(6) / 3])
    return O, A, B, C


def p02():
    O, A, B, C = _tet()
    ok0 = all(sp.simplify((U - V).norm() - 1) == 0 for U, V in itertools.combinations([O, A, B, C], 2))
    proj = lambda v: sp.Matrix([v[0], v[1], 0])         # 평면 α: z=0
    m, nn = sp.symbols("m nn")
    sol = sp.solve(list(proj(C) - m * A - nn * B), [m, nn], dict=True)[0]
    d = md_parts(2, "1")
    ok1 = sol[m] == S(d["m"]) and sol[nn] == S(d["n"])
    a, b, c = sp.symbols("a b c")
    p, q = sp.symbols("p q")
    PS = a * A + b * B + c * C
    s2 = sp.solve(list(proj(PS) - p * A - q * B)[:2], [p, q], dict=True)[0]
    d2 = md_parts(2, "2")
    ok2 = sp.simplify(s2[p] - S(d2["p"], a=a, b=b, c=c)) == 0 and sp.simplify(s2[q] - S(d2["q"], a=a, b=b, c=c)) == 0
    P = A / 3; Q = (7 * A + 3 * B) / 10
    tt = sp.Symbol("tt")
    R = tt * A + (1 - tt) * C
    nrm = (Q - P).cross(sp.Matrix([0, 0, 1]))           # α 에 수직이고 PQ 포함 평면의 법선
    tv = sp.solve(sp.Eq(nrm.dot(R - P), 0), tt)
    ok3 = len(tv) == 1 and sp.simplify(tv[0] - S(md_answer(2, "3"))) == 0 and 0 < tv[0] < 1
    report(2, ok0 and ok1 and ok2 and ok3, f"C1=({sol[m]},{sol[nn]}) p={s2[p]} q={s2[q]} t={tv}")


def p03():
    a = sp.Symbol("a")
    # (1) (가)에서 f'(1)=f(1)-5, f(1)=a → 접선
    h = (a - 5) * (x - 1) + a
    ok1 = sp.expand(h - S(md_answer(3, "1"), a=a)) == 0
    # 조건을 만족하는 함수 후보 f=5x+5 를 (가)(나)(다)로 직접 확인한 뒤 (2)~(4) 수치 확인 [증명-수치확인]
    f = 5 * x + 5
    okf = sp.simplify(sp.diff(f, x) - (f.subs(x, x**2) - 5 * x**2)) == 0 and sp.diff(f, x) >= 5 and f.subs(x, 0) == 5
    av = f.subs(x, 1)
    import numpy as np
    xs = np.linspace(0, 20, 20001)
    u = sp.lambdify(x, f - 5 * x)(xs) * np.ones_like(xs)
    ok2 = abs(u.min() - float(S(md_answer(3, "2")))) < 1e-12
    g = sp.lambdify(x, f - h.subs(a, av))(xs) * np.ones_like(xs)
    ok3 = abs(g.min() - float(S(md_answer(3, "3")))) < 1e-12
    d = md_parts(3, "4")
    xs01 = np.linspace(0, 1, 1001)
    ok4 = np.allclose(sp.lambdify(x, f - h.subs(a, av))(xs01) * np.ones_like(xs01), float(S(d["g(x)"]))) and S(d["a"]) == av
    report(3, ok1 and okf and ok2 and ok3 and ok4, f"h=(a-5)x+5, f=5x+5 가 조건 만족, a={av} [증명-수치확인]")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
