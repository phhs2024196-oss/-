#!/usr/bin/env python3
"""SJU-M19 (세종대 2019학년도 논술 모의평가(자연계열)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SJU-M19"
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


BEATS = {("가위", "보"): 1, ("바위", "바위"): 0, ("가위", "바위"): -1, ("바위", "보"): -1}


def p01():
    seqA = ["가위", "바위", "가위", "바위", "가위"]
    cnt = 0
    for seqB in itertools.product(["바위", "보"], repeat=5):
        r = [BEATS[(a, b)] for a, b in zip(seqA, seqB)]
        cnt += (r.count(1), r.count(-1), r.count(0)) == (2, 2, 1)
    ok1 = cnt == int(md_answer(1, "1"))
    combos = list(itertools.product(["가위", "바위"], ["바위", "보"]))
    pw = Fraction(sum(BEATS[c] == 1 for c in combos), len(combos))
    nmin = next(N for N in range(1, 50) if 1 - (1 - pw)**N >= Fraction(1, 2))
    ok2 = nmin == int(md_answer(1, "2"))
    ok3 = True
    for N in range(1, 6):
        ex = Fraction(0)
        for game in itertools.product(combos, repeat=N):
            kw = sum(BEATS[c] == 1 for c in game)
            ex += Fraction(9**kw, len(combos)**N)
        ok3 &= ex == int(S(md_answer(1, "3"), n=sp.Integer(N)))
    report(1, ok1 and ok2 and ok3, f"방법 {cnt}, p={pw}, n_min={nmin}, E(9^k) n=1..5 전수")


def p02():
    def center(nv, d):
        nv = sp.Matrix(nv)
        return nv * d / nv.dot(nv)
    A = center([-1, 2, 3], 1); B = center([1, 0, 1], 1)
    ok1 = sp.Matrix(S(md_answer(2, "1"))) == A
    cs = A.dot(B) / (A.norm() * B.norm())
    ok2 = sp.simplify(cs - S(md_answer(2, "2"))) == 0
    import numpy as np
    b = np.array([float(v) for v in B]); a = np.array([float(v) for v in A])
    r = np.sqrt(1 - b @ b)
    u = np.array([1, 0, -1]) / np.sqrt(2); w = np.array([0, 1, 0])
    th = np.linspace(0, 2 * np.pi, 2000001)
    P = b[:, None] + r * (np.outer(u, np.cos(th)) + np.outer(w, np.sin(th)))
    vals = a @ P
    ok3 = abs(vals.min() - float(S(md_answer(2, "3")))) < 1e-10 and np.allclose((P * P).sum(0), 1)
    report(2, ok1 and ok2 and ok3, f"A={list(A)} cos={sp.nsimplify(cs)} min={vals.min():.12f}")


def p03():
    c0, c1, c2, c3, a, K = sp.symbols("c0 c1 c2 c3 a K")
    f = c0 + c1 * x + c2 * x**2 + c3 * x**3
    lhs = sp.integrate(f.subs(x, t), (t, 0, x))
    rhs = x**3 / 3 + x**2 + 2 + a * (x**2 / 2 + 1) * K
    eqs = sp.Poly(sp.expand(lhs - rhs), x).all_coeffs() + [K - sp.integrate((t * sp.diff(f, x).subs(x, t)), (t, 0, 1))]
    sol = sp.solve(eqs, [c0, c1, c2, c3, a, K], dict=True)
    d = md_parts(3, "1")
    ok1 = len(sol) == 1 and sp.expand(f.subs(sol[0]) - S(d["f(x)"])) == 0 and sol[0][a] == S(d["a"])
    fs = f.subs(sol[0])
    # (2) h=r 일 때 교점(x) 개수: r=1/2 → 1개, r 약간 크면 3개
    def nroots(rv):
        return len([z for z in sp.solve(sp.Eq(x**2 + (fs - rv)**2, rv**2), x) if z.is_real])
    r0 = S(md_answer(3, "2"))
    ok2 = nroots(r0) == 1 and nroots(r0 + sp.Rational(1, 100)) == 3 and nroots(r0 / 2) == 1
    import numpy as np
    fl = sp.lambdify(x, fs)
    ts = np.linspace(-5, 5, 2000001)
    dmin = np.sqrt((ts - 3)**2 + fl(ts)**2).min() - 1
    ok3 = abs(dmin - float(S(md_answer(3, "3")))) < 1e-9
    hh = lambda z: -0.5 * (z - 2)**2 + 0.25
    dfl = sp.lambdify(x, sp.diff(fs, x))
    s = 2 - dfl(ts)                     # h'(s)=-(s-2)=f'(t)
    pr = np.sqrt((ts - s)**2 + (fl(ts) - hh(s))**2).min()
    ok4 = abs(pr - float(S(md_answer(3, "4")))) < 1e-9
    report(3, ok1 and ok2 and ok3 and ok4, f"f={fs}, a={sol[0][a]}, r 경계 {r0}, PQmin={dmin:.10f}, PRmin={pr:.10f}")


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
