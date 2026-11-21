#!/usr/bin/env python3
"""SJU-G18B (세종대 2018학년도 수시 논술고사[B]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SJU-G18B"
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


def p01():
    import mpmath as mp
    mp.mp.dps = 30
    Av = mp.quad(lambda u: mp.e**(u + u**2), [0, 1])
    Asym = sp.Symbol("A")
    I1 = mp.quad(lambda u: 2 * u * mp.e**(u + u**2), [0, 1])
    ok1 = abs(I1 - mp.mpf(str(sp.N(S(md_answer(1, "1"), A=Asym).subs(Asym, sp.Float(str(Av), 30)), 30)))) < 1e-20
    f = lambda u: mp.e**(u + u**2) + mp.e**(u + mp.sqrt(u))
    I2 = mp.quad(f, [0, 1])
    ok2 = abs(I2 - mp.mpf(str(sp.N(S(md_answer(1, "2")), 30)))) < 1e-15
    ok0 = abs(f(0) - 2) < 1e-25 and abs(f(1) - 2 * mp.e**2) < 1e-20
    def ginv(v):                                     # f 는 증가 → 이분법 + 뉴턴
        lo, hi = mp.mpf(0), mp.mpf(1)
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < v else (lo, mid)
        return mp.findroot(lambda u: f(u) - v, (lo + hi) / 2) if lo > 1e-12 else (lo + hi) / 2
    I3 = mp.re(mp.quad(ginv, [2, 2 * mp.e**2]))
    ok3 = abs(I3 - mp.mpf(str(sp.N(S(md_answer(1, "3")), 30)))) < 1e-10
    report(1, ok0 and ok1 and ok2 and ok3, f"∫2xe^(x+x²)={mp.nstr(I1, 12)} ∫f={mp.nstr(I2, 12)} ∫g={mp.nstr(I3, 12)} (f(0)=2)")


def p02():
    Am = sp.Matrix([0, 0, 0]); B = sp.Matrix([1, 0, 0]); C = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 2, 0])
    D = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 6, sp.sqrt(6) / 3])
    b, c, d = B - Am, C - Am, D - Am
    bb, cc, dd = sp.symbols("b c d")
    to_vec = lambda ex: sp.Matrix([sp.expand(ex).coeff(bb) * b[i] + sp.expand(ex).coeff(cc) * c[i] + sp.expand(ex).coeff(dd) * d[i] for i in range(3)])
    H = sp.Matrix([D[0], D[1], 0])                     # 평면 ABC: z=0
    DH_md = S(md_answer(2, "1").split("=", 1)[1], b=bb, c=cc, d=dd)
    ok1 = sp.simplify(to_vec(DH_md) - (H - D)) == sp.zeros(3, 1)
    P = (B + 2 * C) / 3; Q = (Am + P) / 2
    pr = md_parts(2, "2")
    DQ_md = S(pr["DQ"], b=bb, c=cc, d=dd)
    ok2 = sp.simplify(to_vec(DQ_md) - (Q - D)) == sp.zeros(3, 1) and sp.simplify((Q - D).norm() - S(pr["|DQ|"])) == 0
    import numpy as np
    ts = np.linspace(0, 1, 1000001)
    Pn = np.array([float(v) for v in P]); Dn = np.array([float(v) for v in D])
    L2 = ((ts[:, None] * Pn - Dn)**2).sum(1)
    ok3 = abs(L2.min() - float(S(md_answer(2, "3")))) < 1e-10
    report(2, ok1 and ok2 and ok3, f"|DQ|={sp.simplify((Q - D).norm())} minL²={L2.min():.10f}")


def p03():
    import numpy as np
    a, b, c = sp.symbols("a b c")
    f = a * sp.exp(x) + x**2 + b * x + c
    sol = sp.solve([f.subs(x, 0), f.subs(x, 1)], [b, c], dict=True)[0]
    pr2 = md_parts(3, "2")
    ok2 = sp.simplify(sol[b] - S(pr2["b"], a=a)) == 0 and sp.simplify(sol[c] - S(pr2["c"], a=a)) == 0
    pr1 = md_parts(3, "1")
    ok1 = S(pr1["f(0)"]) == 0 and S(pr1["f(1)"]) == 0
    fa = sp.lambdify((x, a), f.subs(sol))
    ga = sp.lambdify((x, a), sp.diff(f.subs(sol), x))
    xs = np.linspace(-6, 6, 6001); xi = np.linspace(1e-6, 1 - 1e-4, 200001)

    def valid(av):
        if np.any(xs * (xs - 1) * fa(xs, av) < -1e-12):
            return False
        phi = ga(xi, av) - ((av + 4) * xi - 3)
        return bool(np.any(phi[:-1] * phi[1:] <= 0))
    # (가) 는 b,c 가 정해지지 않으면 깨지는지 확인: f(0)≠0 인 예
    ok1 &= np.any(xs * (xs - 1) * sp.lambdify(x, f.subs({a: 1, b: 0, c: 1}))(xs) < 0)
    grid = np.linspace(0.05, 5, 9901)
    vs = [av for av in grid if valid(av)]
    lo_md, hi_md = [float(S(p, a=a)) for p in md_answer(3, "3").split("<") if "a" not in p.replace("exp", "")]
    ok3 = abs(min(vs) - lo_md) < 1e-3 and abs(max(vs) - hi_md) < 1e-3
    # (4) 접선 y절편 k, (4-4e)/(k+1) 자연수인 a 를 유효 a 중에서 탐색
    g = sp.diff(f.subs(sol), x)
    k = sp.simplify(g.subs(x, 1) - sp.diff(g, x).subs(x, 1))
    V = sp.simplify((4 - 4 * sp.E) / (k + 1))
    found = set()
    for m in range(1, 30):
        for av in sp.solve(sp.Eq(V, m), a):
            if av.is_real and av > 0 and valid(float(av)):
                found.add(sp.nsimplify(av))
    md4 = {S(v.split("=")[1]) for v in md_answer(3, "4").split(";")}
    ok4 = found == md4
    report(3, ok1 and ok2 and ok3 and ok4, f"b,c={sol} a∈({min(vs):.4f},{max(vs):.4f}) k={k} → a∈{sorted(found)}")


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
