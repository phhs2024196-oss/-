#!/usr/bin/env python3
"""KU-G26A (고려대 2026학년도 논술 자연계열 오전) — sympy 검산. 정답 하드코딩 금지: 문제 조건에서 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KU-G26A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no, sub):
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s):
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    n = sp.symbols("n", positive=True)
    a, b = sp.symbols("a b", positive=True)
    # (가) 넓이를 직접 적분: 교점은 x^2-a x-1=0의 근
    r1, r2 = sorted(sp.solve(x**2 - a * x - 1, x), key=lambda e: float(e.subs(a, 1)))
    area = sp.simplify(sp.integrate(a * x + 1 - x**2, (x, r1, r2)))
    a2 = sp.solve(sp.Eq(area, n / 6), a)[0] ** 2              # a_n^2
    L1 = sp.limit(a2 / n**sp.Rational(2, 3), n, sp.oo)
    an2 = sp.Lambda(n, a2)
    L2 = sp.limit(n**sp.Rational(1, 3) * (an2(n + 1) - an2(n)), n, sp.oo)
    # (나) 거리^2 직접 계산
    s1, s2 = sorted(sp.solve(x**2 - b * x - 1, x), key=lambda e: float(e.subs(b, 1)))
    d2 = sp.expand((s2 - s1)**2 + (s2**2 - s1**2)**2)
    bsq = sp.symbols("B", positive=True)
    poly = sp.simplify(d2.subs(b, sp.sqrt(bsq)))               # (B+4)(B+1)
    Bsol = [s for s in sp.solve(sp.Eq(poly, n**2), bsq) if s.subs(n, 20).is_positive][0]
    L3 = sp.limit(Bsol / n, n, sp.oo)
    ok = (sp.simplify(L1 - sp.sympify(md_answer(1, "1"))) == 0 and
          sp.simplify(L2 - sp.sympify(md_answer(1, "2"))) == 0 and
          sp.simplify(L3 - sp.sympify(md_answer(1, "3"))) == 0)
    # 수치 확인 n=10^6
    nn = 10**6
    ok = ok and abs(float(a2.subs(n, nn) / nn**(2/3)) - 1) < 1e-3
    report(1, ok, f"lim1={L1} lim2={L2} lim3={L3} area={area}")


def p02():
    a, b, c, d = sp.symbols("a b c d", real=True)
    g = -x**2 + a * x + b
    h = x**2 + c * x + d
    # (나): 분자→0, 그리고 lim (1/x)∫_0^x h = h(0) = 0
    eq1 = sp.integrate(h, (x, -1, 0)) + sp.Rational(1, 6)
    eq2 = h.subs(x, 0)
    # (가): 연속, 좌우미분계수
    fR = sp.exp(-x) * g - 1
    eq3 = fR.subs(x, 0) - h.subs(x, 0)
    eq4 = sp.diff(fR, x).subs(x, 0) - sp.diff(h, x).subs(x, 0)
    sol = sp.solve([eq1, eq2, eq3, eq4], [a, b, c, d], dict=True)[0]
    G, H = g.subs(sol), h.subs(sol)
    # (나) 원식 확인(양쪽 극한)
    f_int = lambda X: sp.integrate(H, (x, -1, 0)) + sp.integrate(sp.exp(-x) * G - 1, (x, 0, X))
    chk = sp.limit((f_int(t) + sp.Rational(1, 6)) / t, t, 0, "+")
    md = parts(md_answer(2, "1"))
    ok1 = chk == 0 and sp.expand(G - sp.sympify(md["g"], locals={"x": x})) == 0 and \
        sp.expand(H - sp.sympify(md["h"], locals={"x": x})) == 0
    # (2-2) k(t) = -t f'(t)+f(t); 접점 개수를 수치로 세어 범위 확인
    fr = sp.exp(-x) * G - 1
    kR = sp.simplify(-x * sp.diff(fr, x) + fr)
    kL = sp.simplify(-x * sp.diff(H, x) + H)
    crit = sp.solve(sp.diff(kR, x) * sp.exp(x), x)
    kmin = kR.subs(x, 5)
    lim_inf = sp.limit(kR, x, sp.oo)
    import numpy as np
    kRf, kLf = sp.lambdify(x, kR, "numpy"), sp.lambdify(x, kL, "numpy")
    xsR = np.linspace(0, 80, 400001); xsL = np.linspace(-60, 0, 200001)[:-1]
    def count(k):
        cnt = 0
        for xs, fn in ((xsR, kRf), (xsL, kLf)):
            v = fn(xs) - k
            cnt += int(np.sum(np.sign(v[:-1]) * np.sign(v[1:]) < 0)) + int(np.sum(v == 0))
        return cnt
    ks = np.linspace(-3, 1, 801)
    three = [k for k in ks if count(k) == 3]
    md_set = sp.sympify(md_answer(2, "2"))
    ok2 = (abs(min(three) - float(kmin)) < 0.01 and abs(max(three) - (-1)) < 0.01 and lim_inf == -1
           and md_set.left_open and md_set.right_open and sp.simplify(md_set.start - kmin) == 0
           and sp.simplify(md_set.end - lim_inf) == 0 and sorted(crit) == [0, 1, 5])
    # (2-3)
    al, be = sorted(sp.solve(G - H, x))
    import mpmath as mp
    ph = sp.lambdify(x, sp.Abs(G - H), "mpmath")
    grid = [i / 200 for i in range(-800, 801)]
    xm = min(grid, key=lambda X: mp.quad(ph, sorted({X, X + float(be - al)} | {u for u in (float(al), float(be)) if X < u < X + float(be - al)})))
    Fmin = sp.Abs(sp.integrate(H - G, (x, al, be)))   # 두 근 사이 |g-h| 적분 = F(alpha)
    md3 = parts(md_answer(2, "3"))
    ok3 = abs(xm - float(al)) < 0.01 and sp.simplify(sp.sympify(md3["x0"]) - al) == 0 and \
        sp.simplify(sp.sympify(md3["F"]) - Fmin) == 0
    report(2, ok1 and ok2 and ok3,
           f"g={G} h={H} k-range=({float(kmin):.5f},-1) numeric3=[{min(three):.3f},{max(three):.3f}] x0={al} F={Fmin} argmin~{xm}")


def p03():
    a, b, c, d = sp.symbols("a b c d", real=True)
    A, B = sp.Matrix([0, 0, sp.Rational(11, 3)]), sp.Matrix([0, 0, 11])
    P, Q = sp.Matrix([a, 0, b]), sp.Matrix([0, c, d])
    sP = [s for s in sp.solve([P.dot(P) - 11, P.dot(P - A)], [a, b], dict=True) if s[a] > 0][0]
    sQ = [s for s in sp.solve([Q.dot(Q) - 11, Q.dot(Q - B)], [c, d], dict=True) if s[c] > 0][0]
    P, Q = P.subs(sP), Q.subs(sQ)
    R, S = sp.Matrix([P[0], 0, 0]), sp.Matrix([0, Q[1], 0])
    RS = sp.sqrt((R - S).dot(R - S))
    u, v = Q - P, S - R
    th = sp.acos(sp.Abs(u.dot(v)) / sp.sqrt(u.dot(u) * v.dot(v)))
    md = parts(md_answer(3, "1"))
    ok1 = sp.simplify(RS - sp.sympify(md["RS"])) == 0 and sp.simplify(th - sp.sympify(md["theta"])) == 0
    C2 = sp.Matrix([sp.sqrt(2), 6, sp.sqrt(10)])
    s = sp.symbols("s", real=True)
    M = R + s * v
    dist2 = sp.expand((M - C2).dot(M - C2))
    dmin = sp.sqrt(sp.simplify(dist2.subs(s, sp.solve(sp.diff(dist2, s), s)[0])))
    rmin = sp.simplify(dmin - 2)                        # dmin > 2 이므로 m은 S2와 떨어져 있음
    V = sp.Rational(4, 3) * sp.pi * rmin**3
    ok2 = dmin > 2 and sp.simplify(V - sp.sympify(md_answer(3, "2"))) == 0
    report(3, ok1 and ok2, f"P={list(P)} Q={list(Q)} RS={RS} theta={th} dist={dmin} V={V}")


def p04():
    total = 0
    # 정의대로: 특수 정확히 1, 대문자 정확히 1, 나머지 3은 숫자/소문자(36종) — 자리 패턴 전수
    for pat in itertools.product("SUO", repeat=5):
        if pat.count("S") == 1 and pat.count("U") == 1:
            total += 32 * 26 * 36**3
    cands = [(total // k**3, k) for k in range(31, 40) if total % k**3 == 0]
    ok1 = len(cands) == 1 and cands[0][0] - cands[0][1] == int(md_answer(4, "1"))
    cnt = sum(1 for p in itertools.product(range(10), repeat=5) if sum(p) == 38)
    ok2 = cnt == int(md_answer(4, "2"))
    report(4, ok1 and ok2, f"total={total} (n,k)={cands} count38={cnt} wrong(no cap)={sp.binomial(42, 4)}")


if __name__ == "__main__":
    fns = sorted((nm for nm in dir() if re.fullmatch(r"p\d+", nm)), key=lambda s: int(s[1:]))
    for nm in fns:
        try:
            globals()[nm]()
        except Exception as e:  # noqa: BLE001
            report(int(nm[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
