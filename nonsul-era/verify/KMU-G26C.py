#!/usr/bin/env python3
"""KMU-G26C (국민대 2026학년도 수시 논술 자연계 오후2, 수학 1~8번) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "KMU-G26C"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KMU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def same(expr, no: int, sub: str) -> bool:
    return sp.simplify(sp.nsimplify(expr) - sp.sympify(md_answer(no, sub))) == 0


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


import itertools
import math
import mpmath as mp
n, A_ = sp.symbols("n A", real=True)


def expr_ans(no, sub, **loc):
    return sp.sympify(md_answer(no, sub), locals=loc)


def p01():
    # 정의대로 S_1..S_40 을 점화식으로 생성 → a_k = S_k - S_{k-1}
    S = {1: sp.Integer(3), 2: sp.Integer(3 - 2)}
    for k in range(1, 39):
        S[k + 2] = sp.sqrt(3) * S[k]
    fN, gN = expr_ans(1, "1", n=n), expr_ans(1, "2", n=n)
    ok1 = all(sp.simplify(S[2 * j] - 3**fN.subs(n, j)) == 0 for j in range(1, 20))
    ok2 = all(sp.simplify(S[2 * j + 1] - 3**gN.subs(n, j)) == 0 for j in range(1, 19))
    a = {k: S[k] - S.get(k - 1, 0) for k in S}
    v3 = sp.nsimplify(sp.expand(a[16] + a[17] + a[18]))
    ok3 = a[1] == 3 and a[2] == -2 and same(v3, 1, "3")
    report(1, ok1 and ok2 and ok3, f"f(n)={fN} g(n)={gN} a16+a17+a18={v3}")


def p02():
    def a(k):
        return sp.Rational(1, 3) if k == 1 else sp.log(sp.Rational(3 * (k + 25), k + 26), 27)
    lim = sp.limit(sp.log(3 * (n + 25) / (n + 26), 27), n, sp.oo)
    ok1 = same(lim, 2, "1")
    # 부분합을 수치(고정밀)로 누적 → 정수에 가까운 m 후보만 sympy 로 정확 확인
    mp.mp.dps = 40
    s = mp.mpf(1) / 3
    hits = []
    for m in range(1, 8000):
        if m >= 2:
            s += mp.log(mp.mpf(3 * (m + 25)) / (m + 26)) / mp.log(27)
        r = mp.nint(s)
        if r >= 1 and abs(s - r) < mp.mpf(10)**-25:
            exact = sp.nsimplify(sp.Rational(1, 3) + sum(sp.expand_log(sp.log(sp.Rational(3 * (k + 25), k + 26), 3), force=True) for k in range(2, m + 1)) / 3)
            if exact.is_integer:
                hits.append((m, int(exact)))
    # N≈m/3 이므로 m<8000 이면 N<2026 인 경우를 모두 포함
    Ns = [N for _, N in hits if N < 2026]
    ok2 = hits and same(hits[0][0], 2, "2")
    ok3 = same(sum(Ns), 2, "3")
    report(2, ok1 and ok2 and ok3, f"lim={lim} hits={hits} sum={sum(Ns)}")


def nonright(nn):
    pts = 2 * nn
    cnt = 0
    for tri in itertools.combinations(range(pts), 3):
        right = any((j - i) % pts == nn for i, j in itertools.combinations(tri, 2))
        cnt += not right
    return cnt


def p03():
    c5, c10 = nonright(5), nonright(10)
    ks = list(range(1, 9))
    vals = [3 * nonright(k) for k in ks]
    poly = sp.interpolate(list(zip(ks, vals)), x)
    P = sp.Poly(sp.expand(poly), x)
    Aa, Bb, Cc, Dd = [P.coeff_monomial(x**e) for e in (3, 2, 1, 0)]
    ok = same(c5, 3, "1") and same(c10, 3, "2") and P.degree() == 3 and same(Aa - Bb + Cc - Dd, 3, "3")
    report(3, ok, f"n=5:{c5} n=10:{c10} 3*count={P.as_expr()} A-B+C-D={Aa - Bb + Cc - Dd}")


def p04():
    # 지름이 sec x 인 반원 단면
    Ax = sp.pi / 2 * (1 / sp.cos(x) / 2)**2
    ok1 = sp.simplify(Ax - expr_ans(4, "1", x=x)) == 0
    V = sp.integrate(Ax, (x, 0, sp.pi / 4))
    ok2 = same(sp.simplify(V), 4, "2")
    mp.mp.dps = 50
    I = mp.quad(lambda u: u / mp.cos(u)**4, [0, mp.pi / 4])
    rel = mp.pslq([I, mp.pi, mp.log(2), 1], maxcoeff=10**6, maxsteps=10**6)
    Bv, Cv, Dv = [sp.Rational(-rel[i], rel[0]) for i in (1, 2, 3)]
    ok3 = same(Bv + Cv + Dv, 4, "3")
    report(4, ok1 and ok2 and ok3, f"A(x)={sp.simplify(Ax)} V={V} I: B={Bv} C={Cv} D={Dv}")


def p05():
    f = sp.cos(x**2 + 2 * x) * sp.cos(sp.pi / 2 + x**2) + sp.sin(x**2 + 2 * x) * sp.sin(sp.pi / 2 - x**2)
    v1 = sp.simplify(f.subs(x, sp.pi / 3))
    v2 = sp.limit(f.subs(x, -4 * x) / sp.log(1 + 2 * x), x, 0)
    y = sp.symbols("y", real=True)
    F = x**3 + f.subs(x, y**3) - x * y - sp.exp(y)
    a_roots = [r for r in sp.solve(F.subs(y, 0), x) if r.is_real]
    av = a_roots[0]
    m = sp.simplify((-sp.diff(F, x) / sp.diff(F, y)).subs({x: av, y: 0}))
    ok = same(v1, 5, "1") and same(v2, 5, "2") and len(a_roots) == 1 and same(m - av, 5, "3")
    report(5, ok, f"f(pi/3)={v1} lim={v2} a={av} m={m}")


def p06():
    def cnt(S):
        return sum(1 for A in range(1, S) if A < S - A)
    tot = sum(cnt(2 * k) for k in range(1, 2027))
    fac = sp.factorint(tot)
    inv = {e: p for p, e in fac.items()}
    ok3 = sorted(fac.values()) == [1, 2, 4] and same(inv[4] + inv[2] + inv[1], 6, "3")
    ok = same(cnt(100), 6, "1") and same(cnt(2026), 6, "2") and ok3
    report(6, ok, f"(1)={cnt(100)} (2)={cnt(2026)} sum={tot}={fac}")


def p07():
    t = sp.symbols("t", real=True)
    fx = sp.log(x, 2)**2 + A_ * sp.log(x, sp.Rational(1, 4))
    ft = sp.simplify(sp.expand_log(fx.subs(x, 2**t), force=True))
    ok1 = sp.simplify(ft - expr_ans(7, "1", t=t, A=A_)) == 0
    lo, hi = sp.log(1, 2), sp.log(8, 2)
    ok2 = same(lo + hi, 7, "2")
    # 최솟값 m(A) (A>0) 를 [1,8] 위 촘촘한 수치 + 꼭짓점 후보로 구해 m(A)=-15 를 이분법으로
    fl = sp.lambdify((x, A_), fx)

    def minf(Av):
        cands = [1.0, 8.0]
        xv = 2**(Av / 4)
        if 1 <= xv <= 8:
            cands.append(xv)
        return min(fl(c, Av) for c in cands)
    import numpy as np
    grid = np.linspace(0.001, 200, 200001)
    vals = np.array([minf(g) + 15 for g in grid])
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            a_, b_ = grid[i], grid[i + 1]
            for _ in range(80):
                mid = (a_ + b_) / 2
                if (minf(a_) + 15) * (minf(mid) + 15) <= 0:
                    b_ = mid
                else:
                    a_ = mid
            roots.append(round((a_ + b_) / 2, 9))
    ok3 = len(roots) == 1 and abs(roots[0] - float(md_answer(7, "3"))) < 1e-6
    report(7, ok1 and ok2 and ok3, f"f(t)={ft} t∈[{lo},{hi}] A roots={roots}")


def p08():
    f = sp.cos(x * sp.pi / 180)
    fp = sp.diff(f, x)
    # A sin(Bx), B>0: B = π/(첫 양의 영점), A = (A sin Bx)'(0)/B
    z = sp.nsolve(fp, x, 150)
    B = sp.pi / sp.nsimplify(z)
    Aa = sp.diff(fp, x).subs(x, 0) / B
    ok_form = all(abs(sp.N(fp.subs(x, v) - Aa * sp.sin(B * v))) < 1e-12 for v in (7, 33, 101)) and 0 < B < sp.Rational(1, 10)
    C = sp.nsimplify(4 * sp.cos(45 * Aa + 60 * B), [sp.sqrt(2), sp.sqrt(6)])
    ok1 = ok_form and same(C, 8, "1") and sp.simplify(C - expr_ans(8, "1")) == 0
    K = sp.symbols("K")
    g = sp.integrate(f, x) + K
    g = g.subs(K, sp.solve(g.subs(x, 30), K)[0])
    F = g.subs(x, 0)                     # g-F 는 홀함수
    zE = sp.nsolve(g - F, x, 170)
    E = sp.pi / sp.nsimplify(zE)
    D = sp.diff(g, x).subs(x, 0) / E
    ok_g = all(abs(sp.N(g.subs(x, v) - (D * sp.sin(E * v) + F))) < 1e-9 for v in (7, 33, 101)) and 0 < E < sp.Rational(1, 10)
    ok2 = ok_g and same(sp.simplify((D + F) * E), 8, "2")
    report(8, ok1 and ok2, f"A={Aa} B={B} C={C} D={sp.simplify(D)} E={E} F={sp.simplify(F)} (D+F)E={sp.simplify((D + F) * E)}")


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
