#!/usr/bin/env python3
"""CUK-G24B (가톨릭대 2024 수시 논술 의예·약학 1~4) — 문제 조건에서 독립 재계산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G24B", "items/CUK/PAST")
mp.mp.dps = 30


def p01():
    xf = lambda t: 2 * mp.sin(t) - (2 * t - mp.pi) * mp.cos(t)
    yf = lambda t: 2 * mp.cos(t) + (2 * t - mp.pi) * mp.sin(t)
    sp_ = lambda t: mp.sqrt(mp.diff(xf, t) ** 2 + mp.diff(yf, t) ** 2)       # 속력(수치미분)
    # a: 속력이 t=0 의 3배
    a = mp.findroot(lambda t: sp_(t) - 3 * sp_(0), 6)
    vq = lambda t: mp.diff(xf, t)          # Q=(x,0) 속도 = dx/dt
    vr = lambda t: mp.diff(yf, t)          # R=(0,y) 속도 = dy/dt
    def turns(v):                          # 부호가 실제로 바뀌는 시각(수치 탐색)
        out, prev, N = [], None, 40000
        for i in range(1, N):
            t = mp.mpf(i) * 8 / N
            sg = mp.sign(v(t)) if abs(v(t)) > 1e-12 else 0
            if sg != 0:
                if prev is not None and sg != prev:
                    out.append(round(float(t), 3))
                prev = sg
        return out
    tq, tr = turns(vq), turns(vr)
    rq = [mp.findroot(vq, x0) for x0 in tq[:2]]       # Q 의 방향 전환 시각(첫째, 둘째)
    rr = [mp.findroot(vr, x0) for x0 in tr[:2]]       # R 의 방향 전환 시각
    b, c = rq[1], rr[1]
    dist = lambda v, T, kn: mp.quad(lambda t: abs(v(t)), [mp.mpf(0)] + list(kn) + [T])
    s = dist(sp_, a, [mp.pi / 2])
    d = dist(vq, b, rq[:1])
    l = dist(vr, c, rr[:1])
    V.report(1, abs(a - 2 * mp.pi) < 1e-9 and abs(b - mp.pi) < 1e-9 and abs(c - 5 * mp.pi / 2) < 1e-9, f"a={mp.nstr(a,8)} b={tq[:3]} c={tr[:3]}")
    exp = dict(p.split("=", 1) for p in V.ans(1, "1").replace(" ", "").split(";"))
    for k, val in (("s", s), ("d", d), ("l", l)):
        e = sp.N(sp.sympify(exp[k]), 20)
        V.report(1, abs(val - mp.mpf(str(e))) < 1e-12, f"{k}: 계산 {mp.nstr(val,15)} / YAML {sp.N(e,15)}")


def p02():
    X, k_ = sp.symbols("x k", real=True)
    t = sp.symbols("t", real=True)
    f = 2 * sp.exp(2 * t) * (X - t) + sp.exp(2 * t)                       # 접선
    g1 = f + k_ - 2 * sp.log(X)
    xc = sp.solve(sp.diff(g1, X), X)[0]                                    # 극점
    mt = sp.solve(sp.Eq(sp.simplify(g1.subs(X, xc)), 0), k_)[0]            # 극솟값=0 이 되는 최소 k
    h = sp.simplify(sp.exp(-t) * (mt + 2 * t + 1))
    V.report(2, sp.simplify(h - (sp.exp(t) * (2 * t - 1) - sp.exp(-t) * (2 * t + 1))) == 0, f"h(t)={h}")
    hf = sp.lambdify(t, h, "mpmath")
    F = lambda a, b: mp.quad(lambda u: (u - 1) ** 2 * hf(u), [a, b])
    # h 의 영점: 격자 탐색
    grid = [mp.mpf(i) / 50 for i in range(-500, 501)]
    zs = [mp.findroot(hf, (grid[i] + grid[i + 1]) / 2) for i in range(len(grid) - 1) if hf(grid[i]) * hf(grid[i + 1]) < 0]
    q = max(zs); nq = min(zs)
    ok = len(zs) == 2 and abs(q + nq) < 1e-20 and hf(0) < 0
    # s 최소: (-q,q) 에서 음, 밖에서 양 — 교란 비교
    base = F(nq, q)
    pert = [F(nq + dx, q + dy) for dx in (-0.1, 0, 0.1) for dy in (-0.1, 0, 0.1) if (dx, dy) != (0, 0)]
    ok = ok and all(base < v for v in pert) and F(nq, q) == min([base] + pert)
    V.check(2, "α+β", sp.nsimplify(float(nq + q), rational=True, tolerance=1e-12), V.ans(2, "1")) if ok else V.report(2, False, "영점·최소 확인 실패")


def p03():
    sn = lambda n: sum(mp.sin(mp.mpf(n + j) / 100) for j in range(1, 101))
    vals = {n: sn(n) for n in range(1, 501)}
    m = max(vals, key=vals.get)
    b = sum(mp.cos(mp.mpf(m + j) / 100) for j in range(1, 101))
    A = [kk for kk in range(-50, 50) if 2 * b - kk > 0]
    V.report(3, m == 107 and -0.5 < b < 0, f"m={m}, b={mp.nstr(b,8)}")
    V.check(3, "max A", max(A), V.ans(3, "1"))


def p04():
    def fval(a, x):
        return x ** 3 - a * x ** 2 - a * a * x + 4 * a + 2 * a * a if x >= 0 else -(a + 4) ** 2 * x + 4 * a + 2 * a * a
    def g(a, k):
        cand = [-k, k, 0.0]
        for c in (a, -a / 3):                       # x>=0 의 극점
            if 0 < c < k:
                cand.append(c)
        return max(fval(a, x) for x in cand)
    def minimizers(k):
        lo, hi, h = -40.0, 40.0, 1e-3
        n = int((hi - lo) / h)
        gs = [g(lo + i * h, k) for i in range(n + 1)]
        gm = min(gs)
        # 격자 최솟값 근방을 3분법으로 정밀화
        def refine(a0):
            l, r = a0 - h, a0 + h
            for _ in range(80):
                m1, m2 = l + (r - l) / 3, r - (r - l) / 3
                if g(m1, k) <= g(m2, k): r = m2
                else: l = m1
            return (l + r) / 2
        cands = [i for i in range(n + 1) if gs[i] <= gm + 1e-3]
        pts = []
        for i in cands:
            if (i == 0 or gs[i] <= gs[i - 1]) and (i == n or gs[i] <= gs[i + 1]):
                pts.append(refine(lo + i * h))
        m = min(g(p, k) for p in pts)
        sel = sorted({round(p, 3) for p in pts if g(p, k) <= m + 1e-9})
        return sel
    notA = []
    for kk in [x / 4 for x in range(1, 41)] + [1.99, 2.01, 3.99, 4.01]:
        if len(minimizers(kk)) != 1:
            notA.append(kk)
    exp = sorted(float(t) for t in V.ans(4, "1").split(","))
    V.report(4, notA == exp, f"A 에 속하지 않는 k(격자)={notA}")


if __name__ == "__main__":
    V.run(globals())
