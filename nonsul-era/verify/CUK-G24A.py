#!/usr/bin/env python3
"""CUK-G24A (가톨릭대 2024 수시 논술 자연·공학/간호 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G24A", "items/CUK/PAST")


def p01():
    # 변이 자연수인 모든 삼각형을 전수 조사: 조건식을 각으로 직접 판정
    N = 80
    cr, A2 = set(), set()
    for a in range(1, N):
        for b in range(1, N):
            for c in range(abs(a - b) + 1, min(a + b, N)):
                s = (a + b + c) / 2
                area = math.sqrt(s * (s - a) * (s - b) * (s - c))
                A = math.acos((b * b + c * c - a * a) / (2 * b * c))
                B = math.acos((a * a + c * c - b * b) / (2 * a * c))
                C = math.pi - A - B
                if abs(math.sin(C) * math.cos(B + C) + math.sin(B)) > 1e-9:
                    continue
                R = a * b * c / (4 * area)
                cr.add(round(c / R, 9))                       # 논제 1
                r = area / s
                if abs(r - 2) > 1e-9:
                    continue
                # 접점 P: AP = s-a, PB = s-b (AB=c 위)
                ratio = (s - a) / (s - b)
                if any(abs(ratio - m / n) < 1e-9 for m in (1, 2, 3) for n in (1, 2, 3)):
                    A2.add(a)
    V.report(1, cr == {float(V.ans(1, "1"))}, f"c/R={cr}")
    V.report(1, A2 == {int(t) for t in V.ans(1, "2").split(",")}, f"a 집합={sorted(A2)}")


def p02():
    a, b, c, d, X = sp.symbols("a b c d X", real=True)
    f = a * X ** 3 + b * X ** 2 + c * X + d
    eqs = sp.Poly(sp.expand(f + f.subs(X, -X) - 4), X).coeffs()
    s1 = sp.solve(eqs, [b, d], dict=True)[0]
    s2 = sp.solve(sp.Eq(sp.diff(f, X).subs(X, 1) - sp.diff(f, X).subs(X, 0), -3).subs(s1), a)
    V.check(2, "최고차항 계수", s2[0], V.ans(2, "1"))
    fx = (-X ** 3 + c * X + 2)
    cc = sp.symbols("cc", positive=True)
    t = sp.sqrt(cc / 3)
    M1 = sp.simplify(fx.subs(X, t).subs(c, cc)); m1 = sp.simplify(fx.subs(X, -t).subs(c, cc))
    def A(k, cv):
        mi, ma = m1.subs(cc, cv), M1.subs(cc, cv)
        if sp.simplify(k - ma) == 0 or sp.simplify(k - mi) == 0:
            return 2
        return 3 if mi < k < ma else 1
    # 경계 c: 극값이 0, 18, 20 이 되는 c
    bps = set()
    for kv in (0, 18, 20):
        for ex in (M1, m1):
            for r in sp.solve(sp.Eq(ex, kv), cc):
                if r.is_real and r > 0:
                    bps.add(sp.nsimplify(r))
    bps = sorted(bps, key=float)
    pts = []
    edges = [sp.Integer(0)] + bps + [bps[-1] + 10]
    for i in range(len(edges) - 1):
        pts.append((edges[i] + edges[i + 1]) / 2)
    pts += bps
    ok = []
    for cv in pts:
        a0, a18, a20 = A(0, cv), A(18, cv), A(20, cv)
        if a0 >= a18 >= a20 and a0 - a18 == a20:
            ok.append(cv)
    Is = sorted({sp.integrate(fx.subs(c, cv), (X, 0, 2)) for cv in ok})
    exp = sorted(sp.sympify(v) for v in V.ans(2, "2").split(","))
    V.report(2, Is == exp and len(ok) == len(exp), f"c={ok}, I={Is}")


def p03():
    a = sp.symbols("a", positive=True)
    # (1) 순서쌍 개수 전수: b=-2a/l 일 때 k+m=l+2, k<m 의 개수
    good = []
    for l in range(1, 40):
        cnt = sum(1 for k in range(1, l + 3) for m in range(k + 1, l + 3) if k + m == l + 2)
        if cnt in (1, 2):
            good.append(l)
    d = max(-2 * a / l for l in good)        # 가장 큰 b
    # a_k+a_m=0 이 실제로 성립하는지 a=7 로 직접 확인
    av = sp.Integer(7)
    bchk = -2 * av / max(good)
    pairs = [(k, m) for k in range(1, 40) for m in range(k + 1, 40) if 2 * av + bchk * (k + m - 2) == 0]
    V.report(3, len(pairs) in (1, 2), f"l={max(good)} 순서쌍 {pairs}")
    V.check(3, "d", d, sp.sympify(V.ans(3, "1").split("=", 1)[1], locals={"a": a}))
    mp.mp.dps = 30
    def S(av):
        av = mp.mpf(av); dd = -av / 2
        bn = lambda n: av + dd * (n - 1)
        return mp.fsum(1 / (mp.sqrt(abs(bn(k))) + mp.sqrt(abs(bn(k + 1)))) for k in range(1, 53))
    a_exp = mp.mpf(V.ans(3, "2"))
    V.report(3, abs(S(a_exp) - 3) < 1e-20 and all(abs(S(v) * mp.sqrt(v) - 12) < 1e-20 for v in (1, 4, 9, 25)), "S(16)=3, S=12/√a")


if __name__ == "__main__":
    V.run(globals())
