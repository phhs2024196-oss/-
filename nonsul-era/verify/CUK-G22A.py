#!/usr/bin/env python3
"""CUK-G22A (가톨릭대 2022 수시 논술 자연·공학/간호 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
V = Verifier("CUK-G22A", "items/CUK/PAST")


def p01():
    x = sp.symbols("x")
    sols = []
    # 자연수 m,n,l(n>l), 정수 a,b,c 탐색: f(1)=a+b=8, g(1)=1+c=2 는 직접 적용, 극한 조건은 수치 선별 후 sympy 로 확정
    for m in range(1, 7):
        for n in range(2, 9):
            for l in range(1, n):
                for a in range(-12, 13):
                    b, c = 8 - a, 2 - 1
                    if a == 0 or b == 0:
                        continue
                    f = a * x ** (m + 1) + b * x ** m
                    g = x ** n + c * x ** l
                    fn = sp.lambdify(x, [sp.diff(g, x) / f, x * g / f], "mpmath")
                    big, small = fn(sp.Float(10) ** 15), fn(sp.Float(10) ** -15)
                    if abs(big[0] - 1) > 1e-3 or abs(small[1] - 0.25) > 1e-3:
                        continue
                    if sp.limit(sp.diff(g, x) / f, x, sp.oo) == 1 and sp.limit(x * g / f, x, 0) == sp.Rational(1, 4):
                        sols.append((m, n, l, a, b, c))
    V.report(1, len(sols) == 1, f"조건을 만족하는 (m,n,l,a,b,c)={sols}")
    m, n, l, a, b, c = sols[0]
    exp = {k: int(v) for k, v in (p.split("=") for p in V.ans(1, "1").replace(" ", "").split(";"))}
    V.report(1, (a, b, c) == (exp["a"], exp["b"], exp["c"]), f"a,b,c={a},{b},{c} / YAML {exp}")
    f = a * x ** (m + 1) + b * x ** m
    g = x ** n + c * x ** l
    V.check(1, "논제2 L", sp.limit(g / f, x, -1), V.ans(1, "2"))


def p02():
    # 정사각형 둘레 4: 동전 앞 3·뒤 2, 8회 후 A 로 복귀 = 이동 합이 4의 배수
    a = sum(1 for s in itertools.product((3, 2), repeat=8) if sum(s) % 4 == 0)
    V.check(2, "논제1 a", a, V.ans(2, "1"))
    # 정팔각형 둘레 8: 구슬 3·2·1 중복 허용 8회(순서 있음), 합이 8의 배수
    b = sum(1 for s in itertools.product((3, 2, 1), repeat=8) if sum(s) % 8 == 0)
    V.check(2, "논제2 b", b, V.ans(2, "2"))


def p03():
    x, p, q, c = sp.symbols("x p q c", real=True)
    f = x ** 3 + p * x ** 2 + q * x + 1                       # 최고차 1, f(0)=1
    pv = sp.solve(sp.diff(f, x, 2).subs(x, 0), p)[0]   # f 의 도함수가 x=0 에서 극값 -> f''(0)=0
    f = f.subs(p, pv)
    x0 = sp.symbols("x0", real=True)
    sol = sp.solve([f.subs(x, x0) - 3, sp.diff(f, x).subs(x, x0)], [x0, q], dict=True)   # 극점 x0 에서 극값 3
    qs = sorted({v[q] for v in sol if v[q] < 0})                                         # 극값을 가지려면 q<0
    V.report(3, len(qs) == 1, f"q 후보={qs}")
    f = f.subs(q, qs[0])
    fn = sp.lambdify(x, f, "numpy")
    def roots(k):
        cf = sp.Poly(f - k, x).all_coeffs()
        r = np.roots([float(t) for t in cf])
        return sorted(float(z.real) for z in r if abs(z.imag) < 1e-7)
    # 방정식 f(x)=k 의 서로 다른 실근 개수 n(k)
    def cnt(k):
        r = roots(k)
        return len({round(z, 6) for z in r})
    ks = np.round(np.arange(-3, 5.0001, 0.005), 6)
    two = [k for k in ks if cnt(k) >= 2]
    a_, b_ = min(two), max(two)
    ok = all((cnt(k) >= 2) == (a_ <= k <= b_) for k in ks)
    V.report(3, ok, f"n>=2 인 k 범위 [{a_}, {b_}]")
    V.check(3, "논제1 a", a_, V.ans(3, "1", "a"), 1e-9)
    V.check(3, "논제1 b", b_, V.ans(3, "1", "b"), 1e-9)
    ds = []
    for k in np.linspace(a_, b_, 4001):
        r = roots(k)
        ds.append(r[-1] - r[0])
    M, m = max(ds), min(ds)
    exp = float(sp.N(sp.sympify(V.ans(3, "2"))))
    V.report(3, abs(M / m - exp) < 1e-5, f"M={M:.6f} m={m:.6f} M/m={M/m:.7f} / YAML {exp:.7f}")


if __name__ == "__main__":
    V.run(globals())
