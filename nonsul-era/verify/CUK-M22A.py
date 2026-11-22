#!/usr/bin/env python3
"""CUK-M22A (가톨릭대 2022학년도 모의논술 자연공학계열·간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, re, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-M22A", "items/CUK/PAST")
mp.mp.dps = 30

def p01():
    # 최댓값 조건: c+b=9, a+c=8 (a,b,c>0) → b=a+1, 0<a<8. 2π/a+b/(2π) 를 수치로 최소화
    g = lambda a: 2 * mp.pi / a + (a + 1) / (2 * mp.pi)
    a0 = mp.findroot(lambda a: mp.diff(g, a), 6)
    m = g(a0)
    al, be = a0, a0 + 1
    c0 = 8 - a0
    V.report(1, 0 < a0 < 8 and c0 > 0 and be > 0, "1-1 a,b,c 양수")
    V.check(1, "1-1 m", sp.Float(m, 20), sp.N(sp.sympify(V.ans(1, "1", "m")), 20), 1e-12)
    V.check(1, "1-1 alpha", sp.Float(al, 20), sp.N(sp.sympify(V.ans(1, "1", "alpha")), 20), 1e-10)
    V.check(1, "1-1 beta", sp.Float(be, 20), sp.N(sp.sympify(V.ans(1, "1", "beta")), 20), 1e-10)
    # 논제 2: tan(αx-β) 의 점근선 = cos(αx-β)=0 의 해. 절댓값이 가장 작은 것
    cosf = lambda X: mp.cos(al * X - be)
    roots = set()
    for i in range(-80, 80):
        try:
            r = mp.findroot(cosf, mp.mpf(i) / 8)
            if abs(r) < 6: roots.add(round(float(r), 12))
        except Exception: pass
    roots = sorted(roots)
    # 인접 점근선 간격 = π/α = 1/2 확인
    V.report(1, all(abs((roots[i + 1] - roots[i]) - 0.5) < 1e-9 for i in range(len(roots) - 1)), "1-2 점근선 간격 1/2")
    p = min(roots, key=abs)
    V.check(1, "1-2 p", sp.Float(p, 12), sp.N(sp.sympify(V.ans(1, "2")), 12), 1e-10)

def p02():
    cs = sp.symbols("c0:6")
    f = sum(cs[i] * x**i for i in range(6))                   # 일반 5차 다항식
    h = sp.Symbol("h", positive=True)
    right = sp.limit((f.subs(x, h) - f.subs(x, 0)) / h, h, 0)        # x→0+ 에서 f(|x|)=f(x)
    left = sp.limit((f.subs(x, h) - f.subs(x, 0)) / (-h), h, 0)      # x=-h→0-: f(|x|)=f(h), 분모 -h
    sol = sp.solve(sp.Eq(left, right), cs[1])
    V.check(2, "2-1 f'(0)", sol[0], V.ans(2, "1"))
    # 논제 2: g=x^3+p x^2+q x+r, y=g(|x|) 가 x=0 미분가능 ⇒ q=0, x=3 에서 극솟값 6
    p, q, r = sp.symbols("p q r")
    G = x**3 + p * x**2 + q * x + r
    s = sp.solve([sp.diff(G, x).subs(x, 0), sp.diff(G, x).subs(x, 3), G.subs(x, 3) - 6], [p, q, r], dict=True)[0]
    Gs = G.subs(s)
    V.report(2, sp.diff(Gs, x, 2).subs(x, 3) > 0, "2-2 x=3 에서 극소(g''(3)>0)")
    V.check(2, "2-2 g(1)", Gs.subs(x, 1), V.ans(2, "2"))

def p03():
    bad = []
    for m in range(1, 9):
        for N in range(1, 9):
            if sp.integrate(x**m * (1 - x)**N, (x, 0, 1)) != sp.Rational(sp.factorial(m) * sp.factorial(N), sp.factorial(m + N + 1)):
                bad.append((m, N))
    V.proof(3, "3-1 공식 p(n)", bad)
    c = sp.Symbol("c", real=True)
    A = set()
    for N in range(1, 16):
        I = sp.integrate(x**N * (1 - x)**N * (1 + 2 * sp.sqrt(3) * c * x + sp.Rational(27, 10) * c**2 * x**2), (x, 0, 1))
        P = sp.Poly(sp.expand(I), c)
        a2, a1, a0 = P.all_coeffs()
        if sp.simplify(a1**2 - 4 * a2 * a0) < 0 and a2 > 0: A.add(N)     # 모든 c 에서 양수
    want = {int(v) for v in re.findall(r"\d+", V.ans(3, "2"))}
    V.report(3, A == want, f"3-2 A 계산 {sorted(A)} / YAML {sorted(want)}")

if __name__ == "__main__": V.run(globals())
