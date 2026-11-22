#!/usr/bin/env python3
"""CUK-M26A (가톨릭대 2026학년도 모의논술 자연·공학 계열/간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
mp.mp.dps = 40
V = Verifier("CUK-M26A", "items/CUK/PAST")
X0 = sp.Symbol("x")

def p01():
    # 조건: b=f(a), a=g(b) → a=e^{2b}, b=ln((a+3)/4). 수치로 연립해를 찾는다(여러 초기값).
    f = lambda u: mp.log((u + 3) / 4)
    sols = set()
    for a0 in [0.5, 1, 2, 5, 9, 12, 20]:
        try:
            a = mp.findroot(lambda a: mp.exp(2 * f(a)) - a, a0)
            sols.add((mp.nstr(f(a), 25), mp.nstr(a, 25)))
        except Exception:
            pass
    pts = sorted((mp.mpf(b), mp.mpf(a)) for b, a in sols)   # B=(b,a)
    want = sorted((mp.mpf(str(sp.N(p[0], 30))), mp.mpf(str(sp.N(p[1], 30)))) for p in sp.sympify(V.ans(1, "1"), locals={"log": sp.log}))
    V.report(1, len(pts) == 2 and all(abs(p[0] - q[0]) < 1e-20 and abs(p[1] - q[1]) < 1e-20 for p, q in zip(pts, want)), f"1-1 D {pts} / YAML {want}")
    (b1, a1), (b2, a2) = pts
    h = lambda u: a1 + (a2 - a1) / (b2 - b1) * (u - b1)
    S = mp.quad(lambda u: h(u) - mp.exp(2 * u), [b1, b2])
    V.check(1, "1-2 S", S, sp.N(S_(V.ans(1, "2")), 30), 1e-20)

S_ = lambda s: sp.sympify(s, locals={"log": sp.log})

def p02():
    p, q, r, s = sp.symbols("p q r s")
    F = lambda u: p * u**3 + q * u**2 + r * u + s
    f = sp.Lambda(x, F(x)); fp = sp.diff(F(x), x)
    sol = sp.solve([F(4) - 4, fp.subs(x, 4) - 1, fp.subs(x, 2) - 1, F(2) - F(4) - 2], [p, q, r, s], dict=True)[0]
    fx = F(x).subs(sol)
    ok = sol[p] > 0 and sp.diff(fx - x, x, 2).subs(x, 2) < 0
    V.report(2, ok, f"f 결정: {sp.expand(fx)}")
    a, t = sp.symbols("a t")
    # K 의 원소 = 접점 b: 접선 y=f(b)+f'(b)(x-b) 가 (0,a) 지남 → 원소 개수 = 실근 수
    poly = sp.expand(t * sp.diff(fx, x).subs(x, t) - fx.subs(x, t) + a)
    def nreal(av):
        co = [mp.mpf(str(c)) for c in sp.Poly(poly.subs(a, av), t).all_coeffs()]
        return len({round(float(mp.re(z)), 5) for z in mp.polyroots(co, maxsteps=500, extraprec=200) if abs(mp.im(z)) < 1e-5})
    # L: a 격자에서 원소 수 1 인 집합 ↔ YAML 조건 a>11 or a<-16
    cond = lambda av: av > 11 or av < -16
    bad = [av for av in [sp.Rational(k, 2) for k in range(-80, 81)] if (nreal(av) == 1) != cond(av)]
    V.report(2, not bad, f"2-1 L 격자 불일치 {bad[:3] or '없음'}")
    ystr = V.ans(2, "1").replace("or", "|")
    V.report(2, ystr.replace(" ", "") == "a>11|a<-16", f"YAML L 표기 {ystr}")
    roots = sp.Poly(poly.subs(a, 0), t).nroots(n=30)
    Ssum = sum(sp.re(z) for z in roots if abs(sp.im(z)) < 1e-12)  # a=0: 삼중근 아님, 서로 다른 실근 3개
    V.check(2, "2-2 S", Ssum, V.ans(2, "2"), 1e-20)

def p03():
    # a_n 정의로 직접: a_{n+2}=f(n)a_{n+1}+g(n)a_n (n=3..40) 이 되는 (a,b) 를 연립해로 탐색
    from math import factorial as fa
    an = lambda n: sum(fa(k - 1) * fa(k) for k in range(2, n + 1))
    a_, b_ = sp.symbols("a b")
    eqs = [an(n + 2) - ((n**2 + a_ * n + 3) * an(n + 1) + (-n**2 + b_ * n - 2) * an(n)) for n in range(3, 8)]
    sol = sp.solve(eqs, [a_, b_], dict=True)
    V.report(3, len(sol) == 1, f"3-1 (a,b) 연립해 {sol}")
    fx = X0**2 + sol[0][a_] * X0 + 3; gx = -X0**2 + sol[0][b_] * X0 - 2
    ok_all = all(an(n + 2) == fx.subs(X0, n) * an(n + 1) + gx.subs(X0, n) * an(n) for n in range(3, 60))
    V.report(3, ok_all, "n=3..59 점화식 성립")
    f1, g1 = [sp.sympify(s.split("=")[1]) for s in V.ans(3, "1").split(";")]
    V.check(3, "3-1 f", fx - f1, 0); V.check(3, "3-1 g", gx - g1, 0)
    A = sum(sp.Rational(1) * (fx.subs(X0, n) + gx.subs(X0, n)) / gx.subs(X0, n) for n in range(3, 26))
    V.check(3, "3-2 A", sp.nsimplify(A), V.ans(3, "2"))

if __name__ == "__main__": V.run(globals())
