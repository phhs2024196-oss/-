"""KW-M18 검산: 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, itertools, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-M18", "items/KW/PAST")


def p01():
    # [1] f(n)=n mod 12, 조건을 만족하는 (a,b) 전수(1..300)
    vals = {(2 * a + b * b) % 12 for a in range(1, 300) for b in range(1, 300)
            if a % 12 > b % 12 and (a % 12) * (b % 12) - 1 == a % 12 + b % 12}
    V.check(1, "1", list(vals)[0] if len(vals) == 1 else -999, V.ans(1, "1"))
    # [2](1) 정수 a,b,c in X 에서 a+b√3+c√7=0
    X = range(-2, 3)
    sums = {a + b + c for a in X for b in X for c in X if abs(a + b * mp.sqrt(3) + c * mp.sqrt(7)) < 1e-12}
    V.check(1, "2-1", list(sums)[0] if len(sums) == 1 else -999, V.ans(1, "2-1"))
    # [2](2) 뺄셈에 닫힌 공집합 아닌 X 의 부분집합
    found = []
    for r in range(1, 6):
        for A in itertools.combinations(X, r):
            if all((a - b) in A for a in A for b in A):
                found.append(tuple(sorted(A)))
    V.check(1, "2-2", sorted(found) == [(0,)], V.ans(1, "2-2") == "{0}")
    # [3](1) 판별식
    a = sp.Symbol("a", real=True)
    V.check(1, "3-1 f 판별식", 16 + 128, 144)
    d = sp.expand(4 * a**2 - 4 * (a - 2))
    V.proof(1, "3-1 g 판별식>0 (1≤a<2)", [v for v in [1 + sp.Rational(i, 100) for i in range(100)] if d.subs(a, v) <= 0])
    # [3](2) L(a): 곱 ≤0 인 집합의 길이 합(수치 적분)
    g_ans = sp.sympify(V.ans(1, "3-2").split("=")[1], locals={"a": a})
    for av in (1, sp.Rational(3, 2), sp.Rational(19, 10)):
        fx = lambda xx: (xx * xx - 4 * xx - 32) * (xx * xx - 2 * av * xx + av - 2)
        # 격자로 길이 측정(구간 [-10,12])
        n = 220000
        step = mp.mpf(22) / n
        L = sum(step for i in range(n) if fx(-10 + i * step) <= 0)
        V.check(1, f"3-2 L(a={av})", float(L), float(g_ans.subs(a, av)), tol=1e-3)


def p02():
    t = sp.Symbol("t", real=True)
    Ff = lambda u: sp.integrate(t**3 + 4 * t + 1 - sp.sin(t**3), (t, 0, u))
    f = lambda tt: tt**3 + 4 * tt + 1 - mp.sin(tt**3)
    d1 = mp.quad(f, [-1, 1])
    V.check(2, "1-1 F(1)-F(-1)", float(d1), float(sp.sympify(V.ans(2, "1-1"))), tol=1e-9)
    F1 = mp.quad(f, [0, 1])
    V.proof(2, "1-2 3≤F(1)≤3.5", [] if 3 <= F1 <= 3.5 else [F1])
    V.proof(2, "1-3 F(0)<2.99<F(1)", [] if 0 < 2.99 < F1 else [F1])
    # [2] G(x)=F(x^2)-1
    X = sp.Symbol("X", real=True)
    Fx = sp.Function("Fx")
    Fp = lambda u: u**3 + 4 * u + 1 - sp.sin(u**3)
    Gp = 2 * X * Fp(X**2)
    Gpp = sp.diff(Gp, X)
    grid = [mp.mpf(i) / 50 for i in range(-200, 201) if i != 0]
    V.proof(2, "2-1 G'>0 (x>0)", [g for g in grid if g > 0 and Gp.subs(X, g).evalf() <= 0])
    V.proof(2, "2-2 G' 부호 x 와 같음", [g for g in grid if (Gp.subs(X, g).evalf() > 0) != (g > 0)])
    G0 = Ff(0) - 1
    V.check(2, "2-2 극솟값 G(0)", G0, V.ans(2, "2-2"))
    V.proof(2, "2-3 G''>0 전구간", [g for g in grid + [mp.mpf(0)] if Gpp.subs(X, g).evalf() <= 0])


if __name__ == "__main__":
    V.run(globals())
