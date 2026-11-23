"""광운대 2024학년도 자연계열 1교시 [문제 1]·[문제 2] 검산 (기대값은 YAML answer 에서만 읽는다)."""
import sys, pathlib, math, itertools; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-G24A", "items/KW/PAST")
L = {"x": x, "n": n, "t": t}
def E(s): return sp.sympify(s, locals=L)

def p01():
    # [1](1)
    V.check(1, "1-1", limit((2*n+1)*sp.log((2*n+1)/(2*n-1)), n, sp.oo), V.ans(1, "1-1"))
    # [1](2) 증명: 공식 f'(x) 표기가 맞는지 + b_n 증가를 수치 확인
    f = -sp.log(1 + 2/(2*x - 1))/(2*x + 1)
    off = 2/(2*x + 1)**2*sp.log(1 + 2/(2*x - 1)) + 1/(2*x + 1)*(4/(2*x - 1)**2)/(1 + 2/(2*x - 1))
    V.report(1, sp.simplify(sp.diff(f, x) - off) == 0, "1-2 공식 f'(x) 표기 = 직접 미분")
    b = lambda m: -sp.log(sp.Rational(2*m + 1, 2*m - 1))/(2*m + 1)
    V.proof(1, "1-2 b_n<b_{n+1} (n=1..300)", search(lambda m: sp.N(b(m + 1) - b(m), 40) > 0, range(1, 301)))
    # [2](1)
    a = sp.symbols("a", real=True)
    fa = a*x*(x - 2*sp.pi)
    sol = sp.solve(sp.Eq(fa.subs(x, sp.pi), 1), a)
    V.report(1, len(sol) == 1 and sol[0] < 0, f"2-1 조건에서 a={sol[0]} (유일, 음수)")
    fx = sol[0]*x*(x - 2*sp.pi)
    h = sp.lambdify(x, fx + sp.Abs(sp.cos(x)) + x/sp.pi, "math")
    mn = min(h(2*math.pi*i/20000) for i in range(20001))
    V.report(1, mn > 0, f"2-1 좌변-우변 최솟값 {mn:.6f} > 0 (실근 없음)")
    # [2](2): 합성함수 최대·최소 존재 <=> f(x)=-2p 의 해가 [0,2pi]에 없음
    ok = True
    for pv in [sp.Rational(v) for v in ["-5", "-2", "-1", "-3/4", "-0.6", "-0.51", "-0.49", "-0.4", "-0.25", "-0.1", "-0.01", "0.01", "0.5", "1", "3"]]:
        sing = sp.solveset(sp.Eq(fx, -2*pv), x, sp.Interval(0, 2*sp.pi))
        exist = (sing == sp.EmptySet)
        claim = eval(V.ans(1, "2-2").replace("p", "pv").replace("pvv", "pv"), {"pv": float(pv)})
        ok &= (exist == bool(claim))
    V.report(1, ok, "2-2 p 범위 표본 일치")
    # [3](1),(2)
    B = sorted((int("".join(map(str, d))) for d in itertools.permutations(range(7)) if d[0] != 0), reverse=True)
    V.check(1, "3-1", sum(1 for v in B if v % 4 == 0), V.ans(1, "3-1"))
    V.check(1, "3-2", B[999], V.ans(1, "3-2"))

def p02():
    # [1] 일반 다항식(6차)에 대입해 계수 결정
    c = sp.symbols("c0:7")
    P = sum(c[i]*x**i for i in range(7))
    eqs = sp.Poly(sp.expand(P.subs(x, 3*x) - P - x**3), x).all_coeffs() + [P.subs(x, 0) - 1]
    s = sp.solve(eqs, c, dict=True)[0]
    V.check(2, "1", P.subs(s), E(V.ans(2, "1")))
    # [2] 유리근 없음
    V.report(2, sp.Poly(8*x**3 - 6*x + 1, x).ground_roots() == {}, "2 유리근 없음(ground_roots)")
    # [3]
    V.check(2, "3-1", integral((1 - x**8)/(1 + x**2), x, 0, 1), V.ans(2, "3-1"))
    I = sp.Rational(76, 105)
    d = abs(I - integral(1/(1 + x**2), x, 0, 1))
    V.report(2, sp.N(d - sp.Rational(1, 9)) <= 0, f"3-2 |I-pi/4|={float(d):.5f} <= 1/9")
    # [4]
    exp4 = V.ans(2, "4-1")
    V.report(2, all(eval(exp4, {"n": m}) for m in range(3, 200)), "4-1 n>=3 에서 n^(n+1) > (n+1)^n (n=3..199)")
    g = sp.log(x + 1)/(x + 1)**2
    V.check(2, "4-2", integral(g, x, 0, sp.exp(2) - 1), E(V.ans(2, "4-2")))

if __name__ == "__main__": V.run(globals())
