"""광운대 2025학년도 모의논술 자연계열 [문제 1]·[문제 2] 검산 (기대값은 YAML answer 에서만 읽는다)."""
import sys, pathlib, math, itertools; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath
V = Verifier("KW-M25", "items/KW/PAST")
L = {"x": x, "n": n, "t": t}
def E(s): return sp.sympify(s, locals=L)

def p01():
    perms = sorted(set(itertools.permutations("KWANGWOON")))
    V.check(1, "1-1", len(perms), V.ans(1, "1-1"))
    V.check(1, "1-2", sum(1 for w in perms if abs(w.index("K") - w.index("G")) == 4), V.ans(1, "1-2"))
    V.check(1, "1-3", perms.index(tuple("KWANGWOON")) + 1, V.ans(1, "1-3"))
    # [2](1)
    tot = 0
    for m in range(1, 6):
        r = sp.solve(x**2 - m*x - (m**3 + 3*m**2 + 2*m), x)
        tot += sum(v**2 for v in r)
    V.check(1, "2-1", sp.simplify(tot), V.ans(1, "2-1"))
    # [2](2): 근을 직접 구해 고정밀 합
    mpmath.mp.dps = 60
    s = mpmath.mpf(0)
    for m in range(1, 2025):
        c = m**3 + 3*m**2 + 2*m
        d = mpmath.sqrt(m*m + 4*c)
        a, b = (m + d)/2, (m - d)/2
        s += 1/a + 1/b
    V.check(1, "2-2", sp.Float(str(s), 50), sp.N(E(V.ans(1, "2-2")), 50), 1e-40)
    # [2](3)
    y = x**2 - n*x - n**3 - 3*n**2 - 2*n
    xs = sp.solve(sp.diff(y, x), x)[0]
    V.check(1, "2-3", sp.simplify(y.subs(x, xs)), E(V.ans(1, "2-3")))

def p02():
    # [1] 귀납 단계의 극한을 n=1..12 에서 확인
    bad = [m for m in range(1, 13) if sp.limit(sp.sin(m*x)/sp.sin(x), x, 0) != m]
    V.proof(2, "1 lim sin(nx)/sin x = n (n=1..12)", bad)
    # [2]
    a = sp.symbols("a", real=True)
    ar = sp.solve(sp.Eq(a**3 - 0, sp.sin(0) + 1), a)
    yy = sp.Function("y")
    F = x**3 - y**3 - sp.sin(x*y) - 1
    b = sp.idiff(F, y, x).subs({x: 1, y: 0})
    V.check(2, "2", ar[0] + b if len(ar) == 1 else -1, V.ans(2, "2"))
    # [3]
    N = 2024
    xv = mpmath.mpf("0.3")
    mpmath.mp.dps = 40
    ssum = sum(mpmath.binomial(N, j)*(mpmath.mpf(1)/4)**j*(mpmath.mpf(3)/4)**(N - j)*mpmath.e**(j*xv/2) for j in range(N + 1))
    closed = (mpmath.e**(xv/2)/4 + mpmath.mpf(3)/4)**N
    V.report(2, abs(ssum/closed - 1) < mpmath.mpf(10)**-25, "3 g(x)=(e^(x/2)/4+3/4)^2024 (x=0.3 에서 합과 일치)")
    g = (sp.exp(x/2)/4 + sp.Rational(3, 4))**N
    ratio = sp.simplify(sp.diff(g, x)/g)
    sol = sp.solve(sp.Eq(ratio, 253), x)
    av = sol[0]
    tang = sp.diff(g, x).subs(x, av)*(x - av) + g.subs(x, av)
    V.check(2, "3 a", av, E(V.ans(2, "3", "a")))
    V.check(2, "3 접선", sp.simplify(tang), E(V.ans(2, "3", "y")))
    # [4](1)
    h, s_ = sp.symbols("h s", real=True)
    sp_ = sp.symbols("sp", positive=True)
    right = sp.limit(sp.Abs(h*sp.sin(1/h)), h, 0, "+")   # |h sin(1/h)| <= h -> 0
    left = sp.limit(-sp.sqrt(sp_**3)/sp_, sp_, 0, "+")   # h=-s: sqrt(-h^3)/h = -sqrt(s)
    V.check(2, "4-1 f'(0)", 0 if (right, left) == (0, 0) else 1, V.ans(2, "4-1"))
    # [4](2): f' 공식 점검 + 0+ 에서 극한 없음(수열 두 개)
    fp = sp.diff(x**2*sp.sin(1/x), x)
    V.report(2, sp.simplify(fp - (2*x*sp.sin(1/x) - sp.cos(1/x))) == 0, "4-2 x>0 에서 f'(x)=2x sin(1/x)-cos(1/x)")
    xp = sp.symbols("xp", positive=True)
    V.report(2, sp.simplify(sp.diff(sp.sqrt(xp**3), xp).subs(xp, -x) * (-1) - (-sp.Rational(3, 2)*sp.sqrt(-x))) == 0, "4-2 x<0 에서 f'(x)=-3/2 sqrt(-x)")
    fpf = sp.lambdify(x, 2*x*sp.sin(1/x) - sp.cos(1/x), "mpmath")
    k_ = 10**6
    v1 = fpf(mpmath.mpf(1)/(2*mpmath.pi*k_)); v2 = fpf(mpmath.mpf(1)/((2*k_ + 1)*mpmath.pi))
    V.report(2, abs(v1 - v2) > 1.9 and V.ans(2, "4-2") == "불연속", f"4-2 x->0+ 에서 f' 가 -1, +1 근방을 오감({float(v1):.4f}, {float(v2):.4f}) -> 불연속")
    # [4](3)
    val = mpmath.quad(lambda xx: xx**2*mpmath.sin(1/xx)/xx**5, [2/mpmath.pi, 6/mpmath.pi])
    V.check(2, "4-3", sp.Float(str(val), 30), sp.N(E(V.ans(2, "4-3")), 30), 1e-20)

if __name__ == "__main__": V.run(globals())
