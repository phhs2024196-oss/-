import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("SHU-G26N", "items/SHU/PAST")
P = lambda s: parse_expr(s.replace("^", "**"), local_dict={"x": x})
def kv(s): return dict(p.split("=", 1) for p in s.replace(" ", "").split(";") if "=" in p)

def p07():
    f = 3 ** (x - 1) + 2
    mn, mx = extrema(f, x, -1, 3)
    V.check(7, "M", mx, P(kv(V.ans(7, "1"))["M"]))
    V.check(7, "m", mn, P(kv(V.ans(7, "1"))["m"]))
    V.report(7, sp.diff(f, x).subs(x, 0) > 0 and sp.diff(f, x).subs(x, 2) > 0, "증가함수")

def p08():
    d = sp.symbols("d", positive=True)
    # log2=0.3 → log5=1-log2=0.7
    l2 = sp.Rational(3, 10); l5 = 1 - l2
    sol = sp.solve(sp.Eq(-l5, -sp.Rational(245, 1000) / sp.Rational(7, 10) * d), d)
    V.check(8, "d", sol[0], P(V.ans(8, "1")))
    import math
    V.report(8, abs(math.log10(5) - 0.7) < 0.01, "log5≈0.7 (근사값 타당)")

def p09():
    a, b, c = 12, 10, 8
    V.report(9, 3 * c == 2 * a, "(가) 3c=2a")
    V.report(9, 5 * (a - c) == 2 * b, "(나) 5(a-c)=2b")
    cosA = sp.Rational(b**2 + c**2 - a**2, 2 * b * c)
    V.check(9, "cosA", cosA, sp.Rational(1, 8))
    area = sp.Rational(1, 2) * b * c * sp.sqrt(1 - cosA**2)
    V.check(9, "넓이", sp.simplify(area), P(V.ans(9, "1")))
    # 삼각형 성립 + 헤론
    s = (a + b + c) / 2
    V.check(9, "헤론", sp.sqrt(s * (s - a) * (s - b) * (s - c)), P(V.ans(9, "1")))
    # 사인비 확인: sinA:sinB:sinC = a:b:c
    V.report(9, a + c > b and a + b > c and b + c > a, "삼각형 성립")

def p10():
    a, b = sp.symbols("a b", real=True)
    ans = kv(V.ans(10, "1"))
    av = sp.solve(sp.Eq(2**2 + (a - 1) * 2 - 3, 0), a)[0]
    V.check(10, "a", av, P(ans["a"]))
    f = (x**2 + (av - 1) * x - 3) / (x - 2)
    V.check(10, "b=극한", sp.limit(f, x, 2), P(ans["b"]))

def p11():
    f = 3 * x**4 + 2 * x**3 - x
    g = -x**3 + 4 * x**2 + 2 * x + 1
    h = sp.expand(f - g)
    V.check(11, "h", h, 3 * x**4 + 3 * x**3 - 4 * x**2 - 3 * x - 1)
    V.check(11, "h(-2)", h.subs(x, -2), 13)
    V.check(11, "h(-1)", h.subs(x, -1), -2)
    r = sp.nroots(h)
    V.report(11, any(abs(sp.im(z)) < 1e-9 and -2 < sp.re(z) < -1 for z in r), "(-2,-1) 에 실근 존재(수치)")

def p12():
    def an(n):
        A = [3**k for k in range(1, n + 3)]
        return len({A[i] * A[j] for i in range(len(A)) for j in range(i + 1, len(A))})
    V.report(12, all(an(n) == 2 * n + 1 for n in range(1, 12)), "a_n=2n+1 전수(n=1..11)")
    V.check(12, "합", sum(an(n) for n in range(1, 9)), P(V.ans(12, "1")))

def p13():
    a, b = sp.symbols("a b", real=True)
    f = x**2 + a * x + b
    g = (2 * x + 3) * f
    c1 = sp.Eq(5 * sp.diff(f, x).subs(x, 1), g.subs(x, 1) + 35)   # (가): g(1)=5f(1) 이므로 5f'(1)
    c2 = sp.Eq(sp.diff(g, x).subs(x, 1), f.subs(x, 1) + 11)
    sol = sp.solve([c1, c2], [a, b], dict=True)
    V.check(13, "해 개수", len(sol), 1)
    V.check(13, "g(1)=5f(1)", g.subs(x, 1), 5 * f.subs(x, 1))
    fs = f.subs(sol[0]); gs = (2 * x + 3) * fs
    V.check(13, "f(6)-g'(2)", fs.subs(x, 6) - sp.diff(gs, x).subs(x, 2), P(V.ans(13, "1")))
    # 극한 조건 직접 검증
    t = sp.symbols("t")
    V.check(13, "(가) 극한", sp.limit((5 * fs - gs.subs(x, 1)) / (x - 1), x, 1), gs.subs(x, 1) + 35)
    V.check(13, "(나) 극한", sp.limit((gs - 5 * fs.subs(x, 1)) / (x - 1), x, 1), fs.subs(x, 1) + 11)

def p14():
    a = sp.symbols("a", positive=True)
    f = 2 * x**2 + 6 * a * x; g = -x**2 + 9 * a**2
    xs = sp.solve(sp.Eq(f, g), x)
    V.check(14, "교점 x", set(xs), {a, -3 * a})
    area = sp.integrate(g - f, (x, -3 * a, a))
    V.check(14, "넓이", sp.expand(area), 32 * a**3)
    av = sp.solve(sp.Eq(area, sp.Rational(32, 27)), a)[0]
    ans = kv(V.ans(14, "1"))
    V.check(14, "a", av, P(ans["a"]))
    p1 = (a, f.subs(x, a)); p2 = (-3 * a, f.subs(x, -3 * a))
    h = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (x - p1[0])
    h = sp.simplify(h.subs(a, av))
    V.check(14, "h(x)", sp.expand(h), sp.expand(P(ans["h"])))
    V.check(14, "h(2)", h.subs(x, 2), P(ans["h2"]))

def p15():
    a, b, k = sp.symbols("a b k", real=True)
    f = 12 * x**3 + 12 * a * x**2 + 12 * b * x
    sol = sp.solve([sp.Eq(f.subs(x, 1), -28 + 8), sp.Eq(sp.diff(f, x).subs(x, 1), -28)], [a, b], dict=True)[0]
    fs = sp.expand(f.subs(sol))
    V.check(15, "f", fs, 12 * x**3 - 32 * x**2)
    h = sp.expand(x**4 - 4 * x**2 + k - fs)
    V.check(15, "h", h, x**4 - 12 * x**3 + 28 * x**2 + k)
    crit = sp.solve(sp.diff(h, x), x)
    V.check(15, "임계점", set(crit), {0, 2, 7})
    kmin = max(sp.solve(sp.Eq(h.subs(x, c), 0), k)[0] for c in crit)
    V.check(15, "k 최솟값", kmin, P(V.ans(15, "1")))

if __name__ == "__main__": V.run(globals())
