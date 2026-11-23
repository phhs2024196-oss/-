import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("SHU-G26H", "items/SHU/PAST")
P = lambda s: parse_expr(s.replace("^", "**"), local_dict={"x": x})
def kv(s): return dict(p.split("=", 1) for p in s.replace(" ", "").split(";") if "=" in p)

def p10():
    m = sp.symbols("m", real=True)
    # y=3^x → y축 대칭: 3^(-x) → x 방향 m 평행이동: 3^(-(x-m)); (3,9) 대입
    sol = sp.solve(sp.Eq(3 ** (-(3 - m)), 9), m)
    V.check(10, "m", sol[0], P(V.ans(10, "1")))
    V.check(10, "해 개수", len(sol), 1)

def p11():
    v = sp.symbols("v", positive=True)
    c = 3 * 10**8
    sol = sp.solve(sp.Eq(8 * (1 - v**2 / c**2) ** sp.Rational(-1, 2), 10), v)
    V.check(11, "v", sol[0], P(V.ans(11, "1")))
    V.check(11, "해 개수", len(sol), 1)

def p12():
    r, k = sp.symbols("r k", positive=True)
    q = sp.symbols("q", real=True)  # q = r^k
    # S_k=4(q-1)/(r-1)=20, S_2k=4(q^2-1)/(r-1)=30 → 비 = (q+1) = 3/2
    q_ = sp.solve(sp.Eq(q + 1, sp.Rational(30, 20)), q)[0]
    V.check(12, "r^k", q_, sp.Rational(1, 2))
    V.check(12, "a_{k+1}", 4 * q_, P(V.ans(12, "1")))
    # 수치 확인: k=1 → r=1/2
    rr = sp.Rational(1, 2)
    S = lambda n: sum(4 * rr**i for i in range(n))
    rr = sp.Rational(1, 2) ** sp.Rational(1, 3)  # k=3
    S3 = sum(4 * rr**i for i in range(3)); S6 = sum(4 * rr**i for i in range(6))
    V.report(12, abs(float(S6 / S3) - 1.5) < 1e-9, "S_2k/S_k=3/2 (r^k=1/2 일 때 비는 1+r^k)")

def p13():
    a, b = sp.symbols("a b", real=True)
    L = sp.limit((x**3 - 3 * a.subs(a, sp.Rational(1, 3))) / (x - 1), x, 1)
    ans = kv(V.ans(13, "1"))
    V.check(13, "a (분자 x=1 에서 0)", sp.solve(sp.Eq(1 - 3 * a, 0), a)[0], P(ans["a"]))
    V.check(13, "b = 극한", L, P(ans["b"]))
    V.check(13, "인수분해 후 극한", sp.limit(sp.cancel((x**3 - 1) / (x - 1)), x, 1), P(ans["b"]))

def p14():
    a, b = sp.symbols("a b", real=True)
    f = -x * (x - 5)
    ans = kv(V.ans(14, "1"))
    V.check(14, "f", sp.expand(f), sp.expand(P(ans["f"])))
    V.check(14, "극대 x=5/2", sp.solve(sp.diff(f, x), x)[0], sp.Rational(5, 2))
    V.check(14, "극댓값 25/4", f.subs(x, sp.Rational(5, 2)), sp.Rational(25, 4))
    area = sp.integrate(f - a * x, (x, 0, 5 - a))
    sol = [s for s in sp.solve(sp.Eq(area, 36), a) if s.is_real and s < 0]
    V.check(14, "해 개수(a<0)", len(sol), 1)
    V.check(14, "a", sol[0], -1)
    V.check(14, "g(x)", sol[0] * x, P(ans["g"]))
    V.check(14, "g(2)", (sol[0] * x).subs(x, 2), P(ans["g2"]))

def p15():
    a = sp.symbols("a", real=True)
    f = sp.Rational(8, 3) * x**3 + 2 * x**2
    g = x**4 - 4 * x**2 + a
    h = sp.expand(g - f)
    crit = sp.solve(sp.diff(h, x), x)
    V.check(15, "임계점", set(crit), {-1, 0, 3})
    mn = min(h.subs(x, c) for c in crit)
    V.check(15, "h(-1)", h.subs(x, -1), a - sp.Rational(7, 3))
    V.check(15, "h(3)", h.subs(x, 3), a - 45)
    V.check(15, "최소 a", sp.solve(sp.Eq(h.subs(x, 3), 0), a)[0], P(V.ans(15, "1")))
    # 수치: a=45 에서 h>=0, a<45 에서 h(3)<0
    import numpy as np
    xs = np.linspace(-5, 6, 20001)
    hv = lambda av: xs**4 - 8 / 3 * xs**3 - 6 * xs**2 + av
    V.report(15, hv(45).min() > -1e-6, "a=45 에서 h≥0")
    V.report(15, hv(44.9).min() < 0, "a<45 에서 h<0 인 x 존재")

if __name__ == "__main__": V.run(globals())
