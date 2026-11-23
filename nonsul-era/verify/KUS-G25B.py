import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUS-G25B", "items/KUS/PAST")
a, b, k, m, r = sp.symbols("a b k m r", positive=True)
loc = {"a": a, "b": b, "k": k, "m": m, "r": r}


def A(no, sub, key=None):
    return sp.sympify(V.ans(no, sub, key), locals=loc)


def p01():
    # 1-1: |sin(pi m x)|=L 의 근(0<=x<=2)이 등차수열이 되는 L
    s = sp.symbols("s", positive=True)                       # s = asin(L)/pi, 한 주기(1/m) 안의 근 s, 1-s, 1+s, 2-s (m=1 기준)
    sols = sp.solve(sp.Eq((1 - s) - s, (1 + s) - (1 - s)), s)
    V.check(1, "L", sp.sin(sp.pi * sols[0]), A(1, "1-1", "L"))
    ok = True
    for mm in range(1, 7):
        L = sp.sqrt(2) / 2
        roots = sorted({sp.nsimplify((sp.asin(L) / sp.pi + j) / mm) for j in range(0, 2 * mm + 1)}
                       | {sp.nsimplify((j - sp.asin(L) / sp.pi) / mm) for j in range(1, 2 * mm + 2)})
        roots = [q for q in roots if 0 <= q <= 2]
        gaps = {sp.simplify(roots[i + 1] - roots[i]) for i in range(len(roots) - 1)}
        ok &= len(roots) == A(1, "1-1", "n").subs(m, mm) and gaps == {sp.Rational(1, 2) / mm} and gaps == {A(1, "1-1", "d").subs(m, mm)}
    V.report(1, ok, "m=1..6 근의 개수·공차 일치")
    # 1-2
    g = k / sp.sqrt(a + b * x**2)
    e = sp.simplify((x * g**3 + sp.diff(g, x)) * (a + b * x**2) ** sp.Rational(3, 2) / (x * k))
    V.check(1, "1-2 k", sp.solve(e, k)[0], A(1, "1-2", "k"))
    # 1-3
    kk = sp.sqrt(b)
    G = kk / sp.sqrt(a + b * x**2); H = kk / sp.sqrt(a - b * x**2)
    xa = sp.sqrt(a / (2 * b))
    aa = sp.solve(sp.simplify(H.subs(x, xa)**2 - G.subs(x, xa)**2 - b / 3), a)[0]
    V.check(1, "1-3 a", aa, A(1, "1-3", "a"))
    tang = sp.simplify((H.subs(x, sp.Rational(1, 4)) - sp.diff(H, x).subs(x, sp.Rational(1, 4)) / 4).subs(a, aa))
    bs = [v for v in sp.solve(sp.numer(sp.together(tang**2)), b) if v.is_real and v > 0 and v < 16 * aa]
    V.check(1, "1-3 b", bs[0], A(1, "1-3", "b")); V.report(1, len(bs) == 1, f"조건 b<16a 만족 해 {bs}")
    Gs = G.subs({a: aa, b: bs[0]})
    P = (sp.Rational(1, 4), sp.simplify(Gs.subs(x, sp.Rational(1, 4))))
    slope = sp.diff(Gs, x).subs(x, sp.Rational(1, 4))
    q = sp.simplify(sp.solve(P[1] + slope * (x - P[0]), x)[0])
    V.check(1, "1-3 Py", P[1], A(1, "1-3", "Py")); V.check(1, "1-3 q", q, A(1, "1-3", "q"))
    V.check(1, "1-3 넓이", sp.simplify(abs(q * P[1]) / 2), A(1, "1-3", "S"))


def p02():
    P = sp.Rational
    V.check(2, "2-1", P(1, 2) - P(48, 100), V.ans(2, "2-1"))                    # 표의 0.48 사용
    V.report(2, abs(float(sp.erf(sp.Integer(2) / sp.sqrt(2)) / 2) - 0.48) < 0.003, "표 값 P(0<=Z<=2)=0.48 근사 확인")
    m1, s1 = sp.symbols("m1 s1")
    sol = sp.solve([sp.Eq(4.5 - m1, 1.5 * s1), sp.Eq(m1 - 2, 1 * s1)], [m1, s1])
    V.check(2, "2-2 m1", sol[m1], sp.sympify(V.ans(2, "2-2", "m1"))); V.check(2, "2-2 s1", sol[s1], sp.sympify(V.ans(2, "2-2", "s1")))
    pn = 0.6 * 0.96 + 0.1 * 0.04
    V.check(2, "2-3", sum(pn * (1 - pn) ** i for i in range(4)), V.ans(2, "2-3"), tol=1e-3)
    V.report(2, abs(pn - 0.58) < 1e-12, f"하루 전환확률 {pn}")
    n = next(n for n in range(1, 1000) if 2 * 2 * 1 / n**0.5 <= 0.4)
    V.check(2, "2-4", n, V.ans(2, "2-4"))


def p03():
    r_ = sp.Integer(1)
    s = sp.sqrt(2) * r_; e = s / 2; r2 = e; h = s
    H = sp.simplify(h * r_ / (r_ - r2))
    V.check(3, "원뿔 높이", H, 2 * (1 + sp.sqrt(2)))
    th = sp.symbols("th")
    d2 = h**2 + r_**2 + r2**2 - 2 * r_ * r2 * sp.cos(th)
    p2, q2 = d2.subs(th, 0), d2.subs(th, sp.pi)
    V.check(3, "3-1", sp.simplify(p2 + q2), V.ans(3, "3-1"))
    c = r_ / H
    y, z = sp.symbols("y z", real=True)
    # 3-2: 평면 x=e, y^2+e^2=(c(H-z))^2 -> (z-H)^2/A^2 - y^2/B^2 = 1
    A2 = sp.simplify(e**2 / c**2); B2 = e**2
    chk = sp.simplify((z - H) ** 2 / A2 - y**2 / B2 - 1 - ((c * (H - z)) ** 2 - e**2 - y**2) / e**2)
    d2h = 4 * (A2 + B2)
    V.report(3, chk == 0, "쌍곡선 표준형 일치")
    V.check(3, "3-2", sp.simplify(d2h), V.ans(3, "3-2"))
    # 3-3: 평면 z=y+e (AB: y=-e,z=0 / GH: y=e,z=2e 을 포함)
    u = sp.symbols("u", real=True)
    cone = x**2 + (u / sp.sqrt(2)) ** 2 - (c * (H - (e + u / sp.sqrt(2)))) ** 2
    poly = sp.Poly(sp.expand(cone), x, u)
    al, be, ga, de = poly.coeff_monomial(x**2), poly.coeff_monomial(u**2), poly.coeff_monomial(u), poly.coeff_monomial(1)
    T = -de + ga**2 / (4 * be)
    a2 = sorted([sp.simplify(T / al), sp.simplify(T / be)], key=lambda v: float(v), reverse=True)
    V.check(3, "3-3", sp.simplify(a2[0] / a2[1]), V.ans(3, "3-3"))
    V.report(3, abs(float(a2[0]) - float((sp.Rational(2, 7) * (3 + sp.sqrt(2))) ** 2)) < 1e-9, "a^2 = (2/7(3+√2) r)^2 일치")


if __name__ == "__main__":
    V.run(globals())
