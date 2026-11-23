"""KW-G19A 검산(2019학년도 오전 문제 1·2): 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G19A", "items/KW/PAST")
mp.mp.dps = 30
a_, p_, q_, b_, c_ = sp.symbols("a p q b c", real=True)
LOC = {"a": a_, "x": x, "e": sp.E}


def p01():
    # [1](1) 직선 y=2ax+b: 넓이 4/3 → 판별식 조건, 여러 a 에서 수치 적분으로 넓이 확인
    line = sp.sympify(V.ans(1, "1-1").split("=", 1)[1], locals=LOC)
    for av in (0, 1, -2.5, 3.7):
        L = sp.lambdify(x, line.subs(a_, av))
        bb = float(line.subs({a_: av, x: 0}))
        disc = (4 + 2 * av) ** 2 + 4 * bb
        x1, x2 = ((4 + 2 * av) - math.sqrt(disc)) / 2, ((4 + 2 * av) + math.sqrt(disc)) / 2
        area = mp.quad(lambda t: L(t) - (t * t - 4 * t), [x1, x2])
        V.check(1, f"1-1 넓이 a={av}", sp.nsimplify(float(area), rational=True, tolerance=1e-9), sp.Rational(4, 3), 1e-8)
    # [1](2) 중점이 y=x^2-4x+1 위
    cur = sp.sympify(V.ans(1, "1-2").split("=", 1)[1], locals=LOC)
    for av in (-3, -1.5, 0, 2, 5):
        bb = float(line.subs({a_: av, x: 0}))
        disc = (4 + 2 * av) ** 2 + 4 * bb
        x1, x2 = ((4 + 2 * av) - math.sqrt(disc)) / 2, ((4 + 2 * av) + math.sqrt(disc)) / 2
        x0 = (x1 + x2) / 2
        y0 = (2 * av * x1 + bb + 2 * av * x2 + bb) / 2
        V.check(1, f"1-2 중점 a={av}", y0, float(cur.subs(x, x0)), 1e-9)
    # [2](1)
    A, B = sp.Matrix([1, 5]), sp.Matrix([7, 2])
    V.check(1, "2-1 |AB|", sp.sqrt(((B - A).T * (B - A))[0]), sp.sympify(V.ans(1, "2-1")))
    # [2](2) P=((1-s)A+sB)/3 의 양 끝점 사이 거리
    s = sp.symbols("s", real=True)
    P = sp.symbols("Px Py", real=True)
    sol = sp.solve([(1 - s) * (A[i] - P[i]) + s * (B[i] - P[i]) + 2 * (0 - P[i]) for i in range(2)], P, dict=True)[0]
    P0 = sp.Matrix([sol[P[0]].subs(s, 0), sol[P[1]].subs(s, 0)]); P1 = sp.Matrix([sol[P[0]].subs(s, 1), sol[P[1]].subs(s, 1)])
    V.check(1, "2-2 길이", sp.sqrt(((P1 - P0).T * (P1 - P0))[0]), sp.sympify(V.ans(1, "2-2")))
    # [3] 좌표로 사면체: C(-1,0,0), D(1,0,0), M 원점, A(0,1,0), B(0,2cosφ,2sinφ), AB^2=5-4cosφ
    Vf = sp.sympify(V.ans(1, "3-1"), locals=LOC)
    for a2 in (1.5, 2.0, 3.0, 5.0, 7.0, 8.5):
        cphi = (5 - a2) / 4
        sphi = math.sqrt(1 - cphi ** 2)
        Av, Bv, Cv, Dv = (0, 1, 0), (0, 2 * cphi, 2 * sphi), (-1, 0, 0), (1, 0, 0)
        m = sp.Matrix([[Av[i] - Cv[i], Bv[i] - Cv[i], Dv[i] - Cv[i]] for i in range(3)])
        vol = abs(float(m.det())) / 6
        V.check(1, f"3-1 V a^2={a2}", vol, float(Vf.subs(a_, math.sqrt(a2))), 1e-9)
    # [3](2) 범위·최댓값
    rng = V.ans(1, "3-2").split(";")[0]
    ok = True
    for a2 in [i / 100 for i in range(0, 1100)]:
        real = (-(a2 ** 2 - 10 * a2 + 9)) > 1e-12
        pred = eval(rng.replace("a^2", "a2"), {"a2": a2})
        ok &= (real == pred) or abs(a2 - 1) < 1e-9 or abs(a2 - 9) < 1e-9
    V.check(1, "3-2 범위 일치", ok, True)
    vmax = max(math.sqrt(max(0, -(t ** 4 - 10 * t ** 2 + 9))) / 6 for t in [math.sqrt(1 + i * 8 / 20000) for i in range(20001)])
    V.check(1, "3-2 최댓값", vmax, float(sp.sympify(V.ans(1, "3-2", "Vmax"))), 1e-6)


def p02():
    t = sp.symbols("t", positive=True)
    # [1] g(x) 최솟점
    g = sp.integrate(sp.expand((t - x) * (t - x / t) * sp.log(t)), (t, 1, sp.E))
    amin = sp.solve(sp.diff(g, x), x)[0]
    V.check(2, "1 a", sp.simplify(amin), sp.sympify(V.ans(2, "1"), locals={"exp": sp.exp}))
    V.check(2, "1 g''>0(최소)", int(bool(sp.simplify(sp.diff(g, x, 2)) > 0)), 1)
    # [2](1) 조건 (i): 모든 p,q → 계수 0, (ii) 로 a
    f = a_ * x**2 + b_ * x + c_
    I = sp.integrate(f * (p_ * x + q_), (x, -1, 1))
    cp, cq = sp.expand(I).coeff(p_), sp.expand(I).coeff(q_)
    s1 = sp.solve([cp, cq], [b_, c_], dict=True)[0]
    f1 = f.subs(s1)
    tt = sp.symbols("tt", real=True)
    cond = sp.simplify(2 * f1 - sp.integrate(f1 - 3 + f1.subs(x, tt), (tt, 0, 2)))
    av = sp.solve(cond, a_)[0]
    f2 = sp.expand(f1.subs(a_, av))
    V.check(2, "2-1 f", f2, sp.sympify(V.ans(2, "2-1"), locals=LOC))
    # [2](2)
    h = sp.symbols("h")
    fy = sp.sympify(V.ans(2, "2-1"), locals=LOC)
    V.check(2, "2-2 극한", sp.limit((fy.subs(x, sp.exp(h)) - 2) / h, h, 0), sp.sympify(V.ans(2, "2-2")))
    # [3]
    F = lambda u: (u * mp.e ** u + 1) / (mp.e ** u + 1)
    roots = set()
    grid = [i / 50 for i in range(-500, 501)]
    for u0, u1 in zip(grid, grid[1:]):
        v0, v1 = F(F(u0)) - 1, F(F(u1)) - 1
        if v0 == 0 or v0 * v1 < 0:
            roots.add(round(float(mp.findroot(lambda z: F(F(z)) - 1, (u0 + u1) / 2)), 8))
    V.check(2, "3-1 α", sorted(roots), [float(sp.sympify(V.ans(2, "3-1")))])
    # [3](2) 극값 p: e^p+p=0, 좌우 부호
    pp = mp.findroot(lambda z: mp.e ** z + z, -0.5)
    fp = lambda u: mp.diff(F, u)
    kind = "극소" if fp(pp - 0.05) < 0 < fp(pp + 0.05) else "극대"
    V.check(2, "3-2 판정", kind == V.ans(2, "3-2"), True)
    # [3](3) a=e, b=sup |e f(x)-e|/|x-1| = e (도달하지 않는 상한)
    ae = float(sp.sympify(V.ans(2, "3-3", "a"), locals={"e": sp.E}))
    ratio = lambda u: abs(mp.e * F(u) - ae) / abs(u - 1)
    sup = max(ratio(mp.mpf(u) / 10) for u in range(-300, 600) if u != 10)
    V.check(2, "3-3 a=e", ae, math.e)
    be = float(sp.sympify(V.ans(2, "3-3", "b"), locals={"e": sp.E}))
    V.check(2, "3-3 상한<=b", sup <= be + 1e-9, True)
    V.check(2, "3-3 b 미만은 부적합", ratio(mp.mpf(60)) > be - 1e-6, True)
    V.check(2, "3-3 lim", sp.limit(sp.E ** (x + 1) / (sp.E ** x + 1), x, sp.oo), be)


if __name__ == "__main__": V.run(globals())
