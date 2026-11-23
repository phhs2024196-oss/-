"""KW-G18A 검산: 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G18A", "items/KW/PAST")
z, s_ = sp.symbols("z s", real=True)


def vec(*a):
    return sp.Matrix(a)


def p01():
    O = vec(0, 0, 0)
    # [1](1) 직선 l: (x+1)/2 = y = 2-z  → 점 P0(-1,0,2), 방향 (2,1,-1)
    P0, d = vec(-1, 0, 2), vec(2, 1, -1)
    V.check(1, "1-1 거리", sp.nsimplify(sp.sqrt(P0.cross(d).dot(P0.cross(d)) / d.dot(d))), V.ans(1, "1-1"))
    # [1](2) 교점 A, 평면 β: OA 에 수직이고 A 를 지남
    P = P0 + s_ * d
    sol = sp.solve(P[0] - 3 * P[1] + 2 * P[2], s_)
    A = P.subs(s_, sol[0])
    V.check(1, "1-2 A", list(A), [1, 1, 1])
    plane = sp.expand(A[0] * (x - A[0]) + A[1] * (y - A[1]) + A[2] * (z - A[2]))
    lhs, rhs = V.ans(1, "1-2").split("=")
    given = sp.expand(sp.sympify(lhs, locals={"x": x, "y": y, "z": z}) - sp.sympify(rhs))
    V.check(1, "1-2 평면(상수배 정규화)", sp.simplify(plane / plane.coeff(x) - given / given.coeff(x)), 0)
    # [1](3)
    n1, n2 = vec(*A), vec(0, 0, 1)
    c = sp.nsimplify(abs(n1.dot(n2)) / (sp.sqrt(n1.dot(n1)) * sp.sqrt(n2.dot(n2))))
    V.check(1, "1-3 cos", c, V.ans(1, "1-3"))
    # [2](1) 구
    B, C = vec(5, 3, sp.sqrt(2)), vec(7, 5, -sp.sqrt(2))
    X = vec(x, y, z)
    eq = sp.expand((X - B).dot(X - C))
    cx, cy, cz = [sp.Rational(-eq.coeff(v, 1), 2) for v in (x, y, z)]
    r2 = cx**2 + cy**2 + cz**2 - eq.subs({x: 0, y: 0, z: 0})
    m = re.search(r"D=\((.+?),(.+?),(.+?)\)", V.ans(1, "2-1"))
    V.check(1, "2-1 중심", [cx, cy, cz], [sp.sympify(g) for g in m.groups()])
    V.check(1, "2-1 반지름", sp.sqrt(r2), V.ans(1, "2-1", "r"))
    # [2](2) 넓이비
    D = vec(cx, cy, cz)
    s1 = sp.sqrt((O - B).cross(D - B).dot((O - B).cross(D - B)))
    s2 = sp.sqrt((O - D).cross(C - D).dot((O - D).cross(C - D)))
    V.check(1, "2-2 넓이비", sp.simplify(s1 / s2), V.ans(1, "2-2"))
    # [2](3) 원 위에서 E 까지 거리 최솟값(수치 전수)
    E = vec(2, 1, 3 * sp.sqrt(3))
    best = min(mp.sqrt((cx + 2 * mp.cos(q) - 2) ** 2 + (cy + 2 * mp.sin(q) - 1) ** 2 + (0 - 3 * mp.sqrt(3)) ** 2)
               for q in [mp.pi * 2 * i / 20000 for i in range(20000)])
    V.check(1, "2-3 최솟값", float(best), float(sp.sympify(V.ans(1, "2-3"))), tol=1e-6)


def p02():
    t_ = sp.Symbol("t", real=True)
    X = sp.Symbol("X", real=True)
    f = lambda u: u - 2 + 2 * sp.integrate(1 - sp.cos(t_) + t_ * sp.sin(t_), (t_, 0, u))
    V.check(2, "1-1 f(0)", f(0), V.ans(2, "1-1"))
    f1 = sp.simplify(f(1))
    V.check(2, "1-2 f(1)=1-2cos1", f1 - (1 - 2 * sp.cos(1)), 0)
    V.proof(2, "1-2 1-√2<f(1)<0", [] if 1 - sp.sqrt(2) < f1.evalf() < 0 else [f1])
    fx = X - 2 + 2 * sp.integrate(1 - sp.cos(t_) + t_ * sp.sin(t_), (t_, 0, X))
    root = sp.nsolve(fx.subs(X, X) + 1, X, 0.5)
    V.proof(2, "1-3 f(0)<-1<f(1), 근 in (0,1)", [] if (f(0) < -1 < f1.evalf() and 0 < root < 1) else [root])
    d1, d2 = sp.diff(fx, X), sp.diff(fx, X, 2)
    grid = [mp.pi / 2 * i / 2000 for i in range(1, 2000)]
    V.proof(2, "1-4 f'>0", [g for g in grid if d1.subs(X, g).evalf() <= 0])
    V.proof(2, "1-5 f''>0", [g for g in grid if d2.subs(X, g).evalf() <= 0])
    # [2]
    nn = sp.Symbol("nn", positive=True)
    a1 = sp.integrate(t_ / 2 * sp.exp(-t_**2), (t_, 0, 1))
    V.check(2, "2-1 a1", a1, V.ans(2, "2-1"))
    tot = sum(mp.quad(lambda u, k=k: u / (k + 1) * mp.e ** (-k * u * u), [0, 1 / mp.sqrt(k)]) for k in range(1, 100))
    V.check(2, "2-2 합", float(tot), float(sp.sympify(V.ans(2, "2-2")).evalf(30)), tol=1e-9)


if __name__ == "__main__":
    V.run(globals())
