import sys, pathlib, math, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SWU-M26", "items/SWU/PAST")
a, b, c, d, m = sp.symbols("a b c d m", positive=True)


def E(v):
    """YAML 문자열 -> 양수 기호로 해석(대학 답안의 a,b,d 는 양의 실수)"""
    return sp.sympify(v, locals={"a": a, "b": b, "d": d, "x": x})


def p01():
    f = -a * x**3 + b * x
    tang = sp.diff(f, x).subs(x, 0) * x
    V.check(1, "1-1 접선", tang, E(V.ans(1, "1")))
    xa = [r for r in sp.solve(f, x) if r.is_positive][0]
    ya = sp.simplify(tang.subs(x, xa))
    V.check(1, "1-2 Ax", xa, E(V.ans(1, "2", "Ax")))
    V.check(1, "1-2 Bx", xa, E(V.ans(1, "2", "Bx")))
    V.check(1, "1-2 By", ya, E(V.ans(1, "2", "By")))
    S1 = sp.simplify(sp.Rational(1, 2) * xa * ya)
    V.check(1, "1-2 S1", S1, E(V.ans(1, "2", "S1")))
    S2 = sp.simplify(sp.integrate(f, (x, 0, xa)))
    V.check(1, "1-3 S2", S2, E(V.ans(1, "3", "S2")))
    V.check(1, "1-3 S1/S2", sp.simplify(S1 / S2), E(V.ans(1, "3", "ratio")))


def p02():
    A = sp.symbols("A", real=True)
    V.check(2, "2-1 접선", (2 * a * x - (a**2 + 1)).subs(a, A), E(V.ans(2, "1")).subs(a, A))
    dist = (A**2 + 1) / sp.sqrt(4 * A**2 + 1)
    # 부채꼴 넓이 4pi/3, r=2 -> theta=2pi/3 -> d=2cos(pi/3)=1 (d<2 이므로 두 점에서 만남)
    theta = 2 * sp.pi / 3
    assert same(sp.Rational(1, 2) * 4 * theta, 4 * sp.pi / 3)
    sols = [s for s in sp.solve(sp.Eq(dist, 2 * sp.cos(theta / 2)), A)]
    V.check(2, "2-2 a", sp.FiniteSet(*sols), V.ans(2, "2"))
    # 부채꼴 넓이를 a로 직접 계산(수치): 모든 해에서 넓이가 4pi/3
    for s in sols:
        th = 2 * sp.acos(dist.subs(A, s) / 2)
        V.check(2, f"2-2 검 a={s}", sp.Rational(1, 2) * 4 * th, 4 * sp.pi / 3)
    # 문제의 단서: 중심각 < pi 이면 d>0, 모든 해에서 d<2 (서로 다른 두 점)
    V.report(2, all(dist.subs(A, s) < 2 for s in sols), "서로 다른 두 점 조건 d<2")
    t0 = 2 * x * 0 - 1                      # a=0 의 접선 y=-1
    t1 = 2 * sp.sqrt(2) * x - 3             # a=√2 의 접선
    xi = sp.solve(sp.Eq(t0, t1), x)[0]
    area = sp.integrate((x**2 - 1) - t1, (x, xi, sp.sqrt(2))) + sp.integrate((x**2 - 1) - t0, (x, 0, xi))
    V.check(2, "2-3 넓이(경계 분할)", sp.simplify(area), V.ans(2, "3"))
    off = sp.integrate((x**2 - 1) - t1, (x, 0, sp.sqrt(2))) - sp.Rational(1, 2) * xi * 2
    V.check(2, "2-3 넓이(대학 풀이 (A)-(B))", sp.simplify(off), V.ans(2, "3"))


def p03():
    # (1) C=pi/6, BC=a=1, 둔각 아님: A,B<=pi/2, A+B=5pi/6
    Rs = []
    N = 20000
    for i in range(N + 1):
        A_ = math.pi / 3 + (math.pi / 6) * i / N          # A in [pi/3, pi/2]
        Rs.append(1 / (2 * math.sin(A_)))
    V.check(3, "3-1 Rmax", max(Rs), float(sp.N(E(V.ans(3, "1", "Rmax")).subs(a, 1))), 1e-6)
    V.check(3, "3-1 Rmin", min(Rs), float(sp.N(E(V.ans(3, "1", "Rmin")).subs(a, 1))), 1e-6)
    # (2) a>b, 둔각 아님, 변 AB 범위. a=5,b=3
    av, bv = 5.0, 3.0
    best = []
    N = 40000
    for i in range(N + 1):
        C = math.acos(bv / av) + (math.pi / 2 - math.acos(bv / av)) * i / N   # A<=pi/2 <=> cosC<=b/a, C<=pi/2
        cc = math.sqrt(av**2 + bv**2 - 2 * av * bv * math.cos(C))
        sinA = av * math.sin(C) / cc
        # 둔각 아님 확인
        if av**2 > bv**2 + cc**2 + 1e-9 or bv**2 > av**2 + cc**2 + 1e-9 or cc**2 > av**2 + bv**2 + 1e-9:
            continue
        best.append((av / (2 * sinA), cc))
    Rmax = max(best)[1]
    Rmin = min(best)[1]
    sub = {a: av, b: bv}
    V.check(3, "3-2 AB@Rmax", Rmax, float(E(V.ans(3, "2", "AB_Rmax")).subs(sub)), 1e-3)
    V.check(3, "3-2 AB@Rmin", Rmin, float(E(V.ans(3, "2", "AB_Rmin")).subs(sub)), 1e-3)
    # (3) AB+CA=d, BC=a (a<d). a=3,d=8: Rr = abc/(2(a+d)) 최대
    import random
    random.seed(1)
    av, dv = 3.0, 8.0
    best = 0
    for i in range(1, 8000):
        bb = dv * i / 8000
        cc = dv - bb
        s = (av + dv) / 2
        area = math.sqrt(max(s * (s - av) * (s - bb) * (s - cc), 0))
        if area == 0:
            continue
        R = av * bb * cc / (4 * area)
        r = area / s
        best = max(best, R * r)
    V.check(3, "3-3 max Rr", best, float(E(V.ans(3, "3")).subs({a: av, d: dv})), 1e-6)


if __name__ == "__main__":
    V.run(globals())
