"""KW-G26A 검산: 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, itertools, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
from fractions import Fraction as Fr

V = Verifier("KW-G26A", "items/KW/PAST")
a_ = sp.Symbol("a")
k_ = sp.Symbol("k")


def clip(subject, clipper):
    """Sutherland-Hodgman(볼록 다각형 둘의 교집합), 좌표는 Fraction."""
    def inside(p, a, b):  # 반시계 clipper 의 왼쪽
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def cross(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cross(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cross(p, q, a, b))
        if not out:
            return []
    return out


def area(poly):
    if len(poly) < 3:
        return Fr(0)
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2


def ccw(poly):
    return poly if sum(poly[i][0] * poly[(i + 1) % 3][1] - poly[(i + 1) % 3][0] * poly[i][1] for i in range(3)) > 0 else poly[::-1]


T1 = ccw([(Fr(-1), Fr(-1)), (Fr(1), Fr(-1)), (Fr(1), Fr(3))])
T2 = ccw([(-x_, -y_) for x_, y_ in T1])


def overlap(shift):
    t2 = ccw([(x_ + shift, y_) for x_, y_ in T2])
    return area(clip(T1, t2))


def S_num(kv):
    r = -mp.sign(kv) * mp.cbrt(abs(kv)) if kv != 0 else mp.mpf(0)
    pts = [0, 1] if not (0 < r < 1) else [0, r, 1]
    return mp.quad(lambda x: abs(x ** 3 + kv), pts)


def p01():
    # [1]
    U = range(1, 7)
    subsets = [frozenset(c) for r in range(7) for c in itertools.combinations(U, r)]
    cnt = sum(1 for A in subsets for B in subsets if len(A) * len(B) != 0 and len(B) < 6 and A <= B)
    V.check(1, "1 경우의 수", cnt, V.ans(1, "1"))
    # [2](1)
    V.check(1, "2-1 겹친 넓이", sp.Rational(area(clip(T1, T2)).numerator, area(clip(T1, T2)).denominator), V.ans(1, "2-1"))
    # [2](2): 평행이동 거리 격자(1/200)에서의 최댓값, 정확한 유리수
    best = max(overlap(Fr(i, 200)) for i in range(-1400, 1401))
    V.check(1, "2-2 최대 넓이", sp.Rational(best.numerator, best.denominator), V.ans(1, "2-2"))
    # [3](1): 구간별 식을 표본 k 에서 수치 적분과 비교
    ks = sp.Symbol("k")
    for key, samples in (("low", [-3, -1.5, -1]), ("mid", [-0.9, -0.5, -0.125, -0.01]), ("high", [0, 0.5, 2])):
        expr = sp.sympify(V.ans(1, "3-1", key), locals={"k": ks, "cbrt": lambda v: sp.real_root(v, 3)})
        for kv in samples:
            V.check(1, f"3-1 {key} k={kv}", float(expr.subs(ks, kv)), float(S_num(mp.mpf(kv))), 1e-9)
    # [3](2): 수치 최소화
    grid = [mp.mpf(i) / 1000 for i in range(-1500, 1501)]
    kmin = min(grid, key=S_num)
    V.check(1, "3-2 k", float(kmin), float(sp.sympify(V.ans(1, "3-2", "k"))), 1e-3)
    kq = sp.Rational(V.ans(1, "3-2", "k"))
    V.check(1, "3-2 min(YAML k)", float(S_num(mp.mpf(kq.p) / kq.q)), float(sp.sympify(V.ans(1, "3-2", "min"))), 1e-12)
    V.check(1, "3-2 min(grid)", float(S_num(kmin)), float(sp.sympify(V.ans(1, "3-2", "min"))), 1e-4)


def p02():
    al, be = sp.symbols("alpha beta", real=True)
    # [1](1): 코사인법칙의 PQ^2 와 거리공식의 PQ^2 가 같고, 덧셈정리 항등식(수치)
    PQ2 = (sp.cos(be) - sp.cos(al)) ** 2 + (sp.sin(be) - sp.sin(al)) ** 2
    V.check(2, "1-1 PQ^2", sp.simplify(PQ2 - (2 - 2 * sp.cos(al - be))), 0)
    bad = search(lambda p, q: abs(mp.cos(p + q) - (mp.cos(p) * mp.cos(q) - mp.sin(p) * mp.sin(q))) < 1e-12,
                 [mp.mpf(i) / 7 for i in range(-20, 21)], [mp.mpf(i) / 5 for i in range(-20, 21)])
    V.proof(2, "1-1 덧셈정리", bad)
    # [1](2)
    h = sp.Symbol("h")
    V.check(2, "1-2 (cos x)'", sp.limit((sp.cos(x + h) - sp.cos(x)) / h, h, 0), -sp.sin(x))
    V.check(2, "1-2 lim (cos h-1)/h", sp.limit((sp.cos(h) - 1) / h, h, 0), 0)
    # [2]
    av = sp.Symbol("a", real=True)
    z = sp.expand((1 + sp.I) ** 2025 + av * (sp.I - 1) ** 2026)
    asol = sp.solve(sp.im(z), av)
    V.check(2, "2 a", asol[0], V.ans(2, "2", "a"))
    zz = sp.simplify(z.subs(av, asol[0]))
    V.check(2, "2 z/2^1012", zz / 2 ** 1012, 1)
    cnt = len({2 ** (1012 // p) for p in range(1, 1013) if 1012 % p == 0})  # b=2^(1012/p), p 자연수
    V.check(2, "2 b 개수", cnt, V.ans(2, "2", "count"))
    # [3](1): 교점 P 를 직접 구해 삼각형 넓이와 비교
    fa = sp.sympify(V.ans(2, "3-1"), locals={"a": a_})
    Q = (sp.Rational(-1, 2), sp.Rational(7, 2))
    for av_ in (-3, -1, 0, sp.Rational(1, 2), 2, 5, 10):
        xs = sp.solve(sp.Eq((x - 1) ** 2 + (av_ * x + 1) ** 2, 2), x)
        xs = [r for r in xs if r != 0]
        assert len(xs) == 1
        P = (xs[0], av_ * xs[0])
        A = sp.Abs(P[0] * Q[1] - P[1] * Q[0]) / 2
        V.check(2, f"3-1 a={av_}", sp.simplify(A), fa.subs(a_, av_))
    # [3](2)
    g = ((a_ - 1) * (a_ + 7)) / (2 * (a_ ** 2 + 1))
    # 직접 구한 넓이 함수(부호 있는 쪽)로 도함수 영점
    crit = sp.solve(sp.diff(g, a_), a_)
    vals = sorted([(c, sp.Abs(g.subs(a_, c))) for c in crit], key=lambda t: float(t[0]))
    V.check(2, "3-2 a1", vals[0][0], V.ans(2, "3-2", "a1"))
    V.check(2, "3-2 max1", vals[0][1], V.ans(2, "3-2", "max1"))
    V.check(2, "3-2 a2", vals[1][0], V.ans(2, "3-2", "a2"))
    V.check(2, "3-2 max2", vals[1][1], V.ans(2, "3-2", "max2"))


if __name__ == "__main__":
    V.run(globals())
