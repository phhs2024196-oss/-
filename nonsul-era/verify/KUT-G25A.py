#!/usr/bin/env python3
"""KUT-G25A (2025학년도 실전 오전 01~10) — 정답을 설정으로부터 독립 재계산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from fractions import Fraction as Fr
import itertools
V = Verifier("KUT-G25A", "items/KUT/PAST")
R = sp.Rational


def p01():
    # 5^(n/2) 7^(n/4) 13^(n/9) 은 서로 다른 소수의 거듭제곱 → 자연수 ⇔ 세 지수가 모두 자연수
    ns = [m for m in range(1, 200) if all((R(m, q)).q == 1 for q in (2, 4, 9))]
    V.check(1, "최소 n", ns[0], V.ans(1, "1"))


def p02():
    a = sp.symbols("a", positive=True)
    sol = sp.solve(sp.Eq(sp.log(2) / sp.log(3) * sp.log(a) / sp.log(4), -1), a)
    V.check(2, "a", sol[0], V.ans(2, "1"))


def p03():
    a, b = sp.symbols("a b")
    s = sp.solve([sp.Eq(2 * a + b, 8), sp.Eq(a, sp.diff(x**3, x).subs(x, 2))], [a, b])
    V.check(3, "a-b", s[a] - s[b], V.ans(3, "1"))


def p04():
    a_, b_ = sp.symbols("a b", real=True)
    aa = sp.solve(sp.Eq(4 + a_ * sp.tan(sp.pi / 3), 7), a_)[0]
    # 점근선 x=-pi/3 < b < 2pi/9 에서 tan(3b/2)=-sqrt3 인 해
    cand = [sp.Rational(2, 3) * (-sp.pi / 3 + 0)]  # 3b/2=-pi/3
    f = lambda xx: 4 + aa * sp.tan(R(3, 2) * xx)
    bb = cand[0]
    ok = same(f(bb), 1) and same(f(2 * sp.pi / 9), 7)
    # 구간 안 단조 증가 확인(수치)
    xs = [float(bb) + i * (float(2 * sp.pi / 9) - float(bb)) / 200 for i in range(201)]
    vals = [float(f(sp.Float(v))) for v in xs]
    ok = ok and max(vals) <= 7 + 1e-9 and min(vals) >= 1 - 1e-9
    V.report(4, ok and same(aa, V.ans(4, "1", "a")) and same(bb, V.ans(4, "1", "b").replace("pi", "pi")), f"a={aa}, b={bb}")


def p05():
    a, b, c, d = sp.symbols("a b c d")
    f = a * x**3 + b * x**2 + c * x + d
    s = sp.solve([f.subs(x, 0), sp.diff(f, x).subs(x, 0) + 12,
                  sp.diff(f, x).subs(x, 2), sp.diff(f, x, 2).subs(x, 2)], [a, b, c, d])
    V.check(5, "f(2)", f.subs(s).subs(x, 2), V.ans(5, "1"))


def p06():
    a, b = sp.symbols("a b")
    t_ = sp.symbols("t")
    f = sp.Function("f")
    # 미분식: 2x^4+ax^2+bx = 2x+f(x)+3x^2 → f
    fx = 2 * x**4 + a * x**2 + b * x - 2 * x - 3 * x**2
    lhs = sp.integrate(2 * t_**4 + a * t_**2 + b * t_, (t_, 0, x))
    rhs = sp.integrate(2 * t_ + fx.subs(x, t_), (t_, 0, x)) + x**3 + a
    s = sp.solve([sp.expand(lhs - rhs).subs(x, 0), fx.subs(x, 1) + 2], [a, b], dict=True)[0]
    fs = fx.subs(s)
    ident = sp.simplify(sp.expand((lhs - rhs).subs(s))) == 0
    V.report(6, ident and same(sp.integrate(fs, (x, 0, 1)), V.ans(6, "1")), f"f={fs}, 항등식 {ident}")


def p07():
    a, b, c, d = sp.symbols("a b c d")
    f = a * x**3 + b * x**2 + c * x + d
    eqs = [(r * f.subs(x, r) - r).subs({}) for r in (-2, -1, 1)]  # a f(a) - a = 0
    s = sp.solve(eqs + [f.subs(x, 2) - 25], [a, b, c, d])
    fs = f.subs(s)
    # (가) 극한 존재 확인
    ok = all(sp.limit((x * fs - r) / (x - r), x, r) is not None and sp.limit((x * fs - r) / (x - r), x, r).is_finite for r in (-2, -1, 1))
    V.report(7, ok and same(fs.subs(x, 3), V.ans(7, "1")), f"f={sp.factor(fs)}")


def p08():
    F = sp.symbols("F")
    roots = sp.solve(F**3 - 3 * F**2 - 4 * x**2 * F + 12 * x**2, F)
    ok = {sp.simplify(r) for r in roots} == {3, 2 * x, -2 * x}
    g = lambda v: min(3, 2 * abs(v)) if True else 0
    f = lambda v: sp.Min(3, 2 * sp.Abs(v))
    # 식 만족(수치 격자), 최댓값 3 최솟값 0
    grid = [R(i, 8) for i in range(-48, 49)]
    ok = ok and all(sp.simplify(f(v) ** 3 - 3 * f(v) ** 2 - 4 * v**2 * f(v) + 12 * v**2) == 0 for v in grid)
    ok = ok and max(f(v) for v in grid) == 3 and min(f(v) for v in grid) == 0
    tot = f(-1) + f(0) + f(R(11, 4))
    V.report(8, ok and same(tot, V.ans(8, "1")), f"합={tot}")


def back(v):
    out = []
    if (5 / v).denominator == 1:
        out.append(5 / v)  # 이전항이 정수이고 5/이전항=v
    if ((v + 1) / 2).denominator != 1:
        out.append((v + 1) / 2)  # 이전항이 비정수이고 2*이전항-1=v
    return out


def fwd(a):
    return 5 / a if a.denominator == 1 else 2 * a - 1


def p09():
    seq = {6: Fr(2)}
    for n_ in range(6, 27):
        seq[n_ + 1] = fwd(seq[n_])
    V.check(9, "9-1 합", 2 * sum(sp.Rational(seq[k].numerator, seq[k].denominator) for k in range(6, 28)), V.ans(9, "9-1"))
    cur = {Fr(2)}
    for _ in range(5):
        cur = {p for v in cur for p in back(v)}
    # 정방향 재확인
    pairs = []
    for a1 in cur:
        s = [a1]
        for _ in range(5):
            s.append(fwd(s[-1]))
        assert s[5] == 2
        pairs.append((a1, s[1]))
    sums = [sp.Rational((a + b).numerator, (a + b).denominator) for a, b in pairs]
    V.check(9, "9-2 max", max(sums), V.ans(9, "9-2", "max"))
    V.check(9, "9-2 min", min(sums), V.ans(9, "9-2", "min"))


def p10():
    def g_at(kk, tt):
        # |f|=t 의 서로 다른 실근 개수(x<0 영역은 f=±t, x>=0 영역 동일)
        if tt < 0:
            return 0
        cnt = set()
        for sg in ((1, -1) if tt != 0 else (1,)):
            for expr, lo, hi in ((-x**2 + 6 * x + kk, None, 0), (x**2 - 4 * x + kk, 0, None)):
                for r in sp.Poly(expr - sg * tt, x).real_roots():
                    if (lo is None or r >= lo) and (hi is None or r < hi):
                        cnt.add(r)
        return len(cnt)
    V.check(10, "10-1 g(1)", g_at(2, 1), V.ans(10, "10-1"))
    tot_ok = True
    for kk in (R(1, 2), R(1), R(3, 2)):
        pts = [R(0), kk, 4 - kk]
        jumps = []
        grid = sorted({R(i, 20) for i in range(-20, 100)} | set(pts))
        eps = R(1, 1000)
        for tt in grid:
            if g_at(kk, tt - eps) != g_at(kk, tt) or g_at(kk, tt) != g_at(kk, tt + eps):
                jumps.append(tt)
        tot_ok = tot_ok and sorted(jumps) == sorted(pts) and sum(jumps) == 4
    V.report(10, tot_ok and V.ans(10, "10-2") == "4", "불연속점 {0,k,4-k} 합 4 (k=1/2,1,3/2)")


if __name__ == "__main__":
    V.run(globals())
