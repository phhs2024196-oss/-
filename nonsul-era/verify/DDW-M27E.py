import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import re, numpy as np

V = Verifier("DDW-M27E", "items/DDW/PAST")


def p01():
    f = lambda u: 1 - abs(2*u - 1)
    g = lambda u: 1 - abs(2*f(u) - 1)
    area = sp.integrate(sp.Piecewise((2*x, x < sp.Rational(1, 2)), (2 - 2*x, True)), (x, 0, 1))
    V.check(1, "1-1 넓이", area, V.ans(1, "1-1"))
    gf = lambda u: 1 - abs(2*(1 - abs(2*u - 1)) - 1)
    pieces = [(4*x, 0, sp.Rational(1, 4)), (-4*x + 2, sp.Rational(1, 4), sp.Rational(1, 2)),
              (4*x - 2, sp.Rational(1, 2), sp.Rational(3, 4)), (-4*x + 4, sp.Rational(3, 4), 1)]
    for b in (sp.Rational(1, 10), sp.Rational(37, 100), sp.Rational(1, 2), sp.Rational(83, 100), sp.Rational(99, 100)):
        roots = [sp.solve(sp.Eq(e, b), x)[0] for e, lo, hi in pieces]
        assert all(lo <= r <= hi for r, (e, lo, hi) in zip(roots, pieces))
        assert all(gf(r) == b for r in roots)          # 원래 정의식으로 되대입
        V.check(1, f"1-2 b={b} 근 개수", len(set(roots)), V.ans(1, "1-2"))
        V.check(1, f"1-3 b={b} 근의 합", sum(roots), V.ans(1, "1-3"))


def p02():
    xx = sp.symbols("xx", real=True)
    # (가): 진수 조건 + 로그 부등식을 수치 격자로
    ok = lambda u: u > 8 and (u - 8)*(u - 6) <= 3
    grid = [8 + i/1000 for i in range(-3000, 3001)]
    sol = [u for u in grid if u > 8 and sp.log(u - 8, 3) + sp.log(u - 6, 3) <= 1 + 1e-12] if False else [u for u in grid if ok(u)]
    lo, hi = min(sol), max(sol)
    V.report(2, 8 < lo < 8.0011 and hi == 9 and V.ans(2, "1-1").replace(" ", "") == "8<x<=9", f"2-1 범위 (8, 9]: 격자 {lo}~{hi} / YAML {V.ans(2, '1-1')}")
    # (나)
    Y = sp.symbols("Y", positive=True)
    ysol = sp.solve_univariate_inequality(Y**2 - 8*Y - 9 <= 0, Y, relational=False, domain=sp.Interval.open(0, sp.oo))
    V.check(2, "2-2 3^y<=9 -> y<=2", sp.log(ysol.end, 3), 2)
    V.report(2, V.ans(2, "1-2").replace(" ", "") == "y<=2", f"2-2 YAML {V.ans(2, '1-2')}")
    # (3): 두 항 모두 증가 → (9, 2)
    val = sp.log(9, sp.sqrt(3)) + sp.sqrt(3**2)
    V.check(2, "2-3 최댓값", sp.simplify(sp.expand_log(val, force=True)), V.ans(2, "1-3"))
    xs = np.linspace(8.0001, 9, 2000); ys = np.linspace(-5, 2, 2000)
    mx = np.max(np.log(xs)/np.log(np.sqrt(3))) + np.max(np.sqrt(3.0**ys))
    V.report(2, abs(mx - 7) < 1e-3, f"2-3 수치 상한 {mx}")


def p03():
    a = sp.symbols("a", positive=True)
    fl = lambda u: (u + 2)*(u + 4)
    fr = lambda u: (u - 2)*(u - 6)
    # g(x)=f(x)f(x-a) 의 x=0 에서: 좌극한 f(0-)f(-a), 우극한 f(0+)f(-a), g(0)=f(0)f(-a)
    f0m, f0p = fl(0), fr(0)
    sol1 = sp.solve(sp.Eq(f0m*fl(-a), f0p*fl(-a)), a)
    V.check(3, "3-1 a", sol1, [int(s) for s in V.ans(3, "1-1").split(",")])
    # x=a: f(x) 는 a>0 에서 연속(오른쪽 식), f(x-a): 좌 f(0-)=8, 우 f(0+)=12
    sol2 = sp.solve(sp.Eq(f0m*fr(a), f0p*fr(a)), a)
    V.check(3, "3-2 a", sol2, [int(s) for s in V.ans(3, "1-2").split(",")])
    both = sorted(set(sol1) & set(sol2))
    V.check(3, "3-3 a", both, [int(V.ans(3, "1-3"))])
    # 수치 확인: 후보 a 에서 g 의 모든 점프를 격자로 점검
    def f(u): return fl(u) if u < 0 else fr(u)
    def jumps(av):
        g = lambda u: f(u)*f(u - av)
        e = 1e-9
        return [abs(g(c + e) - g(c - e)) for c in (0, av)]
    for av, expect in ((2, True), (4, False), (6, False), (3, False)):
        V.check(3, f"3-3 a={av} 연속 여부", all(j < 1e-6 for j in jumps(av)), expect)


def p04():
    a = sp.symbols("a", integer=True)
    ans1 = [aa for aa in range(-30, 31) if ((0 - 3)**2 + aa)*(0 + aa) <= 0]
    V.check(4, "4-1 a", ans1, [int(s) for s in V.ans(4, "1-1").split(",")])
    b, c = sp.symbols("b c")
    g = -x**3 + b*x**2 + c*x
    csol = sp.solve(sp.Eq(sp.diff(g, x).subs(x, 3), 0), c)[0]
    V.check(4, "4-2 c", csol, V.ans(4, "1-2"))
    # 4-3: 정수 b 중 g 가 x=3 에서 극대인 최대값을 부호 변화로 직접 판정
    def is_local_max(bv):
        cv = 27 - 6*bv
        gp = lambda u: -3*u**2 + 2*bv*u + cv
        return gp(3 - 1e-6) > 0 > gp(3 + 1e-6)
    good = [bv for bv in range(-50, 50) if is_local_max(bv)]
    V.check(4, "4-3 b 최대", max(good), V.ans(4, "1-3"))
    V.check(4, "4-3 b<9 범위", good, list(range(-50, 9)))


def p05():
    words = {"㉠": ("산출", "Ⓗ"), "㉡": ("발생", "Ⓒ"), "㉢": ("가정", "Ⓔ"), "㉣": ("유효", "Ⓖ"), "㉤": ("적정", "Ⓕ")}
    defs = {"산출": "계산하여 냄", "발생": "어떤 일이나 사물이 생겨남", "가정": "어떤 조건이나 전제를 내세움", "유효": "보람이나 효과가 있음", "적정": "알맞고 바른 정도"}
    box = {"Ⓗ": "계산하여 냄", "Ⓒ": "어떤 일이나 사물이 생겨남", "Ⓔ": "어떤 조건이나 전제를 내세움", "Ⓖ": "보람이나 효과가 있음", "Ⓕ": "알맞고 바른 정도"}
    for k, (w, _) in words.items():
        V.report(5, box[V.ans(5, "1-1", k)] == defs[w], f"5-1 {k}{w} -> {V.ans(5, '1-1', k)}")
    # 5-2: 손익분기점 판매량 변화 방향
    F, P, v = sp.symbols("F P v", positive=True)
    bep = F/(P - v)
    base = {F: 20000, P: 500, v: 100}
    V.check(5, "본문 예 BEP 50", bep.subs(base), 50)
    cases = {"①": (F, v - 10), "②": (F + 1000, v), "③": (F + 1000, v), "④": (F, v + 10), "⑤": (F + 1000, v)}
    lowers = [k for k, (nf, nv) in cases.items() if bep.subs({F: nf, v: nv}).subs(base) < 50]
    V.report(5, lowers == [V.ans(5, "1-2")], f"5-2 BEP 를 낮추는 보기 {lowers} / YAML {V.ans(5, '1-2')}")


if __name__ == "__main__": V.run(globals())
