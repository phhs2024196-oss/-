#!/usr/bin/env python3
"""KUT-M27-01~10 (한국기술교육대 2027학년도 온라인 모의논술) — 문제 설정에서 독립 재계산, 기대값은 YAML(V.ans)."""
import sys, pathlib, math, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-M27", "items/KUT/PAST")
R = sp.Rational


def p01():
    e = (3 * 3**sp.sqrt(3))**sp.sqrt(3) / 3**(sp.sqrt(3) - 1)
    V.check(1, "값(수치)", sp.N(e, 30), V.ans(1, "1"), tol=1e-12)
    V.check(1, "값(기호)", sp.simplify(sp.powsimp(sp.expand_power_base(e, force=True), force=True)), V.ans(1, "1"))


def p02():
    for k in (R(2), R(7, 3), R(5)):
        a, b, c = k**R(1, 2), k**R(1, 3), k**R(1, 4)
        assert abs(sp.N(a**2 - b**3)) < 1e-12 and abs(sp.N(b**3 - c**4)) < 1e-12
        s = sp.log(b) / sp.log(a) + sp.log(c) / sp.log(b) + sp.log(a) / sp.log(c)
        V.check(2, f"k={k}", sp.N(s, 30), V.ans(2, "1"), tol=1e-12)


def p03():
    # 0<=x<=3π 에서 sin²x+2cos x+2+k=0 의 서로 다른 실근 개수를 t=cos x 의 근으로부터 정확히 센다
    def count(kv):
        tt = sp.symbols("tt")
        cnt = 0
        for r in sp.solve(sp.Eq(-tt**2 + 2*tt + 3 + kv, 0), tt):
            if not r.is_real or abs(r) > 1:
                continue
            cnt += 3 if abs(r) < 1 else 2
        return cnt
    good = [kv for kv in range(-30, 31) if count(kv) == 3]
    V.check(3, "조건 k 목록", good, [-3, -2, -1])
    V.check(3, "합", sum(good), V.ans(3, "1"))
    # 수치 직접 확인: k=-1 에서 실근 3개
    import numpy as np
    xs = np.linspace(0, 3*math.pi, 3_000_001)
    for kv in (-3, -2, -1, 0, 1):
        v = np.sin(xs)**2 + 2*np.cos(xs) + 2 + kv
        sc = int(np.sum(v[:-1]*v[1:] < 0)) + int(np.sum(v == 0))
        V.report(3, (sc == 3) == (kv in good), f"k={kv} 부호변화 {sc}")


def p04():
    a, b, c = sp.symbols("a b c")
    f = -x**2 + 2*x + b          # lim (f+x^2)/(x+1)=2 -> f+x^2 = 2x+b (일차, 계수 2)
    assert sp.limit((f + x**2) / (x + 1), x, sp.oo) == 2
    bs = sp.solve(f.subs(x, 3) - 1, b)[0]
    f = f.subs(b, bs)
    V.check(4, "a", sp.limit((f - 1) / (x - 3), x, 3), V.ans(4, "1"))
    # 다항식 일반형(3차까지) 으로 전수 해: 극한 조건을 만족하는 계수 해가 하나뿐
    c3, c2, c1, c0 = sp.symbols("c3 c2 c1 c0")
    g = c3*x**3 + c2*x**2 + c1*x + c0
    # x→∞ 극한이 유한(=2): 차수 ≤ 2 이므로 c3=0, 이차 계수 c2+1=0, 일차 계수 c1=2
    sol = sp.solve([c3, c2 + 1, c1 - 2, g.subs(x, 3) - 1], [c3, c2, c1, c0], dict=True)
    V.report(4, len(sol) == 1 and sol[0][c0] == 4, f"계수 {sol}")


def p05():
    a = sp.symbols("a", real=True)
    f = sp.expand((x + a)*(x - a)*(x - 2*a))
    sols = [s_ for s_ in sp.solve(sp.Eq(sp.diff(f, x).subs(x, 1), -2), a) if f.subs({x: 0, a: s_}) > 0]
    V.check(5, "a 개수", len(sols), 1)
    fa = f.subs(a, sols[0])
    V.check(5, "f(4)", fa.subs(x, 4), V.ans(5, "1"))
    # 근집합 확인
    V.check(5, "근집합", sorted(sp.roots(sp.Poly(fa, x)).keys()), sorted([-sols[0], sols[0], 2*sols[0]]))


def p06():
    a_ = sp.symbols("a_", real=True)
    L = lambda av: (av**2 - 4*av + 3, 2*av - 6)
    cands = sp.solve(sp.Abs(a_**2 - 4*a_ + 3) - sp.Abs(2*a_ - 6), a_)
    cands2 = sorted(set(sp.solve(a_**2 - 4*a_ + 3 - (2*a_ - 6), a_)) | set(sp.solve(a_**2 - 4*a_ + 3 + (2*a_ - 6), a_)))
    V.check(6, "해집합", cands2, [-1, 3])
    V.check(6, "합", sum(cands2), V.ans(6, "1"))
    # 수치: |f| 가 x=a 에서 연속인지(점프) 격자 점검
    def jump(av):
        l, r = abs(av**2 - 4*av + 3), abs(2*av - 6)
        return abs(l - r)
    bad = [av/4 for av in range(-40, 41) if (jump(av/4) < 1e-12) != (av/4 in (-1, 3))]
    V.report(6, not bad, f"격자 점검 불일치 {bad[:3] or '없음'}")


def p07():
    a = sp.symbols("a", real=True)
    f = sp.expand((x + 1)**2 * (x - a) - 2)
    # 조건: lim_{x→-1+} g = f(-1) , 좌극한+2 = f(-1)-2+2 = f(-1)
    gr = sp.limit((x + 1)**2 / (f + 2), x, -1, "+")      # (x+1)^2/((x+1)^2 (x-a)) = 1/(x-a)
    sa = sp.solve(sp.Eq(gr, f.subs(x, -1)), a)
    V.check(7, "a 개수", len(sa), 1)
    fa = f.subs(a, sa[0])
    V.check(7, "f(x)", sp.expand(fa), sp.expand((x + 1)**2*(x + R(1, 2)) - 2))
    lo, hi = extrema(fa, x, -2, -R(3, 4))
    V.check(7, "최댓값", hi, -2)
    V.check(7, "최솟값", lo, -R(7, 2))
    V.check(7, "합", lo + hi, V.ans(7, "1"))
    # 유일성: f(x)+2 가 (x+1)^2 인수를 갖는 삼차(모닉)를 일반 계수로 풀어 a 가 유일한지
    p, q, r = sp.symbols("p q r")
    g = x**3 + p*x**2 + q*x + r
    # f(-1)+2=0 이고 (x+1)^2 | f+2  -> f+2 의 -1 에서의 값·도함수가 0
    s = sp.solve([(g + 2).subs(x, -1), sp.diff(g + 2, x).subs(x, -1)], [q, r], dict=True)[0]
    gg = sp.expand(g.subs(s) + 2)
    V.report(7, sp.rem(sp.Poly(gg, x), sp.Poly((x + 1)**2, x)).is_zero, "f+2 가 (x+1)^2 로 나누어떨어짐")


def p08():
    a, p, q, r = sp.symbols("a p q r")
    t = sp.symbols("t")
    g = x**3 + a*x
    assert sp.expand(g + g.subs(x, -x)) == 0
    # f 를 3차 이하 일반형으로 두고 (가) 항등식으로 계수를 푼다
    f = sp.symbols("c0:4")
    F = sum(f[i]*x**i for i in range(4))
    lhs = F + sp.integrate(((x - t)*sp.diff(F, x).subs(x, t)), (t, 0, x))
    eqs = sp.Poly(sp.expand(lhs - g), x).all_coeffs()
    sol = sp.solve(eqs, list(f) + [a], dict=True)
    V.check(8, "해 개수", len(sol), 1)
    s0 = sol[0]
    fv = F.subs(s0).subs(x, 1) + g.subs(a, s0[a]).subs(x, 1)
    V.check(8, "f(1)+g(1)", fv, V.ans(8, "1"))
    V.check(8, "f", sp.expand(F.subs(s0)), 3*x**2 - 6*x)


def p09():
    import numpy as np
    def ok(av):
        xs = np.linspace(2, 4, 200001)
        t_ = np.log(xs) / math.log(av)
        return bool(np.all(np.abs(t_**2 - 5*t_ + 5) >= 1 - 1e-12))
    good = [av for av in range(2, 60) if ok(av)]
    V.check(9, "조건 a 처음 5개", good[:5], [4, 5, 6, 7, 8])
    V.check(9, "a>=4 전부", good, list(range(4, 60)))
    V.check(9, "합", sum(good[:5]), V.ans(9, "1"))


def p10():
    c = sp.symbols("c0:4")
    f = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
    # (다) f(0)=0, f'(0)=1  -> f = x Q(x), Q(0)=1. 극값 조건: g'(x)=|f|-|x| = |x|(|Q|-1), g'(2)=g'(6)=0, (나)
    Qa, Qb = sp.symbols("Qa Qb")
    sols = []
    for s2, s6 in itertools.product((1, -1), repeat=2):
        Qx = sp.symbols("q2 q1 q0")
        Q = Qx[0]*x**2 + Qx[1]*x + Qx[2]
        sol = sp.solve([Q.subs(x, 2) - s2, Q.subs(x, 6) - s6, Q.subs(x, 0) - 1], Qx, dict=True)[0]
        Qs = Q.subs(sol)
        fx = sp.expand(x*Qs)
        if fx.subs(x, 2) < 0 and fx.subs(x, 6) > 0 and sp.Poly(fx, x).LC() > 0:
            sols.append(fx)
    V.check(10, "(1) f 개수", len(sols), 1)
    fx = sols[0]
    Q = sp.cancel(fx / x)
    V.check(10, "(1) Q(2)+Q(6)", Q.subs(x, 2) + Q.subs(x, 6), V.ans(10, "10-1"))
    V.check(10, "f(x)", fx, x**3/4 - R(3, 2)*x**2 + x)
    crit = sp.solve(sp.diff(fx, x), x)
    V.check(10, "(2) α+β", sum(crit), V.ans(10, "10-2"))
    # g 의 극값 확인: g'(x)=|f|-|x| 의 x=2,6 에서의 부호 변화(수치)
    gp = lambda v: abs(float(fx.subs(x, v))) - abs(v)
    V.report(10, gp(2) == 0 or abs(gp(2)) < 1e-12, "g'(2)=0")
    V.report(10, gp(1.9)*gp(2.1) < 0 and gp(5.9)*gp(6.1) < 0, "x=2, 6 에서 g' 부호 변화(극값)")


if __name__ == "__main__":
    V.run(globals())
