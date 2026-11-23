#!/usr/bin/env python3
"""UOS-M18 (2018학년도 모의논술 자연계열 4문항) — sympy/열거/수치 검산. 기대값은 YAML 정답(V.ans)에서 읽는다."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
import mpmath as mp

V = Verifier("UOS-M18", "items/UOS/PAST")
P_, N_ = sp.Symbol("p"), sp.Symbol("n")


def p01():
    # 좌표로 직접: 직선 A+s(cosθ,sinθ) 와 원 (x-1)^2+y^2=1 의 교점 중 A 에 가까운 것, 제1사분면 확인
    mp.mp.dps = 40
    exp = mp.mpf(str(sp.N(sp.sympify(V.ans(1, "1-1")), 30)))
    for th in (mp.mpf("1e-3"), mp.mpf("1e-5")):
        c, s_ = mp.cos(th), mp.sin(th)
        disc = (-2 * 3 * c) ** 2 - 4 * 8
        r1 = (6 * c - mp.sqrt(disc)) / 2
        px, py = -2 + r1 * c, r1 * s_
        on_circle = abs((px - 1) ** 2 + py**2 - 1) < mp.mpf("1e-30")
        V.report(1, on_circle and px > 0 and py > 0, f"θ={th}: P=({mp.nstr(px,8)},{mp.nstr(py,8)}) 원 위·제1사분면")
        V.report(1, abs((r1 - 2) / th**2 - exp) < 30 * th**2 + mp.mpf("1e-20"),
                 f"θ={th}: (AP-2)/θ²={mp.nstr((r1-2)/th**2, 12)} / YAML {exp}")
    th = sp.Symbol("th", positive=True)
    xr = 3 * sp.cos(th) - sp.sqrt(9 * sp.cos(th) ** 2 - 8)
    V.check(1, "sympy 극한", sp.limit((xr - 2) / th**2, th, 0, "+"), V.ans(1, "1-1"))


def p02():
    xs = sp.Symbol("xs", real=True)
    A, B, C, D = (sp.Matrix(v) for v in ((-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)))
    Vv = sp.Matrix([3 * xs, 3 * xs, 3])  # 정사영이 AC (y=x) 위
    Pp = (Vv + A + B) / 3
    Qq = (Vv + B + C) / 3
    O = sp.Matrix([0, 0, 0])
    cosv = sp.simplify((Pp.dot(Qq)) / (sp.sqrt(Pp.dot(Pp)) * sp.sqrt(Qq.dot(Qq))))
    V.report(2, sp.simplify(Pp - sp.Matrix([xs, xs - 2, 1])) == sp.zeros(3, 1) and sp.simplify(Qq - sp.Matrix([xs + 2, xs, 1])) == sp.zeros(3, 1),
             "P(x,x-2,1), Q(x+2,x,1)")
    f = sp.lambdify(xs, cosv, "mpmath")
    mp.mp.dps = 30
    vals = [f(mp.mpf(i) / 2000) for i in range(-2000, 2001)]
    m = min(vals)
    exp = mp.mpf(str(sp.N(sp.sympify(V.ans(2, "2-1")), 30)))
    V.report(2, abs(m - exp) < mp.mpf("1e-12"), f"x∈[-1,1] 격자 최솟값 cos = {mp.nstr(m, 12)} / YAML {mp.nstr(exp, 12)}")
    # 엄밀: cos² = 1 - 24/t, t=4x^4+4x^2+25
    expr = sp.simplify(cosv**2 - (1 - 24 / (4 * xs**4 + 4 * xs**2 + 25)))
    V.report(2, expr == 0, "cos² = 1-24/t 항등")
    V.check(2, "x=0 에서 cos", sp.simplify(cosv.subs(xs, 0)), V.ans(2, "2-1"))


def p03():
    # 상태 = 주머니 속 빨간 공 수 r (0,1,2), 총 2개 → 한 시행: 꺼내고 채운다
    def step(dist):
        out = {0: 0, 1: 0, 2: 0}
        for r, pr in dist.items():
            # 빨간 공을 꺼낼 확률 r/2
            for drew_red, pd in ((True, sp.Rational(r, 2)), (False, 1 - sp.Rational(r, 2))):
                if pd == 0:
                    continue
                rem = r - 1 if drew_red else r
                out[rem + 1] += pr * pd * P_        # 빨강 채움
                out[rem] += pr * pd * (1 - P_)      # 흰색 채움
        return out
    d0 = {0: 0, 1: 1, 2: 0}
    dA = step(d0)
    dB = step(dA)
    a1 = sp.simplify(dB[1])
    V.check(3, "3-1 B 후 빨강1·흰1", a1, sp.sympify(V.ans(3, "3-1"), locals={"p": P_}))
    # 해설의 중간값 (A 후 상태)
    V.report(3, [sp.simplify(dA[2] - P_ / 2), sp.simplify(dA[1] - sp.Rational(1, 2)), sp.simplify(dA[0] - (1 - P_) / 2)] == [0, 0, 0],
             "A 후 상태 p/2, 1/2, (1-p)/2")
    pc = sp.simplify(sum(sp.Rational(r, 2) * dB[r] for r in (0, 1, 2)))
    V.check(3, "3-2 C 가 빨간 공", pc, sp.sympify(V.ans(3, "3-2"), locals={"p": P_}))
    V.report(3, sp.simplify(dB[2] - (P_ / 4 + P_**2 / 2)) == 0, "B 후 빨강 2개 확률 p/4+p²/2")
    # 전수열거: p=1/3 에서 모든 경우(꺼냄·채움 색)로 직접 확률
    from fractions import Fraction
    import itertools
    pv = Fraction(1, 3)
    tot_a1 = Fraction(0)
    tot_c = Fraction(0)
    # 재귀 열거
    def rec(r, depth, pr):
        nonlocal tot_a1, tot_c
        if depth == 2:
            # C 꺼냄
            tot_c += pr * Fraction(r, 2)
            return
        for drew_red in (1, 0):
            pd = Fraction(r, 2) if drew_red else 1 - Fraction(r, 2)
            if pd == 0:
                continue
            rem = r - drew_red
            for fill_red in (1, 0):
                pf = pv if fill_red else 1 - pv
                nr = rem + fill_red
                if depth == 1 and nr == 1:
                    tot_a1 += pr * pd * pf
                rec(nr, depth + 1, pr * pd * pf)
    rec(1, 0, Fraction(1))
    V.check(3, "p=1/3 열거: 3-1", sp.Rational(tot_a1.numerator, tot_a1.denominator),
            sp.sympify(V.ans(3, "3-1"), locals={"p": P_}).subs(P_, sp.Rational(1, 3)))
    V.check(3, "p=1/3 열거: 3-2", sp.Rational(tot_c.numerator, tot_c.denominator),
            sp.sympify(V.ans(3, "3-2"), locals={"p": P_}).subs(P_, sp.Rational(1, 3)))


def _count_grid(n):
    """g∘f 와 h 의 그래프 교점 수(연속부 부호변화 + 정확히 0 인 점). 격자 탐색."""
    xs = np.concatenate([np.linspace(-40, 0, 800001)[:-1], np.linspace(0, n + 0.5, 4000001)])
    v = xs * 2.0**xs
    fl = np.floor(v)
    F = v - fl
    H = np.where(xs >= 0, xs / n, -xs)
    d = F - H
    same = fl[:-1] == fl[1:]
    return int(np.sum((d[:-1] * d[1:] < 0) & same) + np.sum(d == 0))


def _count_roots(n):
    """x≥0: G(x)=f(x)-x/n 증가, 톱니 j 번째에서 G(x_j)=j, x_j∈[0,n) 인 j 의 수 + x<0 교점 1(격자 확인)."""
    mp.mp.dps = 40
    G = lambda x: x * mp.mpf(2) ** x - x / n
    Gn = G(mp.mpf(n))
    cnt = 0
    j = 0
    while j < Gn:
        cnt += 1
        j += 1
    return cnt


def p04():
    xs = sp.Symbol("xs", real=True)
    f = xs * 2**xs
    d1, d2 = sp.diff(f, xs), sp.diff(f, xs, 2)
    xmin = sp.solve(d1, xs)[0]
    xinf = sp.solve(d2, xs)[0]
    V.check(4, "4-1 극점 x", xmin, V.ans(4, "4-1", "xmin"))
    V.check(4, "4-1 극솟값", f.subs(xs, xmin), V.ans(4, "4-1", "fmin"))
    V.check(4, "4-1 변곡점 x", xinf, V.ans(4, "4-1", "xinf"))
    V.check(4, "4-1 변곡점 y", f.subs(xs, xinf), V.ans(4, "4-1", "finf"))
    V.report(4, sp.limit(f, xs, -sp.oo) == 0 and sp.limit(f, xs, sp.oo) == sp.oo and sp.N(f.subs(xs, xmin)) > -1, "x→-∞ 0, x→∞ ∞, -1<극솟값<0")
    # 증감: f'<0 (x<xmin), f'>0 (x>xmin); f''<0 (x<xinf), >0 (x>xinf) 표본
    ok = all((float(d1.subs(xs, v)) < 0) == (v < float(xmin)) for v in (-5, -3, -2, -1.2, -1, 0.3, 2)) and \
         all((float(d2.subs(xs, v)) < 0) == (v < float(xinf)) for v in (-6, -3.5, -2, -1, 0.5))
    V.report(4, ok, "증감·오목 표본 부호")
    # 4-2: a_n 수를 격자 탐색으로 직접 세어 n·2^n 과 비교, 합 공식 비교
    an_expr = N_ * 2**N_
    cnts = {}
    for nn in (1, 2, 3, 4):
        cnts[nn] = _count_grid(nn)
        V.report(4, cnts[nn] == nn * 2**nn, f"a_{nn} 격자 직접 세기 {cnts[nn]} / n2^n = {nn*2**nn}")
    a = {}
    for nn in range(1, 11):
        a[nn] = _count_roots(nn) + 1  # x≥0 쪽 (원점 포함 f(n)-1) + x<0 쪽 1
        V.report(4, a[nn] == nn * 2**nn, f"a_{nn} 근 방법 {a[nn]} / n2^n = {nn*2**nn}")
    expr = sp.sympify(V.ans(4, "4-2"), locals={"n": N_})
    for nn in range(1, 9):
        s = sum(sp.Rational(a[k + 1] ** 2, a[k] * a[k + 2]) for k in range(1, nn + 1))
        V.check(4, f"4-2 n={nn} 합", s, expr.subs(N_, nn))


if __name__ == "__main__":
    V.run(globals())
