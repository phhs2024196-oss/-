import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
from fractions import Fraction
from math import comb
V = Verifier("UOS-M27", "items/UOS/PAST")


def p01():
    t = sp.symbols("t", real=True)
    f = lambda xx, a: 3**(3*xx) + 3**(-3*xx) - 3*(3**(2*xx) + 3**(-2*xx)) - 2*(3**(1 + xx) + 3**(1 - xx)) + a
    h = lambda tt, a: tt**3 - 3*tt**2 - 9*tt + 6 + a
    # 치환 확인: f(x)=h(3^x+3^-x)
    ok = all(abs(f(xv, 0.7) - h(3**xv + 3**-xv, 0.7)) < 1e-9 for xv in np.linspace(0, 3, 31))
    V.report(1, ok, "f(x)=h(3^x+3^-x) 치환 일치")
    # (a) 최소점: 격자 수치 최소가 정답 x
    xs = np.linspace(0, 4, 400001); fv = f(xs, 0.0)
    V.check(1, "1-1 argmin x", float(xs[fv.argmin()]), float(sp.N(S(V.ans(1, "1-1")))), 1e-4)
    # (b) a 별 영점 개수: 직접 f 에서 센다(정확 풀이: x=log3((t+sqrt(t^2-4))/2))
    cond = V.ans(1, "1-2").replace("a=21", "a==21")
    bad = []
    for a in [-40, -5, 0, 10, 15, 15.99, 16, 16.01, 18, 20, 20.99, 21, 21.01, 25, 40]:
        p = sp.Poly(t**3 - 3*t**2 - 9*t + 6 + sp.nsimplify(a), t)
        ts = {r for r in sp.real_roots(p) if r >= 2}
        cnt = len(ts)
        for r in ts:   # 각 t 근에 대응 x>=0 의 f 값이 0 인지
            xv = sp.log((r + sp.sqrt(r**2 - 4))/2, 3)
            if abs(float(sp.N(f(xv, sp.nsimplify(a))))) > 1e-8: bad.append(("f!=0", a))
        if (cnt == 1) != bool(eval(cond, {"a": a})): bad.append((a, cnt))
    V.proof(1, "1-2 a 조건 vs 영점 수", bad)


def p02():
    al, be = sp.symbols("alpha beta", real=True)
    F = sp.sympify(V.ans(2, "2-1"), locals={"alpha": al, "beta": be})
    # 접선 두 개를 유리수 기울기로 구성: m1 m2 = 1/alpha, m1+m2 = beta/alpha
    bad = []
    for m1, m2 in [(1, 2), (2, 3), (-1, 3), (sp.Rational(1, 2), 4), (-2, -5), (3, -sp.Rational(1, 3))]:
        m1, m2 = sp.nsimplify(m1), sp.nsimplify(m2)
        a_, b_ = 1/(m1*m2), (m1 + m2)/(m1*m2)
        P = sp.Matrix([a_, b_]); Q = sp.Matrix([1/m1**2, 2/m1]); R = sp.Matrix([1/m2**2, 2/m2])
        # Q, R 이 포물선 위, P 가 접선 y*y1=2(x+x1) 위
        for (x1, y1) in [(Q[0], Q[1]), (R[0], R[1])]:
            if y1**2 != 4*x1 or b_*y1 != 2*(a_ + x1): bad.append(("접선", m1, m2))
        # 접선의 기울기 m 이 판별식 조건: 기울기 m 직선 y=m x+b-m a 가 포물선과 중근
        for m in (m1, m2):
            xx = sp.symbols("xx")
            D = sp.discriminant(sp.expand((m*xx + b_ - m*a_)**2 - 4*xx), xx)
            if sp.simplify(D) != 0: bad.append(("D", m))
        dot = (Q - P).dot(R - P)
        if sp.simplify(dot - F.subs({al: a_, be: b_})) != 0: bad.append(("dot", m1, m2))
        if not (b_**2 - 4*a_ > 0): bad.append(("disc", m1, m2))
    V.proof(2, "2-1 내적식(구체 접선쌍)", bad)
    V.check(2, "내적 인수분해", sp.factor(F), (al + 1)*(4*al - be**2))
    sols = [s_ for s_ in sp.solve(F, al) if sp.simplify(be**2 - 4*s_) != 0]
    V.check(2, "2-2 alpha", sols[0], V.ans(2, "2-2", "alpha"))
    V.report(2, len(sols) == 1, f"2-2 해 개수 {len(sols)}")


def p03():
    xx = sp.symbols("xx", real=True)
    f = sp.cos(xx) - 1 + xx**2/2
    V.check(3, "f''", sp.diff(f, xx, 2), 1 - sp.cos(xx))
    V.check(3, "f'(0)", sp.diff(f, xx).subs(xx, 0), 0)
    g = np.linspace(-12, 12, 240001)
    V.proof(3, "3-1 cos x >= 1-x^2/2 (격자)", [] if np.all(np.cos(g) - (1 - g**2/2) >= -1e-12) else ["위반"])
    V.check(3, "제곱차", sp.expand((sp.Rational(1, 2) - xx**2/2)**2 - (sp.Rational(1, 4) - xx**2)), (xx**4 + 2*xx**2)/4)
    u = np.linspace(-0.5, 0.5, 100001)
    V.proof(3, "원 ⊂ 영역 (r=1/2)", [] if np.all(np.cos(u) >= 0.5 + np.sqrt(0.25 - u**2) - 1e-12) else ["위반"])
    # 영역 안 최대 원 수치 탐색(곡선까지 거리와 x축까지 거리의 최소의 최대)
    X = np.linspace(-np.pi/2, np.pi/2, 4001); Yc = np.cos(X)
    best = 0.0
    for cx in np.linspace(-0.6, 0.6, 121):
        for cy in np.linspace(0.05, 1.0, 191):
            if cy > np.cos(cx): continue
            d = np.sqrt((X - cx)**2 + (Yc - cy)**2).min()
            best = max(best, min(d, cy))
    V.check(3, "3-2 수치 최대 반지름", best, float(sp.N(S(V.ans(3, "3-2")))), 2e-3)


def p04():
    for N in range(4, 16):
        tot = comb(N, 4); bal = 0; cons = 0
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                for c in range(b + 1, N + 1):
                    for d in range(c + 1, N + 1):
                        if a + d == b + c or a + b == c + d or a + c == b + d:
                            bal += 1; cons += (c == b + 1)
        par = "even" if N % 2 == 0 else "odd"
        n_ = sp.Symbol("n")
        pa = Fraction(bal, tot); pb = Fraction(cons, bal)
        ea = sp.nsimplify(S(V.ans(4, "4-1", par)).subs(n_, N)); eb = sp.nsimplify(S(V.ans(4, "4-2", par)).subs(n_, N))
        V.check(4, f"(a) n={N}", sp.Rational(pa.numerator, pa.denominator), ea)
        V.check(4, f"(b) n={N}", sp.Rational(pb.numerator, pb.denominator), eb)


if __name__ == "__main__": V.run(globals())
