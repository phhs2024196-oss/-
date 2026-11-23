#!/usr/bin/env python3
"""서경대 2024학년도 수시 논술 A형 수학 10~15번 — sympy 검산(기대값은 YAML 정답에서 읽음)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("SKU-G24A", "items/SKU/PAST")
h = sp.symbols("h", real=True)


def p10():
    roots = sp.solveset(sp.sin(x) - sp.cos(sp.pi / 5), x, sp.Interval(0, 2 * sp.pi))
    al, be = sorted(roots, key=lambda r: float(r))
    # 두 근 사이(α<x<β)에서 sin x > cos(π/5)임을 확인
    mid = (al + be) / 2
    V.report(10, bool(sp.sin(mid) - sp.cos(sp.pi / 5) > 0), "α<x<β 에서 부등식 성립(중점)")
    V.check(10, "β/α", sp.simplify(be / al), V.ans(10, "1"))


def p11():
    # a^b=b^a, a^b+b^a=32 를 만족하는 (a,b) 를 1<a<b 에서 탐색: 정수 후보 + 절편 직접 계산
    a_, b_ = sp.Integer(2), sp.Integer(4)
    icpt = lambda base: sp.log(b_, base) * 0 + (sp.log(a_, base) - a_ * (sp.log(b_, base) - sp.log(a_, base)) / (b_ - a_))
    V.check(11, "두 y절편 같음(a=2,b=4)", sp.simplify(icpt(3) - icpt(9)), 0)
    f = lambda t: a_ ** (b_ * t) + b_ ** (a_ * t)
    V.check(11, "f(1)=32", f(1), 32)
    V.check(11, "f(2)", f(2), V.ans(11, "1"))
    # 유일성: ln a/a = ln b/b 곡선 위에서 a^b+b^a 가 단조 증가함을 수치 확인
    import math
    vals = []
    for a in [1.2, 1.5, 1.9, 2.0, 2.3, 2.6]:
        lo, hi = math.e, 1e4          # ln b / b 는 b>e 에서 감소 → 이분법
        for _ in range(200):
            m = (lo + hi) / 2
            if math.log(m) / m > math.log(a) / a:
                lo = m
            else:
                hi = m
        vals.append(a ** lo + lo ** a)
    V.report(11, all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)) and abs(vals[3] - 32) < 1e-6, "a^b+b^a 단조감소(유일성) 수치확인")


def p12():
    a = sp.symbols("a", positive=True)
    seq = [a, sp.Integer(-2)]
    for _ in range(20):
        seq.append(sp.expand(seq[-1] - seq[-2]))
    total = sp.expand(sum(sp.Abs(s) for s in seq))
    V.check(12, "합=15a+28", total, 15 * a + 28)
    V.check(12, "a", sp.solve(sp.Eq(total, 103), a)[0], V.ans(12, "1"))


def p13():
    f1 = sp.symbols("f1", positive=True)
    # 좌: |f g| → 6 ; 우: |f g| = f^2*(3x-1) → 2 f1^2 ; 연속이므로 같다
    f1v = sp.solve(sp.Eq(2 * f1**2, 6), f1)[0]
    # f ≡ f1v 상수로 모델: g = (x^2-7)/f1v (x<1), (3x-1) f1v (x>1)
    gl = sp.limit((x**2 - 7) / f1v, x, 1, "-")
    gr = sp.limit((3 * x - 1) * f1v, x, 1, "+")
    V.check(13, "|fg| 연속", sp.Abs(f1v * gl) - sp.Abs(f1v * gr), 0)
    V.report(13, gr > gl, "우극한>좌극한")
    V.check(13, "f(1)*(우-좌)", sp.simplify(f1v * (gr - gl)), V.ans(13, "1"))


def p14():
    cs = sp.symbols("c1:5", real=True)
    for deg in (2, 3, 4, 5):
        c = sp.symbols(f"c1:{deg}", real=True)
        f = lambda z: z**deg + sum(c[i] * z ** (i + 1) for i in range(deg - 1))
        num = (f(x + h**2 + h + 1) - f(x + 1)) * f(h / 3)
        lim = sp.limit(sp.cancel(sp.expand(num) / h**2), h, 0)
        eqs = sp.Poly(sp.expand(lim - (x**2 + sp.Rational(4, 3) * x + sp.Rational(2, 3))), x).all_coeffs()
        sol = sp.solve(eqs, list(c), dict=True)
        if deg == 3:
            V.report(14, len(sol) == 1, f"3차 해 {sol}")
            V.check(14, "f(3)", f(3).subs(sol[0]), V.ans(14, "1"))
        else:
            V.report(14, not sol, f"{deg}차 해 없음 {sol}")


def p15():
    C = sp.symbols("C", real=True)
    fp = 3 * x**2 - 2 * x - 1
    f = sp.integrate(fp, x) + C
    # (나)
    Cv = sp.solve(sp.Eq(sp.integrate(f, (x, 0, 1)) + sp.Rational(11, 12), sp.integrate(f, (x, 0, 2))), C)
    V.report(15, Cv == [1], f"C={Cv}")
    f = f.subs(C, Cv[0])
    # (가): 양변 미분한 f'(f'-(3x^2-2x-1))=0 확인
    V.check(15, "(가) 성립", sp.simplify(sp.diff(f, x) * (sp.diff(f, x) - fp)), 0)
    # 넓이: [-1,1] 에서 f 의 부호 확인 후 ∫|f|
    xs = [sp.Rational(i, 50) for i in range(-50, 51)]
    V.report(15, all(f.subs(x, v) >= 0 for v in xs), "[-1,1] 에서 f>=0")
    V.check(15, "넓이", sp.integrate(f, (x, -1, 1)), V.ans(15, "1"))


if __name__ == "__main__":
    V.run(globals())
