import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import math
V = Verifier("KNG-M26", "items/KNG/PAST")
h, c = sp.symbols("h c")
from sympy.parsing.sympy_parser import parse_expr
P = lambda q: parse_expr(q, local_dict={"x": x})

def p01():
    a1 = sp.Integer(4) ** sp.Rational(1, 3)       # a>1: x=1 에서 최대, a^3-1=3
    V.check(1, "a>1 해", a1, S(V.ans(1, "1", "a"))); V.report(1, a1 > 1, "a>1 만족")
    a2 = 4                                         # 0<a<1: x=-1 에서 최대, a-1=3
    V.report(1, not (0 < a2 < 1), "0<a<1 후보 a=4 는 조건 위반 -> 해 없음")
    xs = [-1 + i * 2 / 2000 for i in range(2001)]
    V.check(1, "수치 최댓값", round(max(float(a1) ** (xv + 2) - 1 for xv in xs), 9), 3)
    bad = [av for av in [0.001 + 0.001 * i for i in range(998)] if abs(max(av ** (xv + 2) - 1 for xv in xs[::100]) - 3) < 1e-6]
    V.report(1, not bad, f"0<a<1 에서 최댓값 3 인 a 없음 {bad[:1]}")

def p02():
    t = sp.symbols("t", real=True)
    # 근과 계수: alpha+beta=-6 tan(pi/2-th)=-6/t, alpha*beta=tan(pi+th)=t
    s_, p_ = -6 / t, t
    sol = sp.solve(sp.Eq(9 * p_ + 3 * s_ + 1, 1), t)
    V.report(2, len(sol) == 2, f"방정식 해 {sol}")
    V.check(2, "tan^2", sp.simplify(sol[0] ** 2), S(V.ans(2, "1", "tan2")))
    for th in (math.atan(math.sqrt(2)), math.pi + math.atan(math.sqrt(2)), math.atan(-math.sqrt(2))):
        # 수치 대입: 근 두 개의 곱 확인 (tan(pi/2-th)=1/tan th)
        b = 6 / math.tan(th); cc = math.tan(math.pi + th)
        disc = b * b - 4 * cc
        if disc >= 0 and math.tan(th) > 0:
            r1 = (-b + math.sqrt(disc)) / 2; r2 = (-b - math.sqrt(disc)) / 2
            V.report(2, abs((3 * r1 + 1) * (3 * r2 + 1) - 1) < 1e-9, f"th={th:.3f} 수치")
    # 정정 전 식(tan(pi-th)): 같은 방식이면 tan^2=-2 로 실수해 없음
    tt = sp.symbols("tt", real=True)
    V.report(2, sp.solve(sp.Eq(9 * (-tt) + 3 * (-6 / tt) + 1, 1), tt) == [], "정정 전 문제식은 실수 tanθ 해가 없다(정정 필요성)")

def p03():
    f = lambda u: u ** 2 + 2 * u + 3
    fp = sp.limit((f(x + h) - f(x)) / h, h, 0)
    V.check(3, "f'", fp, P(V.ans(3, "1", "fprime")))
    slope = sp.Rational(f(1) - f(-2), 3); V.check(3, "평균변화율", slope, 1)
    cs = sp.solve(sp.Eq(2 * c + 2, slope), c)
    V.check(3, "c", cs[0], S(V.ans(3, "1", "c"))); V.report(3, -2 < cs[0] < 1, "c in (-2,1)")

def p04():
    a = sp.symbols("a", positive=True)
    f = a - x ** 3
    A = sp.integrate(f, (x, 0, 1)); V.check(4, "넓이", A, a - sp.Rational(1, 4))
    sol = sp.solve(sp.Eq(A / 2, 1), a)[0]
    V.check(4, "a", sol, S(V.ans(4, "1", "a"))); V.report(4, sol > 2, "a>2")
    V.report(4, f.subs(x, 1).subs(a, sol) >= 1, "[0,1]에서 f>=1 이므로 y=1 아래는 1x1 정사각형")

if __name__ == "__main__": V.run(globals())
