import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("DDW-M25", "items/DDW/PAST")
tt = sp.symbols("tt", real=True)
c2 = sp.cbrt(2)


def p01():
    """공차 d(|d|<=3 정수) 전수 조사: sum(|S_n|-S_n, n=1..6)=34 인 d 와 일반항."""
    sols = []
    for d in range(-3, 4):
        S = [sp.Rational(m, 2)*(2*3 + (m - 1)*d) for m in range(1, 7)]
        if sum(abs(s) - s for s in S) == 34:
            sols.append(d)
    V.check(1, "조건 만족 공차 유일", sols, [-2])
    an = 3 + (n - 1)*sols[0]
    V.check(1, "일반항", an, sp.sympify(V.ans(1, "1"), locals={"n": n}))
    # 해설 중간값: d=-3 일 때 S4..S6 합 -48, d=-2 일 때 S5+S6=-17, d=-1 은 음수항 없음
    V.check(1, "d=-3 음인 S 합", sum(sp.Rational(m, 2)*(6 - 3*(m - 1)) for m in (4, 5, 6)), -48)
    V.check(1, "d=-2 음인 S 합", sum(m*(4 - m) for m in (5, 6)), -17)
    V.check(1, "d=-1 S6", sp.Rational(6, 2)*(6 - 5), 3)


def I_of(f, t0):
    x_ = x
    ft = f.subs(x, t0)
    return (sp.integrate(x_**2*(ft - f), (x_, 0, t0)) + sp.integrate(x_**2*(f - ft), (x_, t0, c2)))


def p02():
    """여러 증가함수 f 로 I(t) 를 정확히 구해 I'(t)=(2/3)(t^3-1)f'(t) 와 최소점 t=1 확인."""
    T = sp.symbols("T", positive=True)
    for f in (x, x**3 + x, 2*x + x**2, sp.exp(x)):
        I = sp.simplify(I_of(f, T))
        formula = sp.Rational(2, 3)*(T**3 - 1)*f.subs(x, T) - sp.integrate(x**2*f, (x, 0, T)) + sp.integrate(x**2*f, (x, T, c2))
        V.check(2, f"I(t)=(2) 꼴 f={f}", sp.simplify(I - formula), 0)
        dI = sp.diff(I, T)
        V.check(2, f"I'(t)=(3) 꼴 f={f}", sp.simplify(dI - sp.Rational(2, 3)*(T**3 - 1)*sp.diff(f, x).subs(x, T)), 0)
        crit = [r for r in sp.solve(sp.Eq(dI, 0), T) if r.is_real and 0 < r < c2]
        V.check(2, f"임계점 f={f}", crit, [S(V.ans(2, "1"))])
        # 수치 최소: 격자에서 최소 t
        g = sp.lambdify(T, I, "mpmath")
        import mpmath as mp
        ts = [mp.mpf(i)/200*mp.mpf(2)**(mp.mpf(1)/3) for i in range(201)]
        best = min(ts, key=g)
        V.check(2, f"격자 최소 근방 f={f}", abs(float(best) - 1.0) < 0.01, True)


def p03():
    F = sp.Function("F")
    f = 2*x - 2 + 2*sp.exp(-x)
    V.check(3, "3-1 답", f, sp.sympify(V.ans(3, "3-1"), locals={"x": x}))
    # 주어진 관계식 f(x)=x^2-∫_0^x (x-t) f'(t) dt 에 대입
    tau = sp.symbols("tau", real=True)
    rhs = x**2 - sp.integrate((x - tau)*sp.diff(f.subs(x, tau), tau), (tau, 0, x))
    V.check(3, "3-1 관계식 만족", sp.simplify(f - rhs), 0)
    V.check(3, "3-1 f(0)", f.subs(x, 0), 0)
    V.check(3, "3-1 (e^x f)'=2x e^x", sp.simplify(sp.diff(sp.exp(x)*f, x) - 2*x*sp.exp(x)), 0)
    val = sp.integrate(2*x*f, (x, -1, 1))
    V.check(3, "3-2 값", val, V.ans(3, "3-2"))
    G = 4*(-x*sp.exp(-x) - sp.exp(-x) + x**3/3 - x**2/2)
    V.check(3, "3-2 원시함수 미분", sp.simplify(sp.diff(G, x) - 2*x*f), 0)

if __name__ == "__main__": V.run(globals())
