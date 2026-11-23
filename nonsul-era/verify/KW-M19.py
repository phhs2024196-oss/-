"""KW-M19 검산(2019학년도 모의논술 문제 1·2): 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-M19", "items/KW/PAST")
mp.mp.dps = 40
k_ = sp.Symbol("k"); th = sp.Symbol("theta", real=True); C = sp.Symbol("C")
LOC = {"k": k_, "theta": th, "x": x, "C": C, "n": sp.Symbol("n")}


def p01():
    # [1] k 음의 정수: 직접 합
    ex = sp.sympify(V.ans(1, "1"), locals=LOC)
    for kv in (-1, -2, -5, -30):
        tot = sum(2 / (mp.sqrt(n - kv + 2) + mp.sqrt(n - kv + 1)) for n in range(1, 2019))
        V.check(1, f"1 k={kv}", float(tot), float(ex.subs(k_, kv)), 1e-9)
    # [2](1) f 를 3번 합성하면 항등 → g(x)=x, 극한
    f = lambda u: 1 / (1 - u)
    ff3 = sp.simplify(f(f(f(x))))
    V.check(1, "2-1 f∘f∘f", ff3, x)
    V.check(1, "2-1 극한", sp.limit(f(x) / ff3, x, 3), sp.sympify(V.ans(1, "2-1")))
    # [2](2) n=1..60 에서 h=1-1/x 인 n 은 모두 n≡2 (mod 3)
    h = x; cex = []
    for n in range(1, 61):
        h = sp.simplify(f(h))
        if sp.simplify(h - (1 - 1 / x)) == 0 and n % 3 != 2:
            cex.append(n)
        if n % 3 == 2 and sp.simplify(h - (1 - 1 / x)) != 0:
            cex.append(n)
    V.proof(1, "2-2 h=1-1/x ⇔ n≡2 (mod 3)", cex)
    # [3] w 허근, a,b 실수
    w = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
    a, b = sp.symbols("a b", real=True)
    expr = sp.expand(a * w**52 + b * w**27 + 1)
    sol = sp.solve([sp.re(expr), sp.im(expr)], [a, b], dict=True)[0]
    V.check(1, "3 a", sol[a], sp.sympify(V.ans(1, "3", "a")))
    V.check(1, "3 b", sol[b], sp.sympify(V.ans(1, "3", "b")))
    # [4] 정수 x 전수
    fl = sp.floor
    xs = [xx for xx in range(-2000, 2001) if 2 * fl(xx**2 - 2 * xx - 3) + 1 == fl(xx**2 - 2 * xx - 5) - 1]
    V.check(1, "4 정수해", xs, [int(V.ans(1, "4"))])


def p02():
    # [1] x절편
    ex = V.ans(2, "1")
    tot = mp.mpf(0)
    for kk in range(1, 120):
        # y=log2 x + (k - log2 3): x절편은 log2 x = -(k-log2 3)
        xk = 2 ** (-(kk - mp.log(3, 2)))
        tot += xk
    V.check(2, "1 급수", float(tot), float(sp.sympify(ex)), 1e-12)
    # [2] 극소점의 tan, cos
    F = lambda u: mp.cos(u) / mp.e ** (3 * u)
    d1 = lambda u: mp.diff(F, u)
    d2 = lambda u: mp.diff(F, u, 2)
    cands = []
    grid = [mp.mpf(i) * 2 * mp.pi / 4000 for i in range(1, 4000)]
    for u0, u1 in zip(grid, grid[1:]):
        if d1(u0) * d1(u1) < 0:
            r = mp.findroot(d1, (u0 + u1) / 2)
            if d2(r) > 0:
                cands.append(r)
    V.check(2, "2 극소점 개수", len(cands), 1)
    r = cands[0]
    V.check(2, "2 tan a", float(mp.tan(r)), float(sp.sympify(V.ans(2, "2", "tan"))), 1e-9)
    V.check(2, "2 cos a", float(mp.cos(r)), float(sp.sympify(V.ans(2, "2", "cos"))), 1e-9)
    # [3](1) 부정적분: 미분하면 tan x
    Fx = sp.sympify(V.ans(2, "3-1"), locals={**LOC, "Abs": sp.Abs})
    V.check(2, "3-1 d/dx", sp.simplify(sp.diff(Fx, x).subs(x, 0.7) - sp.tan(0.7)), 0, 1e-9)
    # [3](2)
    Sn = sp.sympify(V.ans(2, "3-2"), locals=LOC)
    for nn in (1, 2, 3, 6):
        a0, a1 = (nn - mp.mpf(1) / 4) * mp.pi, (nn + mp.mpf(1) / 4) * mp.pi
        val = mp.quad(lambda u: abs(mp.mpf(1) / 3) ** nn * abs(mp.tan(u)), [a0, nn * mp.pi, a1])
        V.check(2, f"3-2 n={nn}", float(val), float(Sn.subs(LOC["n"], nn)), 1e-9)
    # [4] 원 (x+2)^2+(y-3)^2<=9, 직선 y=t(x+2)=tan(θ)(x+2)
    def geo(theta):
        t = mp.tan(theta)
        xB = -2 + 6 * t / (1 + t * t)  # 원과 직선의 두 번째 교점: (x+2)^2(1+t^2) = 6 t (x+2)
        yB = t * (xB + 2)
        A = (mp.mpf(-2), mp.mpf(0)); B = (xB, yB); O = (mp.mpf(-2), mp.mpf(3))
        ua = (A[0] - O[0], A[1] - O[1]); ub = (B[0] - O[0], B[1] - O[1])
        ang = mp.acos((ua[0] * ub[0] + ua[1] * ub[1]) / 9)
        tri = abs(ua[0] * ub[1] - ua[1] * ub[0]) / 2
        return t, xB, ang, tri
    for tv in (0.3, 0.7, 1.0, 1.2):
        t, xB, ang, tri = geo(tv)
        V.check(2, f"4-1 각 θ={tv}", float(ang), float(sp.sympify(V.ans(2, "4-1", "AOB"), locals=LOC).subs(th, tv)), 1e-9)
        V.check(2, f"4-1 부채꼴 θ={tv}", float(ang * 9 / 2), float(sp.sympify(V.ans(2, "4-1", "sector"), locals=LOC).subs(th, tv)), 1e-9)
        V.check(2, f"4-1 삼각형 θ={tv}", float(tri), float(sp.sympify(V.ans(2, "4-1", "triangle"), locals=LOC).subs(th, tv)), 1e-9)
    # [4](2) 영역 넓이를 x 적분으로 직접
    gexpr = sp.sympify(V.ans(2, "4-2"), locals={**LOC, "pi": sp.pi})
    def region(theta):
        t, xB, *_ = geo(theta)
        def h(u):
            r = mp.sqrt(max(mp.mpf(0), 9 - (u + 2) ** 2))
            low = max(t * (u + 2), 3 - r)
            return max(mp.mpf(0), (3 + r) - low)
        return mp.quad(h, sorted([-5, -2, xB, 1]))
    for tv in (0.3, 0.7, 1.0, 1.2):
        V.check(2, f"4-2 g θ={tv}", float(region(tv)), float(gexpr.subs(th, tv)), 1e-8)
    # [4](3) f(t)=g(arctan t) → f'(1)=g'(π/4)/sec^2(π/4)
    gnum = lambda u: region(u)
    fp1 = mp.diff(lambda tt: gnum(mp.atan(tt)), 1)
    V.check(2, "4-3 f'(1)", float(fp1), float(sp.sympify(V.ans(2, "4-3"))), 1e-5)


if __name__ == "__main__": V.run(globals())
