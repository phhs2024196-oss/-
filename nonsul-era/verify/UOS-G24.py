import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math
V = Verifier("UOS-G24", "items/UOS/PAST")

def p01():
    # 교점: x^4-2x^2+1 = -x^2+x+2
    pts = sp.solve(sp.Eq(x**4 - 2*x**2 + 1, -x**2 + x + 2), x)
    real = sorted([sp.N(r) for r in pts if abs(sp.im(sp.N(r))) < 1e-12], key=lambda r: float(sp.re(r)))
    xs = [float(sp.re(r)) for r in real]
    V.check(1, "실근 2개", len(xs), 2)
    xp, xq = xs[0], xs[1]
    V.check(1, "P x좌표 -1", xp, -1, 1e-9)
    f = lambda t: -t**2 + t + 2
    P = (xp, f(xp)); Q = (xq, f(xq)); A = (0, -1)
    dot = (Q[0] - P[0])*(A[0] - P[0]) + (Q[1] - P[1])*(A[1] - P[1])
    V.check(1, "내적 = a^2-1", dot, xq**2 - 1, 1e-9)
    V.check(1, "k", math.floor(dot), V.ans(1, "1-1"))
    V.check(1, "x^3-x^2-1 의 유일한 실근", len([r for r in sp.Poly(x**3 - x**2 - 1, x).nroots() if abs(sp.im(r)) < 1e-12]), 1)
    V.report(1, math.sqrt(2) < xq < math.sqrt(3), "√2 < a < √3")

def p02():
    from math import comb
    # 직접 열거: 사과 (x1,y1,z1), 배 (x2,y2,z2) 합 11
    tot = 0; cnt = 0
    for x1, y1, z1 in itertools.product(range(12), repeat=3):
        if x1 + y1 + z1 > 11: continue
        r = 11 - (x1 + y1 + z1)
        for x2 in range(r + 1):
            for y2 in range(r - x2 + 1):
                z2 = r - x2 - y2
                tot += 1
                if x1 + x2 <= y1 + y2 <= z1 + z2: cnt += 1
    V.check(2, "(a)", tot, V.ans(2, "2-1"))
    V.check(2, "(b)", cnt, V.ans(2, "2-2"))
    V.check(2, "6H11", comb(16, 5), V.ans(2, "2-1"))

def p03():
    def A_n(n):
        c = 0
        for a in range(-4, 5):
            for b in range(-60, 260):
                ok = all(-abs(xx - 2) <= a*xx + b <= abs(xx - 2*n) + 5 for xx in (2, 2*n, 0, -10**4, 10**4, 2*n - 1, 2*n + 1, 1, 3))
                c += ok
        return c
    V.check(3, "A_1..A_4", [A_n(n) for n in (1, 2, 3, 4)], [6 + (2*1 + 4) + 6, 6 + 8 + 4, 6 + 10 + 2, 6 + 12 + 0])
    V.check(3, "합", sum(A_n(n) for n in range(1, 101)), V.ans(3, "3-1"))

def p04():
    import numpy as np
    def bc(n):
        l = 2*np.sin(np.pi/(2*n)); k = np.arange(1, 2*n + 1)
        tk = l/np.sqrt(2 - k/(2*n))
        b = tk.sum(); c = tk[:n].sum()
        d = n*l - (2*n*l/b)*c
        return b, c, d
    b, c, d = bc(200000)
    V.check(4, "b_n 극한", b, sp.N(sp.sympify(V.ans(4, "4-1"))), 1e-3)
    V.check(4, "c_n 극한", c, sp.N(2*(2*sp.sqrt(2) - sp.sqrt(6))*sp.pi), 1e-3)
    V.check(4, "d_n 극한", d, sp.N(sp.sympify(V.ans(4, "4-2"))), 1e-3)
    V.check(4, "대수 정리", sp.pi - 4*(2*sp.sqrt(2) - sp.sqrt(6))*sp.pi**2/(4*(sp.sqrt(2) - 1)*sp.pi) - (sp.sqrt(6) + 2*sp.sqrt(3) - 2*sp.sqrt(2) - 3)*sp.pi, 0)
    V.check(4, "∫0^1 (2-x)^(-1/2)", sp.integrate(1/sp.sqrt(2 - x), (x, 0, 1)), 2*(sp.sqrt(2) - 1))
    V.check(4, "∫0^1 (2-x/2)^(-1/2)", sp.integrate(1/sp.sqrt(2 - x/2), (x, 0, 1)), 2*(2*sp.sqrt(2) - sp.sqrt(6)))

if __name__ == "__main__": V.run(globals())
