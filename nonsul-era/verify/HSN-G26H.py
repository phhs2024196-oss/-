import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("HSN-G26H", "items/HSN/PAST")

def vals(no):
    return [S(v) for v in V.ans(no, "1").split(",")]

def p11():
    sols = [r for r in sp.solve(sp.Eq(sp.log(5*x+3, 3), 2 + sp.log(x-1, 3)), x) if r > 1]
    # 진수 조건 x>1 을 만족하는 해만
    c = [r for r in sp.solve(sp.Eq(5*x+3, 9*(x-1)), x) if r > 1]
    V.check(11, "진수조건 만족 해", c[0], vals(11)[0])
    V.check(11, "수치대입", sp.N(sp.log(5*c[0]+3, 3) - 2 - sp.log(c[0]-1, 3)), 0)

def p12():
    n_ = sp.symbols("n", integer=True)
    a = 10 + (n_-1)*sp.Rational(-2, 3)
    best = max(m for m in range(1, 200) if (10 - sp.Rational(2, 3)*(m-1)) > 0)
    V.check(12, "최대 n", best, vals(12)[0])

def p13():
    a = sp.symbols("a")
    f = -x**3 + a*x**2 + 6*x - 3
    av = sp.solve(sp.diff(f, x).subs(x, 2), a)[0]
    g = f.subs(a, av)
    crit = sp.solve(sp.diff(g, x), x)
    mins = [g.subs(x, c) for c in crit if sp.diff(g, x, 2).subs(x, c) > 0]
    V.check(13, "극솟값", mins[0], vals(13)[0])

def p14():
    a = sp.symbols("a", real=True)
    left = (3**2 - 3 - 1)**2
    right = (3 + a - 1)**2
    r = sp.solve(sp.Eq(left, right), a)
    V.check(14, "a 해집합", set(r), set(vals(14)))

def p15():
    a = sp.symbols("a", positive=True)
    k = sp.symbols("k", positive=True)
    # 점 C=(3+k, 3/2): a sin(pi k/3)=3/2, 넓이=(3-2k)*3=6
    kv = sp.solve(sp.Eq((3-2*k)*3, 6), k)[0]
    av = sp.solve(sp.Eq(a*sp.sin(sp.pi*kv/3), sp.Rational(3, 2)), a)[0]
    # 직접 교점(수치)으로 넓이 재계산
    import math
    A = float(av)
    pts = []
    for h in (-1.5, 1.5):
        for m_ in range(-3, 4):
            for base in (math.asin(h/A), math.pi - math.asin(h/A)):
                xx = 3 + (3/math.pi)*(base + 2*math.pi*m_)
                if -1e-12 <= xx <= 6+1e-12: pts.append((round(xx, 9), h))
    pts = sorted(set(pts), key=lambda q: (q[1], q[0]))
    assert len(pts) == 4, pts
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts   # 아래 두 점, 위 두 점
    area = abs(x2 - x1) * (y3 - y1)
    V.check(15, "a", av, vals(15)[0])
    V.check(15, "넓이 6", area, 6)

if __name__ == "__main__": V.run(globals())
