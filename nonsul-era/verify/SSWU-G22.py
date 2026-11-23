import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math
from fractions import Fraction
V = Verifier("SSWU-G22", "items/SSWU/PAST")
LOC = {"x": x, "n": n, "a": sp.Symbol("a", positive=True)}
def E(s): return sp.sympify(s, locals=LOC)
def kv(no, sub): return {k: v for k, v in (p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))}

def p01():
    a, b, c = sp.symbols("a b c")
    f = x**3 + a*x**2 + b*x + c            # (가): f/x^2 - x 가 유한 => 최고차 x^3 (조건에서 가정 아닌 차수 판별은 아래)
    sol = sp.solve([sp.limit(f/x**2 - x, x, sp.oo) + 1, f.subs(x, 0), sp.diff(f, x).subs(x, 0) - sp.Rational(1, 4)], [a, b, c], dict=True)[0]
    F = f.subs(sol)
    V.check(1, "1-1 f(x)", F, E(V.ans(1, "1")))
    d = kv(1, "2")
    cr = sorted(sp.solve(sp.diff(F, x), x)); infl = sp.solve(sp.diff(F, x, 2), x)[0]
    V.check(1, "1-2 극대 x,y", (cr[0], F.subs(x, cr[0])), (E(d["xmax"]), E(d["ymax"])))
    V.check(1, "1-2 극소 x,y", (cr[1], F.subs(x, cr[1])), (E(d["xmin"]), E(d["ymin"])))
    V.check(1, "1-2 변곡점", (infl, F.subs(x, infl)), (E(d["xinfl"]), E(d["yinfl"])))
    aa = sp.Symbol("aa")
    V.check(1, "1-2 f(a-x)+f(a+x)-2f(a) 가 a=1/3 에서 0 [증명-수치확인]", sp.expand(F.subs(x, infl - x) + F.subs(x, infl + x) - 2*F.subs(x, infl)), 0)
    g = (F.subs(x, sp.Rational(1, 3) - x) + F.subs(x, sp.Rational(1, 3) + x))/2
    g = sp.simplify(g); V.check(1, "1-3 g 상수", g, F.subs(x, sp.Rational(1, 3)))
    h = sp.expand(F - g)
    def p(k):
        k = sp.nsimplify(k); s = set()
        if k < 0: return 0
        for sg in (1, -1):
            for r in sp.Poly(h - sg*k, x).real_roots(): s.add(r)
        return len(s)
    d3 = kv(1, "3"); k1 = sp.Rational(1, 108)
    ok = (p(-1) == int(d3["pneg"]) and p(0) == int(d3["pzero"]) and p(k1/2) == int(d3["pmid"]) and p(k1) == int(d3["pone"]) and p(2*k1) == int(d3["phigh"]))
    # 구간 안에서 상수인지 격자 확인
    inmid = {p(k1*sp.Rational(i, 50)) for i in range(1, 50)}; hi = {p(k1*(1 + sp.Rational(i, 10))) for i in range(1, 60)}
    ok = ok and inmid == {int(d3["pmid"])} and hi == {int(d3["phigh"])}
    V.report(1, ok, f"1-3 p(k) 구간별 개수 {p(-1)},{p(0)},{inmid},{p(k1)},{hi}")
    V.check(1, "1-3 불연속점", [0, k1], [E(t) for t in d3["disc"].split(",")])

def p02():
    A = sp.Symbol("A")
    # (1) f=1+x, g=11-6x 는 f(0)=1,f(1)=2,g(1)=5,g(2)=-1 만족
    f1 = 1 + x; g1 = 11 - 6*x
    assert [f1.subs(x, 0), f1.subs(x, 1), g1.subs(x, 1), g1.subs(x, 2)] == [1, 2, 5, -1]
    H1 = sp.integrate(sp.diff(f1, x)*g1.subs(x, 2 - x), (x, 0, 1)) - sp.integrate(sp.diff(g1, x)*f1.subs(x, 2 - x), (x, 1, 2))
    V.check(2, "2-1 H(1)", H1, E(V.ans(2, "1")))
    def H(nn, av):
        f = av**x; g = sp.sin(sp.pi*x/4)
        return sp.integrate(sp.diff(f, x)*g.subs(x, 2*nn - x), (x, 0, nn)) - sp.integrate(sp.diff(g, x)*f.subs(x, 2*nn - x), (x, nn, 2*nn))
    form = E(V.ans(2, "2")); a = LOC["a"]
    bad = [(nn, av) for nn in range(1, 7) for av in (2, sp.Rational(1, 3), 3) if abs(complex(sp.N(H(nn, av) - form.subs({n: nn, a: av}), 20))) > 1e-9]
    V.proof(2, "2-2 H(n) 식 n=1..6, a=2,1/3,3", bad)
    av = 1/sp.sqrt(2)
    import mpmath as mp
    mp.mp.dps = 30; A_ = 1/mp.sqrt(2)
    def Hn(nn):
        return mp.quad(lambda t: mp.log(A_)*A_**t*mp.sin(mp.pi*(2*nn - t)/4), [0, nn]) - mp.quad(lambda t: (mp.pi/4)*mp.cos(mp.pi*t/4)*A_**(2*nn - t), [nn, 2*nn])
    tot = sum(Hn(4*m - 3) + Hn(4*m - 1) for m in range(1, 31))
    V.check(2, "2-3 급수합(수치적분, 30항)", sp.Float(str(tot), 25), sp.N(E(V.ans(2, "3")), 25), 1e-9)

def p03():
    th = sp.pi/3
    P1 = (sp.sin(th), 1 - sp.cos(th))
    m = sp.tan(th); tang = m*x + (P1[1] - m*P1[0])
    a_, b_, c_ = sp.symbols("a_ b_ c_"); f = a_*x**2 + b_*x + c_
    P2 = (-P1[0], P1[1])
    mm2 = -m
    eqs = [f.subs(x, P1[0]) - P1[1], f.subs(x, P2[0]) - P2[1], sp.diff(f, x).subs(x, P1[0]) - m, sp.diff(f, x).subs(x, P2[0]) - mm2]
    s = sp.solve(eqs, [a_, b_, c_], dict=True)[0]; F = f.subs(s)
    d = kv(3, "1")
    V.check(3, "3-1 접선", tang, E(d["tangent"])); V.check(3, "3-1 f", F, E(d["f"]))
    # 원 아래쪽 호 y=1-sqrt(1-x^2)
    area = sp.integrate(1 - sp.sqrt(1 - x**2) - F, (x, -P1[0], P1[0]))
    V.check(3, "3-2 영역 넓이(원 아래호-포물선)", area, E(V.ans(3, "2")))
    c = sp.Symbol("c", positive=True)
    # 접점 P1 의 접선 L: y = tan(th) x ... 일반 θ: 접선과 y축 교점 Q=(0,1-1/c)
    def radii(cs):
        s_ = sp.sqrt(1 - cs**2); Q = 1 - 1/cs
        # 접선 방향: 기울기 tanθ -> 직선 s x - cs y + (cs*Q)=0 (x=0,y=Q 지남)
        res = []
        for y0 in ([-sp.Symbol("r", positive=True)], [2 + sp.Symbol("r", positive=True)]):
            r = sp.Symbol("r", positive=True)
            dist = sp.Abs(-cs*(y0[0] - Q))/1  # 점 (0,y0) 에서 직선 s x - cs y + cs Q = 0 까지 거리 = cs|y0-Q| (법선 길이 1)
            sol = sp.solve(sp.Eq(dist, r), r)
            res.append(sol[0])
        return res
    r1, r2 = radii(c)
    sol = sp.solve(sp.Eq(r2/r1, 4), c)
    sol = [t for t in sol if 0 < t < 1]
    V.check(3, "3-3 cosθ", sol[0], E(V.ans(3, "3")))
    V.check(3, "3-3 r1 (교점이 원점)", sp.simplify(r1.subs(c, sp.Rational(1, 3))), sp.Rational(1, 2))

def p04():
    from collections import Counter
    def paths(start, steps, d):
        out = [start]
        for _ in range(steps):
            out = [(px + dx, py + dy) for (px, py) in out for dx, dy in ((d, 0), (0, d))]
        return Counter(out)
    S4 = paths((0, 0), 4, 1); T4 = paths((4, 4), 4, -1)
    meet = sum(S4[k]*T4[k] for k in S4)
    V.check(4, "4-1 만날 확률", Fraction(meet, 2**8), Fraction(*map(int, V.ans(4, "1").split("/"))))
    S3 = paths((0, 0), 3, 1); T3 = paths((4, 4), 3, -1)
    dist = Counter()
    for p1, c1 in S3.items():
        for p2, c2 in T3.items(): dist[abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])] += c1*c2
    tab = {int(a): Fraction(*map(int, b.split("/"))) for a, b in (t.split(":") for t in V.ans(4, "2").replace(" ", "").split(";"))}
    V.report(4, all(Fraction(dist[k], 64) == tab[k] for k in tab) and set(dist) == set(tab), f"4-2 분포 {dict(dist)}/64")
    Ex = sum(k*Fraction(v, 64) for k, v in dist.items()); Vx = sum((k - Ex)**2*Fraction(v, 64) for k, v in dist.items())
    d = kv(4, "3")
    V.check(4, "4-3 E", Ex, E(d["E"])); V.check(4, "4-3 σ", sp.sqrt(sp.Rational(Vx.numerator, Vx.denominator)), E(d["sd"]))

if __name__ == "__main__": V.run(globals())
