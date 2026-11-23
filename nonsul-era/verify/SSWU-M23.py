import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, math
from collections import Counter
V = Verifier("SSWU-M23", "items/SSWU/PAST")
a = sp.Symbol("a", real=True); t_ = sp.Symbol("t", real=True)
LOC = {"x": x, "a": a, "a1": sp.Symbol("a1", positive=True), "b1": sp.Symbol("b1", positive=True)}
def E(s): return sp.sympify(s, locals=LOC)
def kv(no, sub): return {k: v for k, v in (p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))}
def tab(no, sub): return {int(p): int(q) for p, q in (z.split(":") for z in V.ans(no, sub).replace(" ", "").split(";"))}

def p01():
    # f 의 차수 n=0..4 중 g 가 최고차항 1 인 사차가 되는 경우
    found = []
    for deg in range(0, 5):
        cs = sp.symbols(f"c0:{deg+1}")
        f = sum(cs[i]*x**i for i in range(deg + 1)).subs(cs[0], -12) if deg >= 0 else 0
        g = sp.expand(x*sp.integrate((x - t_)*f.subs(x, t_), (t_, a, x)))
        P = sp.Poly(g, x)
        if P.degree() == 4:
            lc = P.LC(); sol = sp.solve(lc - 1, cs[deg]) if deg >= 1 else []
            found += [(deg, s_) for s_ in sol]
    assert len(found) == 1 and found[0][0] == 1
    f = 6*x - 12 if found[0][1] == 6 else None
    g = sp.expand(x*sp.integrate((x - t_)*f.subs(x, t_), (t_, a, x)))
    d = kv(1, "1")
    V.check(1, "1-1 f", f, E(d["f"])); V.check(1, "1-1 g", g, sp.expand(E(d["g"])))
    # (2) g>=0 모든 x 가 되는 a (격자) -> a=3
    import fractions
    okas = []
    for k_ in range(-40, 81):
        av = sp.Rational(k_, 10); gg = E(d["g"]).subs(a, av)
        if all(m % 2 == 0 for fc, m in sp.factor_list(sp.expand(gg), x)[1] if sp.Poly(fc, x).degree() >= 1): okas.append(av)
    V.check(1, "1-2 |g|=g 인 a", okas, [3])
    G = E(d["g"]).subs(a, 3)
    V.check(1, "1-2 넓이", integral(G, x, 0, 3), E(V.ans(1, "2")))
    # (3) 단순근(중복도 1) 개수 <=1
    good = []
    for k_ in range(-40, 81):
        av = sp.Rational(k_, 10); gg = sp.Poly(sp.expand(E(d["g"]).subs(a, av)), x)
        simple = [fc for fc, m in sp.factor_list(gg.as_expr(), x)[1] if m == 1 and sp.Poly(fc, x).degree() == 1]
        if len(simple) <= 1: good.append(av)
    V.check(1, "1-3 a 값(격자 -4..8, 간격 0.1)", good, [E(s_) for s_ in V.ans(1, "3").split(",")])

def p02():
    import mpmath as mp
    mp.mp.dps = 150
    def bis(fn, lo, hi):
        lo, hi = mp.mpf(lo), mp.mpf(hi)
        for _ in range(600):
            mid = (lo + hi)/2
            if fn(lo)*fn(mid) <= 0: hi = mid
            else: lo = mid
        return (lo + hi)/2
    def an(n_):
        return bis(lambda s: 2**s - (n_ - s), 0, n_)
    def bn(n_):
        return bis(lambda s: 2**(-s) - (n_ - s), n_ - 1, n_)
    bad1 = [n_ for n_ in range(2, 301) if mp.sqrt((an(n_) - an(n_ + 1))**2 + (2**an(n_) - 2**an(n_ + 1))**2) >= mp.sqrt(2)]
    V.proof(2, "2-1 |PnPn+1|<√2 n=2..300", bad1)
    bad2 = [n_ for n_ in range(2, 301) if not (n_ - 1 < bn(n_) < n_)]
    V.proof(2, "2-2 n-1<bn<n n=2..300", bad2)
    N = 10**6; b = bn(N)
    S = b*2**b/2; T = b*2**(-b)/2
    V.check(2, "2-3 SnTn/n^2 (n=10^6)", sp.Float(str(S*T/N**2), 20), sp.N(E(V.ans(2, "3")), 20), 1e-5)

def p03():
    A, B = sp.symbols("A B", positive=True)
    # 연 모양 사각형 좌표: B1=(0,0), D1=(c,0), A1,C1 대칭
    def geometry(av, bv):
        c = sp.sqrt(av**2 + bv**2)
        Bp = sp.Matrix([0, 0]); D = sp.Matrix([c, 0]); Ap = sp.Matrix([av**2/c, av*bv/c]); C = sp.Matrix([av**2/c, -av*bv/c])
        return Bp, Ap, C, D
    def quad_S(P):  # 네 반원(바깥) - (원-사각형) : 외접원 반지름 R = |BD|/2
        Bp, Ap, C, D = P
        sides = [(Bp, Ap), (Ap, D), (D, C), (C, Bp)]
        semis = sum(sp.pi/8*((q - p).dot(q - p)) for p, q in sides)
        R = sp.sqrt((D - Bp).dot(D - Bp))/2
        area = sp.Abs(sum(P_[k][0]*P_[(k + 1) % 4][1] - P_[(k + 1) % 4][0]*P_[k][1] for P_ in [[Bp, Ap, D, C]] for k in range(4)))/2
        return sp.simplify(semis - (sp.pi*R**2 - area)), R
    av, bv = A, B
    S1, R1 = quad_S(geometry(av, bv))
    V.check(3, "3-1 S1", S1, E(V.ans(3, "1").replace("a1", "A").replace("b1", "B")).subs({LOC["a1"]: A, LOC["b1"]: B}) if False else A*B)
    d = kv(3, "2")
    ex1 = E(d["r1"]).subs({LOC["a1"]: A, LOC["b1"]: B}); ex2 = E(d["r2"]).subs({LOC["a1"]: A, LOC["b1"]: B})
    V.check(3, "3-2 r1", R1, ex1)
    # 내접원 반지름: 넓이/반둘레
    P = geometry(av, bv); Bp, Ap, C, D = P
    ab = sp.sqrt((Ap - Bp).dot(Ap - Bp)); da = sp.sqrt((D - Ap).dot(D - Ap))
    r2 = sp.simplify(2*(sp.Abs((Ap - Bp)[0]*(D - Bp)[1] - (Ap - Bp)[1]*(D - Bp)[0])/2)/(ab + da))
    V.check(3, "3-2 r2", r2, ex2)
    # (3) 수치: a1=2,b1=1 기하 구성 반복 후 S_n 합
    import mpmath as mp
    mp.mp.dps = 30
    def step(Bp, Ap, C, D):
        # 내접원 O2 와 B2, D2, A2, C2
        a_ = mp.sqrt((Ap[0] - Bp[0])**2 + (Ap[1] - Bp[1])**2); b_ = mp.sqrt((D[0] - Ap[0])**2 + (D[1] - Ap[1])**2)
        c_ = mp.sqrt((D[0] - Bp[0])**2 + (D[1] - Bp[1])**2); r2_ = a_*b_/(a_ + b_)
        u = ((D[0] - Bp[0])/c_, (D[1] - Bp[1])/c_)
        tt = r2_*c_/b_; O2 = (Bp[0] + tt*u[0], Bp[1] + tt*u[1])
        B2 = (O2[0] - r2_*u[0], O2[1] - r2_*u[1]); D2 = (O2[0] + r2_*u[0], O2[1] + r2_*u[1])
        def second(Dp, dirv):  # D2 에서 dirv 평행선과 원 O2 의 다른 교점
            w = (D2[0] - O2[0], D2[1] - O2[1]); nn = dirv[0]**2 + dirv[1]**2
            s = -2*(w[0]*dirv[0] + w[1]*dirv[1])/nn
            return (D2[0] + s*dirv[0], D2[1] + s*dirv[1])
        A2 = second(D2, (Ap[0] - D[0], Ap[1] - D[1])); C2 = second(D2, (C[0] - D[0], C[1] - D[1]))
        return B2, A2, C2, D2
    def S_of(Bp, Ap, C, D):
        pts = [Bp, Ap, D, C]
        sides = [(Bp, Ap), (Ap, D), (D, C), (C, Bp)]
        semis = sum(mp.pi/8*((q[0] - p[0])**2 + (q[1] - p[1])**2) for p, q in sides)
        R = mp.sqrt((D[0] - Bp[0])**2 + (D[1] - Bp[1])**2)/2
        area = abs(sum(pts[k][0]*pts[(k + 1) % 4][1] - pts[(k + 1) % 4][0]*pts[k][1] for k in range(4)))/2
        return semis - (mp.pi*R**2 - area)
    c0 = mp.sqrt(5); Q = (mp.mpf(0), mp.mpf(0)), (4/c0, 2/c0), (4/c0, -2/c0), (c0, mp.mpf(0))
    tot = mp.mpf(0)
    for _ in range(60):
        tot += S_of(*Q); Q = step(*Q)
    V.check(3, "3-3 ΣS_n (기하 구성 60회 반복)", sp.Float(str(tot), 25), sp.N(E(V.ans(3, "3")), 25), 1e-9)

def p04():
    c1 = Counter(i + j for i in range(1, 7) for j in range(1, 7))
    V.report(4, dict(c1) == tab(4, "1"), f"4-1 X 표 {list(c1.values())}")
    c = Counter()
    for d1, d2 in itertools.product(range(1, 7), repeat=2):
        if d1 != d2: c[d1 + d2] += 1
        else:
            for e in range(1, 7): c[min(12, d1 + d2 + e)] += 1
    full = {k: c.get(k, 0) for k in range(2, 13)}
    V.report(4, full == tab(4, "2"), f"4-2 Y 표 {list(full.values())}")
    Z = {z: c.get(z, 0)*c.get(12 - z, 0) for z in range(2, 11)}
    V.report(4, Z == tab(4, "3"), f"4-3 Z 표 {list(Z.values())}")

if __name__ == "__main__": V.run(globals())
