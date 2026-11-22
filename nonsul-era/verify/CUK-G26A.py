#!/usr/bin/env python3
"""CUK-G26A (가톨릭대 2026 수시 논술 자연공학·간호) 문항 1~3 검산. 정답은 V.ans() 로 YAML 에서 읽고 문제 조건에서 독립 재계산한다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G26A", "items/CUK/PAST")


def p01():
    X, m = sp.symbols("X m")
    f = X**3 - 6*X**2
    g = sp.Rational(3, 2)*X - sp.Rational(11, 2)
    # 논제 1: f-g=0 의 근 중 유리수가 아닌 두 근
    rts = sp.roots(sp.Poly(f - g, X))
    irr = [r for r in rts if not r.is_rational]
    assert len(irr) == 2 and len(rts) == 3
    V.check(1, "논제1 L", sp.nsimplify(sp.simplify(irr[0]**3 + irr[1]**3)), V.ans(1, "1"))
    # 논제 2: f(x)=m(x-1)-4 가 서로 다른 실근 정확히 2개인 m (판별식=0 후 실근 개수 직접 확인)
    c = sp.Poly(f - m*X + m + 4, X)
    disc = sp.discriminant(c.as_expr(), X)
    good = []
    for mv in sp.solve(disc, m):
        mv = sp.nsimplify(mv)
        if not mv.is_real:
            continue
        pl = sp.Poly(f - mv*X + mv + 4, X)
        nreal = len(sp.real_roots(pl, multiple=False))
        if nreal == 2:
            good.append(mv)
    V.check(1, "논제2 m 개수", len(good), 1)
    V.check(1, "논제2 m", good[0], V.ans(1, "2"))


def p02():
    x0 = sp.symbols("x0")
    a, b, c, d = sp.symbols("a b c d")
    f = lambda y: a*y**3 + b*y**2 + c*y + d   # 임의 다항함수(3차)로 조건에서 계수 제한을 구함
    eps = sp.Rational(1, 10**9)
    def piece_g(s, _side=None):  # g 의 우극한을 취할 때 점 s 근방에서 쓰이는 식 (s 는 경계점이 아닌 표본점)
        return f if 0 <= s < 3 else (lambda y: 3*y)
    def hlim(x, side):  # x 의 좌(-1)·우(+1) 근방에서 h(x)=g(x+)·g((x+3)+) 의 식을 x 에 대입한 극한
        s = x + side*eps
        return piece_g(s)(x) * piece_g(s + 3)(x + 3)
    eqs = [sp.expand(hlim(p, +1) - hlim(p, -1)) for p in (-3, 0, 3)]
    # 논제 1: 연속 조건 → f(0), f(3)
    ls = list(sp.linsolve(eqs, [d, a]))
    assert len(ls) == 1
    s0 = dict(zip([d, a], ls[0]))
    assert s0[d] == 0
    f0 = sp.simplify(f(0).subs(s0)); f3 = sp.simplify(f(3).subs(s0))
    V.check(2, "논제1 f(0)+f(3)", f0 + f3, V.ans(2, "1"))
    # 논제 2: 구간별 h 를 직접 적분 (-5,-3),(-3,0),(0,3),(3,5)
    xs = sp.symbols("xs")
    def hx(lo):  # 구간 (lo,lo+) 내부 점에서의 h 식
        t = lo + sp.Rational(1, 7)
        return sp.expand(piece_g(t)(xs) * piece_g(t + 3)(xs + 3))
    tot = sum(sp.integrate(hx(lo), (xs, lo, hi)) for lo, hi in ((-5, -3), (-3, 0), (0, 3), (3, 5)))
    tot = sp.expand(tot.subs(s0))
    cond = sp.solve(sp.Eq(tot, 600), c)  # 적분 조건으로 c 결정
    res = []
    for cv in cond:
        val = sp.integrate(xs*f(xs).subs(s0).subs(c, cv), (xs, 0, 3))
        res.append(sp.simplify(val))
    # 남은 자유 계수 a? (d,a 는 이미 결정) → b 와 무관해야 함
    V.check(2, "논제2 적분값", sp.simplify(res[0]), V.ans(2, "2"))
    V.check(2, "논제2 b 무관", sp.simplify(sp.diff(res[0], b)), 0)


def p03():
    mp.mp.dps = 30
    def tri(B, C):
        A = mp.pi - B - C
        return A, B, C
    def sides(A, B, C):  # 사인법칙으로 변 길이(비율)
        return mp.sin(A), mp.sin(B), mp.sin(C)   # (BC, CA, AB) 비율
    def circ_R(P, Q, Rr):  # 외접원 반지름
        a = mp.norm(Q - Rr); b = mp.norm(P - Rr); c_ = mp.norm(P - Q)
        ar = abs((Q[0]-P[0])*(Rr[1]-P[1]) - (Q[1]-P[1])*(Rr[0]-P[0]))/2
        return a*b*c_/(4*ar)
    def build(A, B, C, scale):  # BC 를 x축에 두는 좌표
        a, b, c_ = [s*scale for s in sides(A, B, C)]
        Bp = mp.matrix([0, 0]); Cp = mp.matrix([a, 0])
        Ap = mp.matrix([c_*mp.cos(B), c_*mp.sin(B)])
        return Ap, Bp, Cp, a, b, c_
    def valid(B, C):  # 조건 (나): sin2B=sin2C 또는 cos2B=cos2C
        return abs(mp.sin(2*B) - mp.sin(2*C)) < 1e-20 or abs(mp.cos(2*B) - mp.cos(2*C)) < 1e-20
    def cases(ratio=None, cosB=None):
        """직각(A=π/2) / 이등변(B=C) 두 유형 후보를 각도로 열거하고 조건 (나)를 직접 확인한 뒤 BC 를 모은다"""
        out = []
        # 유형1 A=π/2: B 가 자유, C=π/2-B.  유형2 B=C: B 자유.
        import numpy as np
        for typ in (1, 2):
            if cosB is not None:
                Bs = [mp.acos(cosB)]
            else:
                f = (lambda B: build(*tri(B, mp.pi/2-B), 1)) if typ == 1 else (lambda B: build(*tri(B, B), 1))
                def S1S2(B):
                    C = mp.pi/2 - B if typ == 1 else B
                    A_, B_, C_ = tri(B, C)
                    Ap, Bp, Cp, *_ = build(A_, B_, C_, 1)
                    P = (Ap + Bp)/2
                    return (circ_R(Ap, P, Cp)/circ_R(Bp, P, Cp))**2
                Bs = []
                lo, hi = (mp.mpf('1e-6'), mp.pi/2 - mp.mpf('1e-6')) if typ == 1 else (mp.mpf('1e-6'), mp.pi/2 - mp.mpf('1e-6'))
                grid = [lo + (hi - lo)*i/4000 for i in range(4001)]
                vals = [S1S2(B) - ratio for B in grid]
                for i in range(4000):
                    if vals[i] == 0:
                        Bs.append(grid[i])
                    elif vals[i]*vals[i+1] < 0:
                        Bs.append(mp.findroot(lambda B: S1S2(B) - ratio, (grid[i], grid[i+1]), solver='anderson'))
            for B in Bs:
                C = mp.pi/2 - B if typ == 1 else B
                if C <= 0 or not valid(B, C):
                    continue
                A_, B_, C_ = tri(B, C)
                # 가장 짧은 변이 3 이 되도록 scale
                a, b, c_ = sides(A_, B_, C_)
                sc = 3/min(a, b, c_)
                out.append(build(A_, B_, C_, sc)[3])
        return out
    # 논제 1: cos B = 1/3
    Ms = sorted({mp.nstr(v, 12) for v in cases(cosB=mp.mpf(1)/3)})
    exp = sorted(mp.nstr(mp.mpf(float(sp.N(s, 20))), 12) for s in str(V.ans(3, "1")).strip("{}").split(","))
    V.report(3, [float(v) for v in Ms] == [float(v) for v in exp], f"논제1 M {Ms} / YAML {exp}")
    # 논제 2: S1/S2 = 1/8, 3/8, 5/8 일 때 BC 최솟값
    pe = [sp.N(sp.sympify(s.split('=')[1]), 20) for s in V.ans(3, "2").replace(" ", "").split(";")]
    for rat, e in zip((mp.mpf(1)/8, mp.mpf(3)/8, mp.mpf(5)/8), pe):
        vals = cases(ratio=rat)
        V.report(3, vals and abs(min(vals) - mp.mpf(str(e))) < 1e-8, f"논제2 S1/S2={mp.nstr(rat,4)} 최소 BC {mp.nstr(min(vals),12) if vals else None} / YAML {e}")


if __name__ == "__main__": V.run(globals())
