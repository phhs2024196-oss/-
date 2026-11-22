#!/usr/bin/env python3
"""CUK-M27A (가톨릭대 2027학년도 모의논술 자연공학계열·간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-M27A", "items/CUK/PAST")
mp.mp.dps = 40
X0 = sp.Symbol("x")   # YAML 의 식 문자열이 sympify 되는 기호

def p01():
    f = (x + 1) * (x**2 - 4) * sp.exp(-x)
    crit = sp.solve(sp.diff(f, x), x)                       # 임계점
    d2 = sp.diff(f, x, 2)
    maxs = [c for c in crit if d2.subs(x, c).evalf() < 0]   # 극대
    assert len(maxs) == 2
    A = sp.simplify(sp.prod([f.subs(x, c) for c in maxs]))
    V.check(1, "1-1 A", A, V.ans(1, "1"))
    b = max(maxs, key=lambda c: float(c))
    # 구간 [1,2026]: 끝점+극값 비교(수치 고정밀)
    cand = [S(1), S(2026), b]
    vals = [sp.N(f.subs(x, c), 60) for c in cand]
    m, M = min(vals), max(vals)
    V.check(1, "1-2 m", m, sp.N(S(V.ans(1, "2", "m")), 60), 1e-30)
    V.check(1, "1-2 M", M, sp.N(S(V.ans(1, "2", "M")), 60), 1e-30)
    # 구간 안 임의 격자 점이 [m,M] 안인지
    V.report(1, all(m - 1e-40 <= sp.N(f.subs(x, S(v) / 10), 30) <= M + 1e-40 for v in range(10, 400)), "격자 점 [m,M] 포함")

def p02():
    a, h, b = sp.symbols("a h b")
    # 논제 1: 2h=6 → h=3. 임의의 이차함수 f(x)=a(x-3)^2+c 중 조건(가) 만족하는 것이 존재함을 수치 확인 + 중근 경우 배제
    V.check(2, "2-1 h", S(6) / 2, V.ans(2, "1"))
    # 논제 2: g(x)=a(x-3/2)^2+b, b=-2-a/4, 조건 (나)(다)(ㄴ)을 a=1..30 에서 직접 판정
    X = sp.symbols("X")
    sols = []
    for a_ in range(2, 31):
        g = a_ * (X - sp.Rational(3, 2))**2 - 2 - sp.Rational(a_, 4)
        assert g.subs(X, 2) == -2
        # (나) g(g(x))=4 의 서로 다른 실근 수·합
        r = sp.Poly(sp.expand(g.subs(X, g) - 4), X).nroots(n=20, maxsteps=500)
        re = sorted({round(float(sp.re(z)), 12) for z in r if abs(sp.im(z)) < 1e-9})
        if len(re) != 4 or abs(sum(re) - 6) > 1e-8:
            continue
        # (ㄴ) g(2^x)=2 의 실근 수 = g(t)=2 의 양근 수
        r2 = sp.Poly(sp.expand(g - 2), X).nroots(n=20)
        pos = [z for z in r2 if abs(sp.im(z)) < 1e-9 and sp.re(z) > 0]
        if len(pos) == 1:
            sols.append(a_)
    V.report(2, sols == [2], f"조건 (나)(다)(ㄴ) 만족 a={sols}")
    a0 = sols[0]
    V.check(2, "2-2 g", sp.expand(a0 * (X0 - sp.Rational(3, 2))**2 - 2 - sp.Rational(a0, 4)), V.ans(2, "2"))
    # 논제 1: f 도 같은 방식 — f(f(x))=4 의 실근이 2개이고 합 6인 f=a(x-3)^2+c (a>=2)가 실제로 존재하는지(예: a=2,c=?)
    ok = False
    for a_ in range(2, 6):
        for c_ in [sp.Rational(n_, 2) for n_ in range(-20, 8)]:
            f1 = a_ * (X - 3)**2 + c_
            r = sp.Poly(sp.expand(f1.subs(X, f1) - 4), X).nroots(n=20, maxsteps=500)
            re = sorted({round(float(sp.re(z)), 10) for z in r if abs(sp.im(z)) < 1e-9})
            if len(re) == 2 and abs(sum(re) - 6) < 1e-8:
                ok = True
    V.report(2, ok, "꼭짓점 x=3 인 f 중 (가) 만족 예 존재")

def p03():
    R = mp.mpf(25) / 2
    out_l = set()
    d = mp.sqrt(R**2 - 12**2)                   # 원 중심 O=(0,0) 에서 현 BC 까지의 거리
    for sgn, side in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:   # A=(0,sgn*R), BC 는 중심의 위(+)/아래(-)
        y0 = side * d
        A = mp.matrix([0, sgn * R]); B = mp.matrix([-12, y0]); C = mp.matrix([12, y0]); D = mp.matrix([0, -sgn * R])
        def ang(P, Q, Rr):  # ∠PQR
            u, v = P - Q, Rr - Q
            return mp.acos((u.T * v)[0] / (mp.norm(u) * mp.norm(v)))
        a_, b_, c_ = mp.norm(B - C), mp.norm(A - C), mp.norm(A - B)
        sA, sB = mp.sin(ang(B, A, C)), mp.sin(ang(A, B, C))
        cC = mp.cos(ang(A, C, B))
        assert abs(sA - 2 * sB * cC) < 1e-20          # (ㄱ)(나) 성립
        th = ang(A, C, D)
        V.report(3, abs(mp.cos(th)) < 1e-20, f"3-1 cosθ=0 (θ={mp.nstr(th, 8)})")
        out_l.add(int(mp.nint(mp.norm(C - D))))
    V.report(3, abs(mp.mpf(V.ans(3, "1"))) < 1e-20, "3-1 YAML 값 0")
    V.report(3, sorted(out_l) == sorted(int(s) for s in V.ans(3, "2").split(";")), f"3-2 l 값 집합 {sorted(out_l)}")

if __name__ == "__main__": V.run(globals())
