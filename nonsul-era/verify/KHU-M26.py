#!/usr/bin/env python3
"""경희대 2026학년도 모의논술(자연계) KHU-M26-01~03 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KHU-M26"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I: 사다리꼴 모양 확률밀도함수."""
    x, a, m = sp.symbols("x a m", positive=True)
    fA = m*x; fB = a*m/(1 - 2*a)*(x - a) + a*m; fC = -2*m*(x - 1)
    tot = sp.integrate(fA, (x, 0, a)) + sp.integrate(fB, (x, a, 1 - a)) + sp.integrate(fC, (x, 1 - a, 1))
    half = sp.integrate(fA, (x, 0, a)) + sp.integrate(fB, (x, a, sp.Rational(1, 2)))
    sols = [s for s in sp.solve([tot - 1, half - sp.Rational(3, 8)], [a, m], dict=True) if 0 < s[a] < sp.Rational(1, 2)]
    ok1 = len(sols) == 1 and sols[0][a] == sp.Rational(1, 3) and sols[0][m] == 3
    s = sols[0]
    xx = sp.symbols("xx", positive=True)
    pw = sp.Piecewise((fA.subs(s), x <= s[a]), (fB.subs(s), x <= 1 - s[a]), (fC.subs(s), True))
    Fx = sp.integrate(pw, (x, 0, xx))
    G = Fx*(1 - Fx)
    grid = max(range(1, 2000), key=lambda k: float(G.subs(xx, sp.Rational(k, 2000))))
    cand = [r for r in sp.solve(sp.integrate(fA.subs(s), (x, 0, xx)) - sp.Rational(1, 2), xx) if r < sp.Rational(2, 3)]
    ok2 = len(cand) == 1 and abs(grid/2000 - float(cand[0])) < 1e-3 and sp.simplify(cand[0] - 1/sp.sqrt(3)) == 0
    report(1, ok1 and ok2, f"(1) a={s[a]}, m={s[m]}; (2) 최대 x={cand} (격자 최대 {grid/2000})")


def p02():
    """논제 II: 반지름 2 원 위 A,B,C,D(반시계), AB=CD=2, 두 직선 예각 π/6, BC>AD."""
    R = 2
    th = sp.symbols("th", real=True)
    pt = lambda ang: sp.Matrix([R*sp.cos(ang), R*sp.sin(ang)])
    found = []
    # 호 BC 를 x 라 두고(호 AB=호 CD=π/3) 수치 탐색 후 정확화
    for k in range(1, 240):
        xarc = sp.pi*k/180
        A = pt(0); B = pt(sp.pi/3); C = pt(sp.pi/3 + xarc); D = pt(2*sp.pi/3 + xarc)
        u = B - A; w = D - C
        ang = sp.acos(abs(u.dot(w))/(u.norm()*w.norm()))
        if abs(float(ang) - float(sp.pi/6)) < 1e-9 and float((C - B).norm()) > float((A - D).norm()):
            found.append((xarc, A, B, C, D))
    xarc, A, B, C, D = found[0]
    BC = sp.nsimplify(sp.simplify((C - B).norm()))
    shoe = lambda Ps: sum(Ps[i][0]*Ps[(i+1) % 4][1] - Ps[(i+1) % 4][0]*Ps[i][1] for i in range(4))/2
    area = sp.simplify(shoe([A, B, C, D]))
    ok1 = len(found) == 1 and sp.simplify(BC - (sp.sqrt(2) + sp.sqrt(6))) == 0 and sp.simplify(area - (3 + 2*sp.sqrt(3))) == 0
    # (2) 원기둥 단면: AB 를 y=-√3 에 두고 O=원점, 평면 z=√3(y+√3)
    y = sp.symbols("y", real=True)
    proj = sp.integrate(2*sp.sqrt(4 - y**2), (y, -sp.sqrt(3), 0))   # 0<=z<=3 ⇔ -√3<=y<=0
    sec = sp.simplify(proj*sp.sqrt(1 + 3))                         # 기울기 √3 평면의 넓이 배율
    ok2 = sp.simplify(sec - (sp.Rational(8, 3)*sp.pi + 2*sp.sqrt(3))) == 0
    report(2, ok1 and ok2, f"(1) 호BC={xarc}, BC={BC}, 넓이={area}; (2) 단면={sec}")


def p03():
    """논제 III: 접원 사슬 — 원 O1, O2 를 접촉 조건으로 직접 풀어 비를 확인하고 P 를 구함."""
    def chain(phi, count=3):
        x0, y0 = sp.cos(phi), sp.sin(phi)
        Q = ((y0 + 1)/x0, -1)
        # 직선 PQ: x0 x + y0 y = 1
        rs = []
        prev_c, prev_r = (0.0, 0.0), 1.0
        for _ in range(count):
            cx, r = sp.symbols("cx r", positive=True)
            cy = -1 + r
            eqs = [sp.sqrt((cx - prev_c[0])**2 + (cy - prev_c[1])**2) - (prev_r + r),
                   (1 - (x0*cx + y0*cy)) - r]          # 원은 접선의 원점 쪽(삼각형 내부)
            sol = sp.nsolve(eqs, [cx, r], [float(prev_c[0]) + 0.6*float(prev_r), 0.3*float(prev_r)], prec=30)
            c_new = (sol[0], -1 + sol[1]); rs.append(sol[1])
            prev_c, prev_r = c_new, sol[1]
        return Q, rs
    phi = sp.pi/6
    Q, rs = chain(phi)
    k = (sp.sin(phi) + 1)/sp.cos(phi)
    r1_formula = (sp.sqrt(k**2 + 1) - 1)/(sp.sqrt(k**2 + 1) + 1)
    ok1 = abs(float(rs[0] - r1_formula)) < 1e-20 and abs(float(rs[1]/rs[0] - rs[0])) < 1e-20 and abs(float(rs[2]/rs[1] - rs[0])) < 1e-20
    # 다른 P 에서도 공식 확인(y0<0 포함)
    for ph in (-sp.pi/5, sp.pi/3, sp.Rational(1, 10)):
        _, rr = chain(ph, 1)
        kk = (sp.sin(ph) + 1)/sp.cos(ph)
        ok1 &= abs(float(rr[0] - (sp.sqrt(kk**2 + 1) - 1)/(sp.sqrt(kk**2 + 1) + 1))) < 1e-20
    total = sp.pi*rs[0]**2/(1 - rs[0]**2)
    ok2 = abs(float(total - sp.pi/8)) < 1e-20
    # 유일성: 합은 phi 에 대해 단조(격자 확인)
    def tot(ph):
        kk = (sp.sin(ph) + 1)/sp.cos(ph); r = (sp.sqrt(kk**2 + 1) - 1)/(sp.sqrt(kk**2 + 1) + 1)
        return float(sp.pi*r**2/(1 - r**2))
    vals = [tot(-sp.pi/2 + sp.pi*j/400) for j in range(1, 400)]
    ok3 = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    ph = sp.symbols("ph", real=True)
    kk = (sp.sin(ph) + 1)/sp.cos(ph); rr = (sp.sqrt(kk**2 + 1) - 1)/(sp.sqrt(kk**2 + 1) + 1)
    phs = sp.nsolve(sp.pi*rr**2/(1 - rr**2) - sp.pi/8, ph, 0.3, prec=30)   # 합=π/8 를 풀어 P 를 구함
    ok3 &= abs(float(sp.cos(phs) - sp.sqrt(3)/2)) < 1e-20 and abs(float(sp.sin(phs) - sp.Rational(1, 2))) < 1e-20
    report(3, ok1 and ok2 and ok3, f"P=(cos π/6, sin π/6)=(√3/2,1/2): Q_x={sp.nsimplify(Q[0])}, r1={sp.nsimplify(float(rs[0]))}, 넓이합={float(total):.12f} vs π/8")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
