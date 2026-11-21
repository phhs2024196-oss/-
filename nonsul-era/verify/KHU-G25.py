#!/usr/bin/env python3
"""KHU-G25 (경희대 2025학년도 논술 자연계 실전, KHU-G25-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import math
import random
import re
import sys

import sympy as sp

PREFIX = "KHU-G25"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[논제 Ⅰ] 정사각형 한 변 4, 원 O1 내접, 정삼각형 PQR 내접, 원 O2 는 PR·PD·CD 에 접함."""
    s3 = sp.sqrt(3)
    P = sp.Matrix([0, 2]); Q = sp.Matrix([-s3, -1]); R = sp.Matrix([s3, -1])
    r = sp.symbols("r", positive=True)
    O2 = sp.Matrix([2 - r, 2 - r])                       # 변 AD(y=2), CD(x=2)에 접함
    # 직선 PR: 점 P, 방향 R-P. 거리 = r
    d = R - P
    dist = sp.Abs(d[0] * (O2[1] - P[1]) - d[1] * (O2[0] - P[0])) / sp.sqrt(d.dot(d))
    sols = [s for s in sp.solve(sp.Eq(dist**2, r**2), r) if s.is_positive and s < 2]
    # 선분 PR 위의 접점이고 원 O2 가 정사각형 안(모서리 D 쪽)에 있는 해: 작은 해
    rr = min(sols, key=lambda v: float(v))
    o1o2 = sp.sqrt(2) * (2 - rr)
    ok1 = sp.simplify(rr - (s3 - 1)) == 0 and sp.simplify(o1o2 - (3 * sp.sqrt(2) - sp.sqrt(6))) == 0
    # (2) QR·PX 최대, X 는 원 O2 위
    O2v = O2.subs(r, rr)
    QR = R - Q
    exact_max = sp.nsimplify(sp.simplify(QR.dot(O2v - P) + rr * sp.sqrt(QR.dot(QR))))
    grid = max(float(QR.dot(O2v + rr * sp.Matrix([math.cos(2 * math.pi * k / 20000), math.sin(2 * math.pi * k / 20000)]) - P))
               for k in range(20000))
    ok2 = sp.simplify(exact_max - 4 * s3) == 0 and abs(grid - float(4 * s3)) < 1e-6
    # 대표 오답: X 를 O2 의 중심으로 둠 → 2√3(2-r)
    wrong = sp.simplify(QR.dot(O2v - P))
    report(1, ok1 and ok2, f"r={sp.simplify(rr)}, O1O2={sp.simplify(o1o2)}, max={exact_max} (격자 {grid:.6f}); 중심대입 오답={wrong}")


def p02():
    """[논제 Ⅱ] 막대 길이 5, PQ:QR=2:3."""
    # (1) P(t,0), Q=(q, sqrt(4-q^2)), |PQ|=2
    q = sp.symbols("q", real=True)
    qs = sp.solve(sp.Eq((q - t)**2 + 4 - q**2, 4), q)
    ok1 = qs == [t / 2]
    Qp = sp.Matrix([t / 2, sp.sqrt(4 - t**2 / 4)])
    Pp = sp.Matrix([t, 0])
    Rp = Pp + sp.Rational(5, 2) * (Qp - Pp)                # PR = (5/2) PQ (같은 방향)
    for tv in [-4, -3, -1, 0, sp.Rational(1, 2), 2, 4]:
        xr, yr = Rp.subs(t, tv)
        ok1 &= sp.simplify(yr - 5 * sp.sqrt(1 - xr**2)) == 0
    ok1 &= sp.simplify(Rp[0].subs(t, -4) - 1) == 0 and sp.simplify(Rp[0].subs(t, 4) + 1) == 0
    # (2) 교점 개수: y=5sqrt(1-x^2) (-1<=x<=1), y=5-x^2/a
    def count(a):
        a = sp.nsimplify(a)
        poly = sp.Poly(sp.expand(25 * (1 - x**2) - (5 - x**2 / a)**2), x)
        pts = set()
        for rt in sp.roots(poly, multiple=False) or {}:
            if rt.is_real and -1 <= rt <= 1 and sp.simplify(5 - rt**2 / a) >= 0:
                pts.add(sp.nsimplify(rt))
        if not pts:   # 수치 보조
            for rt in sp.Poly(poly).nroots():
                if abs(sp.im(rt)) < 1e-12 and -1 - 1e-12 <= sp.re(rt) <= 1 + 1e-12 and 5 - sp.re(rt)**2 / a >= -1e-12:
                    pts.add(round(float(sp.re(rt)), 9))
        return len(pts)
    tests = ["1/10", "19/100", "1/5", "1/4", "3/10", "39/100", "2/5", "41/100", "1", "3"]
    ok2 = True
    detail = []
    for a in tests:
        av = sp.Rational(a)
        c = count(av)
        detail.append(f"{a}:{c}")
        ok2 &= ((c >= 2) == (sp.Rational(1, 5) <= av < sp.Rational(2, 5)))
    report(2, ok1 and ok2, f"Q=(t/2,..), R 관계식 y=5√(1-x²) 확인, 교점 수 {' '.join(detail)} → 1/5≤a<2/5")


def _next(d, axis_angle, line_angle):
    """축 위 거리 d 인 점 A(방향 axis_angle)에서 직선(방향 line_angle) 위 점 B 를 ∠ABO=π/3 으로 찾고,
    B 에서 다음 축(axis_angle+π/2)에 내린 수선의 발까지의 거리 반환."""
    A = (d * math.cos(axis_angle), d * math.sin(axis_angle))
    u = (math.cos(line_angle), math.sin(line_angle))

    def ang(s):
        B = (s * u[0], s * u[1])
        v1 = (A[0] - B[0], A[1] - B[1]); v2 = (-B[0], -B[1])
        c = (v1[0] * v2[0] + v1[1] * v2[1]) / math.hypot(*v1) / math.hypot(*v2)
        return math.acos(max(-1, min(1, c)))
    lo, hi = 1e-12, 1e6 * d
    for _ in range(200):                                # ang 는 s 에 대해 감소
        mid = (lo + hi) / 2
        if ang(mid) > math.pi / 3:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    B = (s * u[0], s * u[1])
    na = axis_angle + math.pi / 2
    return abs(B[0] * math.cos(na) + B[1] * math.sin(na))


def chain(al, be, steps):
    d = 1.0
    out = [d]
    lines = [al, math.pi - be, math.pi + al, 2 * math.pi - be]
    for k in range(steps):
        d = _next(d, k * math.pi / 2, lines[k % 4])
        out.append(d)
    return out          # out[j] = OP_{2j+1}


def p03():
    """[논제 Ⅲ] 작도 시뮬레이션으로 OP5, 급수 합 검산."""
    a, b = sp.symbols("alpha beta")
    A, B, C = sp.Rational(4, 3), sp.pi / 3, -sp.pi / 3                    # md 정답
    formula = A * sp.sin(a + B) * sp.sin(a) * sp.cos(b + C) * sp.cos(b)
    random.seed(1)
    ok1 = True
    for _ in range(20):
        al, be = random.uniform(0.05, 1.5), random.uniform(0.05, 1.5)
        op5 = chain(al, be, 2)[2]
        ok1 &= abs(op5 - float(formula.subs({a: al, b: be}))) < 1e-8
    # 조건(0<B<π/2, -π/2<C<0)에서 표현이 유일한지: 다른 (B,C) 후보가 함수로 같지 않음을 몇 점에서 확인
    # (2) β=π/6
    f = 1 / (1 - sp.sin(a + sp.pi / 3) * sp.sin(a))
    ok2 = True
    for al in [0.2, 0.7, 1.0, 1.3]:
        ds = chain(al, math.pi / 6, 8)                # OP1, OP5, OP9 = ds[0], ds[2], ds[4]
        r1, r2 = ds[2] / ds[0], ds[4] / ds[2]
        ok2 &= abs(r1 - r2) < 1e-8 and abs(1 / (1 - r1) - float(f.subs(a, al))) < 1e-8
    crit = [c for c in sp.solve(sp.diff(f, a), a) if 0 < float(c) < math.pi / 2]
    grid = max((float(f.subs(a, k * (math.pi / 2) / 4000)), k) for k in range(1, 4000))
    amax = grid[1] * (math.pi / 2) / 4000
    ok2 &= len(crit) == 1 and abs(float(crit[0]) - amax) < 1e-3 and sp.simplify(f.subs(a, crit[0]) - 4) == 0
    report(3, ok1 and ok2, f"OP5 작도 20점 일치(A=4/3,B=π/3,C=-π/3), 공비 일정, f' 근 {crit}, 최대 {sp.simplify(f.subs(a, crit[0]))}")


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
