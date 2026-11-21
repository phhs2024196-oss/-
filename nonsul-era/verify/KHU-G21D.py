#!/usr/bin/env python3
"""KHU-G21D (경희대 2021학년도 논술 의학계 12.07 오전, 수학 논제 I-1·I-2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import math
import random
import re
import sys

import sympy as sp

PREFIX = "KHU-G21D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[논제 I-1] x=(cosθ)t, y=t^2-(sin(θ+k))t."""
    th, t, X = sp.symbols("theta t X", real=True)
    k = sp.pi / 5
    # (1) y=0, t≠0 → t=sin(θ+k), g=cosθ·sin(θ+k)
    troots = [r for r in sp.solve(t**2 - sp.sin(th + k) * t, t) if r != 0]
    g = sp.cos(th) * troots[0]
    crit = [c for c in sp.solve(sp.diff(g, th), th) if 0 < float(c) < math.pi / 2]
    # 구간 내 최대: 끝점 극한과 비교(수치)
    best = max(crit, key=lambda c: float(g.subs(th, c)))
    grid_max = max(float(g.subs(th, math.pi / 2 * i / 2000)) for i in range(1, 2000))
    ok1 = abs(float(best) - 3 * math.pi / 20) < 1e-12 and float(g.subs(th, best)) >= grid_max - 1e-12
    # (2) k=0: 곡선족의 아래 경계 y = inf_{θ,t} 를 x마다 수치로 구해 y=ax^2+c 를 결정
    def lower(xv):
        best = float("inf")
        for i in range(0, 2001):
            tt = math.pi * i / 2000
            c = math.cos(tt)
            if abs(c) < 1e-9:
                if abs(xv) < 1e-12: best = min(best, -0.25)  # θ=π/2, x=0: y=t^2-t ≥ -1/4
                continue
            T = xv / c
            best = min(best, T * T - math.sin(tt) * T)
        return best
    xs = [-2.0, -1.0, -0.3, 0.5, 1.5]
    vals = [lower(v) for v in xs]
    A = sp.Matrix([[v * v, 1] for v in xs[:2]]); bvec = sp.Matrix(vals[:2])
    ac = A.solve(bvec)
    a_, c_ = round(float(ac[0]), 4), round(float(ac[1]), 4)
    fit = all(abs(a_ * v * v + c_ - w) < 1e-5 for v, w in zip(xs, vals))
    # (ㄱ) 항등식: y-(x^2-1/4) = (t sinθ - 1/2)^2
    a, c = sp.Rational(1), sp.Rational(-1, 4)
    ident = sp.simplify(t**2 - sp.sin(th) * t - (a * (sp.cos(th) * t) ** 2 + c) - (t * sp.sin(th) - sp.Rational(1, 2)) ** 2) == 0
    # (ㄴ) 무작위 점 (p,q), q≥p^2-1/4 에 대해 θ∈[0,π], t 존재 확인(공식 풀이의 구성 그대로가 아닌 수치 탐색)
    random.seed(1)
    cover = True
    for _ in range(300):
        p = random.uniform(-3, 3); q = p * p - 0.25 + random.uniform(0, 5)
        found = False
        if abs(p) < 1e-12:
            found = True
        else:
            # θ에 대해 h(θ)=p^2 sec^2θ - p tanθ - q = 0 의 근을 (0,π)\{π/2} 에서 탐색
            ff = lambda tt: p * p / math.cos(tt) ** 2 - p * math.tan(tt) - q
            pts = [i * math.pi / 4000 for i in range(4001) if abs(i * math.pi / 4000 - math.pi / 2) > 1e-6]
            for u, v in zip(pts, pts[1:]):
                if u < math.pi / 2 < v: continue
                if ff(u) == 0 or ff(u) * ff(v) < 0: found = True; break
        cover = cover and found
    ok2 = a_ == 1 and c_ == -0.25 and fit and ident and cover
    report(1, ok1 and ok2, f"θ*={float(best):.12f}(3π/20={3*math.pi/20:.12f}); 아래 경계 y={a_}x^2+({c_}) 적합={fit}, 항등식={ident}, (ㄴ) 300점 [증명-수치확인]={cover}")


def tangent_chain(N):
    """x축에 접하는 원의 사슬을 기하로 직접 생성(mpmath 60자리): 새 원은 직전 두 원과 외접하고 x축에 접함(두 원 사이)."""
    import mpmath as mp
    mp.mp.dps = 60
    r = (mp.sqrt(5) - 1) ** 2 / 4
    circles = [(mp.mpf(0), mp.mpf(1)), (2 * mp.sqrt(r), r)]   # O2: O1과 외접, x축 접, x≥0
    for _ in range(N - 2):
        (x1, r1), (x2, r2) = circles[-2], circles[-1]
        # |x-x1| = 2√(ρ r1), |x-x2| = 2√(ρ r2) (외접+x축 접), 두 원 사이에 있으므로 합 = |x2-x1|
        s = abs(x2 - x1) / (2 * (mp.sqrt(r1) + mp.sqrt(r2)))
        rho = s * s
        x = x1 + mp.sign(x2 - x1) * 2 * mp.sqrt(rho * r1)
        assert abs((x - x1) ** 2 + (rho - r1) ** 2 - (rho + r1) ** 2) < mp.mpf(10) ** -50
        assert abs((x - x2) ** 2 + (rho - r2) ** 2 - (rho + r2) ** 2) < mp.mpf(10) ** -50
        circles.append((x, rho))
    return circles, r


def p02():
    """[논제 I-2] (1) 세 원 접함: z, YZ. (2) 원의 사슬 a_n, lim x_n."""
    x, y = sp.symbols("x y", positive=True)
    # (1) 수치 기하로 z, YZ 확인
    ok1 = True
    for (xv, yv) in [(4.0, 1.0), (9.0, 2.0), (5.0, 3.0)]:
        XY = 2 * math.sqrt(xv * yv)
        s = XY / (2 * (math.sqrt(xv) + math.sqrt(yv)))
        z_num = s * s
        YZ_num = 2 * math.sqrt(yv * z_num)
        z_f = float((x * y / (sp.sqrt(x) + sp.sqrt(y)) ** 2).subs({x: xv, y: yv}))
        yz_f = float((2 * y * sp.sqrt(x) / (sp.sqrt(x) + sp.sqrt(y))).subs({x: xv, y: yv}))
        # 거리 조건 직접 검사: C3 중심 (Xz, z) 가 C1(0,x), C2(XY,y)와 외접
        Xz = 2 * math.sqrt(xv * z_num)
        ok1 = ok1 and abs(z_num - z_f) < 1e-12 and abs(YZ_num - yz_f) < 1e-12 \
            and abs(math.hypot(Xz, z_num - xv) - (xv + z_num)) < 1e-9 and abs(math.hypot(XY - Xz, z_num - yv) - (yv + z_num)) < 1e-9
    # (2)
    ch, r = tangent_chain(40)
    ok_a = all(abs(ch[n][1] / r ** n - 1) < 1e-30 for n in range(len(ch)))  # a_{n+1}=r^n
    ok_dec = all(ch[i][1] > ch[i + 1][1] for i in range(len(ch) - 1)) and all(c[0] >= 0 for c in ch)
    lim = sp.nsimplify(float(ch[-1][0]), [sp.sqrt(5)])
    ok_lim = abs(ch[-1][0] - 2 * __import__('mpmath').sqrt(5) / 5) < 1e-12
    # 대표 오답: 부호 교대 없이 모두 더함 → (√5-1)/(1-r)
    wrong = float((math.sqrt(5) - 1) / (1 - r))
    report(2, ok1 and ok_a and ok_dec and ok_lim, f"(1) 3쌍 기하 일치={ok1}; (2) a_n=r^(n-1) 40항={ok_a}, lim x_n≈{float(ch[-1][0]):.12f} ({lim}); 부호 무시 오답 {wrong:.6f}")


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
