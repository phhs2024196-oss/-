#!/usr/bin/env python3
"""SKKU-G24A (성균관대 2024학년도 수시 논술우수전형 자연계 1교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "SKKU-G24A"
WIDTH = 2
RESULTS = []
x, y = sp.symbols("x y", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _argmin_grid(fun, ranges, n=200):
    best = None
    axes = [[lo + (hi - lo) * i / n for i in range(n + 1)] for lo, hi in ranges]
    for pt in itertools.product(*axes):
        v = fun(*pt)
        if best is None or v < best[0]:
            best = (v, pt)
    return best[1]


def p01():
    # (i) [증명-수치확인] 곡선 y=x^3-x, 점 P 여러 개: 최단점에서 접선 ⊥ PQ
    f = x**3 - x
    fp = sp.diff(f, x)
    ok1 = True
    for P in [(0.3, 2.0), (-1.5, 0.7), (2.0, -1.0), (0.0, 1.0)]:
        d2 = (x - P[0])**2 + (f - P[1])**2
        d2f = sp.lambdify(x, d2)
        x0 = _argmin_grid(lambda t: d2f(t), [(-4, 4)], 4000)[0]
        c = sp.nsolve(sp.diff(d2, x), x, x0)
        dot = (c - P[0]) * 1 + (f.subs(x, c) - P[1]) * fp.subs(x, c)   # 접선벡터 (1,f'(c))·(Q-P)
        ok1 &= abs(float(dot)) < 1e-9
    # (ii) [증명-수치확인] 만나지 않는 두 곡선 y=x^2+1, y=x-3 사이 최단 선분의 양 끝 접선 ⊥ QR
    s, t = sp.symbols("s t", real=True)
    g, h = s**2 + 1, t - 3
    D2 = (s - t)**2 + (g - h)**2
    D2f = sp.lambdify((s, t), D2)
    s0, t0 = _argmin_grid(D2f, [(-3, 3), (-3, 5)], 300)
    sol = sp.nsolve([sp.diff(D2, s), sp.diff(D2, t)], [s, t], [s0, t0])
    sv, tv = sol
    QR = (tv - sv, h.subs(t, tv) - g.subs(s, sv))
    ok2 = abs(float(QR[0] + QR[1] * sp.diff(g, s).subs(s, sv))) < 1e-9 and abs(float(QR[0] + QR[1] * 1)) < 1e-9
    # (iii) AB+BC+CD 최소: A=(u^2+1,u)(u>=0), D=(c, 2c^2-30c+113). 반사점 A'(u,u^2+1), D'(c,-(2c^2-30c+113)).
    u, c = sp.symbols("u c", real=True)
    Ap = (u, u**2 + 1)
    Dp = (c, -(2 * c**2 - 30 * c + 113))
    L2 = (Ap[0] - Dp[0])**2 + (Ap[1] - Dp[1])**2
    L2f = sp.lambdify((u, c), L2)
    u0, c0 = _argmin_grid(L2f, [(0, 5), (4, 11)], 400)
    us, cs = sp.nsolve([sp.diff(L2, u), sp.diff(L2, c)], [u, c], [u0, c0])
    # 직접 꺾은선 최소 확인: B=(b,b), C=(e,0) 에 대해 AB+BC+CD 를 격자로 최소화한 값 ≈ |A'D'|
    Av = (float(us)**2 + 1, float(us)); Dv = (float(cs), 2 * float(cs)**2 - 30 * float(cs) + 113)
    tot = lambda b, e: math.dist(Av, (b, b)) + math.dist((b, b), (e, 0)) + math.dist((e, 0), Dv)
    bb, ee = _argmin_grid(tot, [(0, 3), (0, 8)], 600)
    ok_path = abs(tot(bb, ee) - math.sqrt(float(L2f(us, cs)))) < 1e-2
    cond = abs(float(cs - us) - 6) < 1e-9                                       # D의 x좌표 - A의 y좌표 = 6
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    mA = sp.sympify(parts["A"]); mD = sp.sympify(parts["D"])
    ok3 = cond and ok_path and all(abs(float(a - b)) < 1e-9 for a, b in zip(mA + mD, Av + Dv))
    report(1, ok1 and ok2 and ok3, f"[증명-수치확인](i)(ii) A={Av} D={Dv} c-b={float(cs-us):.6f} 꺾은선최소≈|A'D'|")


def p02():
    # (i) 기울기 k, y절편 p 미지수. A,D: x^2=kx+p, B=(0,0), C=(k,k^2). P=대각선 AC, BD 교점.
    k, p = sp.symbols("k p", positive=True)
    r1 = (k - sp.sqrt(k**2 + 4 * p)) / 2; r2 = (k + sp.sqrt(k**2 + 4 * p)) / 2
    A = (r1, r1**2); D = (r2, r2**2); B = (sp.Integer(0), sp.Integer(0)); C = (k, k**2)
    lam, mu = sp.symbols("lam mu")
    # P = A + lam(C-A) = B + mu(D-B)
    sol = sp.solve([A[0] + lam * (C[0] - A[0]) - mu * D[0], A[1] + lam * (C[1] - A[1]) - mu * D[1]], [lam, mu], dict=True)[0]
    P = (sp.simplify(mu * D[0]).subs(sol), sp.simplify(mu * D[1]).subs(sol))
    M1 = ((A[0] + D[0]) / 2, (A[1] + D[1]) / 2); M2 = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
    e1 = (P[0] - M1[0])**2 + (P[1] - M1[1])**2 - 25
    e2 = (P[0] - M2[0])**2 + (P[1] - M2[1])**2 - 1
    areas = set()
    for k0 in [0.5, 1.2, 2, 3]:
        for p0 in [3, 6, 9]:
            try:
                kv, pv = sp.nsolve([e1, e2], [k, p], [k0, p0])
            except Exception:  # noqa: BLE001
                continue
            if kv > 0 and pv > 0:
                pts = [tuple(float(sp.N(c.subs({k: kv, p: pv}))) for c in V) for V in (A, B, C, D)]
                ar = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4)))
                areas.add(round(ar, 9))
    ok1 = len(areas) == 1 and abs(areas.pop() - float(sp.sympify(md_answer(2, "1")))) < 1e-8
    # (ii) 정수 중심 (h,kk), r=4 인 원과 y=x^2 의 교점 4개, y좌표 정수, 사다리꼴 → 넓이 전수
    import numpy as np
    found = set()
    for hc in range(-12, 13):
        for kc in range(-12, 40):
            # (x-h)^2+(x^2-k)^2=16 : x^4 + (1-2k)x^2 - 2h x + h^2+k^2-16 = 0
            rts = np.roots([1, 0, 1 - 2 * kc, -2 * hc, hc * hc + kc * kc - 16])
            xs = sorted(z.real for z in rts if abs(z.imag) < 1e-7)
            if len(xs) != 4 or min(b - a for a, b in zip(xs, xs[1:])) < 1e-6:
                continue
            if any(abs(v * v - round(v * v)) > 1e-6 for v in xs):
                continue
            quad = [(v, v * v) for v in xs]
            # 포물선 위 점은 볼록 위치: 둘레 순서는 x 오름차순
            sl = lambda P, Q: (Q[1] - P[1]) / (Q[0] - P[0])
            # 둘레 순서: P0-P1-P2-P3-P0 의 변 중 대변 평행 여부 (P0P1 // P2P3 또는 P1P2 // P3P0)
            par = abs(sl(quad[0], quad[1]) - sl(quad[2], quad[3])) < 1e-7 or abs(sl(quad[1], quad[2]) - sl(quad[3], quad[0])) < 1e-7
            if not par:
                continue
            ar = 0.5 * abs(sum(quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1] for i in range(4)))
            found.add(round(ar, 8))
    md = sp.sympify(md_answer(2, "2"))
    ok2 = found == {round(float(v), 8) for v in md}
    # (iii) m<n 양의 정수, 완전제곱 아님, g=-(x^2-m)(x^2-n) 과 y=x^2 의 네 교점을 지나는 원의 반지름 10
    def issq(v):
        return math.isqrt(v)**2 == v
    cnt = 0
    for m in range(1, 400):
        if issq(m):
            continue
        for n in range(m + 1, 420):
            if issq(n):
                continue
            # 교점: x^2 = -(x^2-m)(x^2-n)
            rts = np.roots([-1, 0, m + n - 1, 0, -m * n])
            xs = sorted(z.real for z in rts if abs(z.imag) < 1e-9)
            if len(xs) != 4 or min(b - a for a, b in zip(xs, xs[1:])) < 1e-9:
                continue
            P3 = [(v, v * v) for v in xs[:3]]
            (x1, y1), (x2, y2), (x3, y3) = P3
            d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
            uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
            R = math.hypot(x1 - ux, y1 - uy)
            R4 = math.hypot(xs[3] - ux, xs[3]**2 - uy)
            if abs(R - 10) < 1e-6 and abs(R4 - 10) < 1e-6:
                cnt += 1
    ok3 = cnt == int(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"넓이(i)={md_answer(2,'1')} 넓이집합(ii)={sorted(found)} 개수(iii)={cnt}")


def p03():
    p, q, r = sp.symbols("p q r", positive=True)
    X = sp.symbols("X", real=True)
    g = -p * x**2 + q * x + r
    m = sp.diff(g, x).subs(x, 0)
    alpha = sp.solve(sp.Eq(m * X + r, 0), X)[0]
    beta = sp.solve(sp.Eq(-X / m + r, 0), X)[0]
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    loc = {"q": q, "r": r}
    ok1 = sp.simplify(alpha - sp.sympify(parts["alpha"], locals=loc)) == 0 and sp.simplify(beta - sp.sympify(parts["beta"], locals=loc)) == 0
    # (ii) 전수: A,B,C 양의 정수, B>C, B^2+4AC=100; q,r 양의 정수; alpha=-r/q, beta=qr 이 f=0 의 해
    res = []
    for Bv in range(1, 11):
        for Av in range(1, 101):
            for Cv in range(1, Bv):
                if Bv * Bv + 4 * Av * Cv != 100:
                    continue
                f = -Av * x**2 + Bv * x + Cv
                for qv in range(1, 60):
                    for rv in range(1, 60):
                        a_ = sp.Rational(-rv, qv); b_ = qv * rv
                        if f.subs(x, a_) == 0 and f.subs(x, b_) == 0:
                            pmin = next(pv for pv in range(1, 10000) if math.isqrt(qv * qv + 4 * pv * rv)**2 == qv * qv + 4 * pv * rv)
                            res.append((Av, Bv, Cv, pmin, qv, rv))
    md2 = {tuple(int(v) for v in t) for t in sp.sympify(md_answer(3, "2"))}
    ok2 = set(res) == md2
    # (iii) 모든 허용 (p,q,r)(p 최소 아닌 것 포함)에 대해 넓이 계산 → 최솟값
    areas = []
    for (Av, Bv, Cv, pmin, qv, rv) in res:
        for pv in range(1, 400):
            if math.isqrt(qv * qv + 4 * pv * rv)**2 != qv * qv + 4 * pv * rv:
                continue
            gg = -pv * x**2 + qv * x + rv
            gam = min(sp.solve(gg, x))
            al = sp.Rational(-rv, qv)
            assert al <= gam < 0
            area = sp.integrate(qv * x + rv, (x, al, 0)) - sp.integrate(gg, (x, gam, 0))
            areas.append(sp.nsimplify(area))
    amin = min(areas)
    ok3 = sp.simplify(amin - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok1 and ok2 and ok3, f"alpha={alpha} beta={beta} (ii)={sorted(res)} 넓이최소={amin}")


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
