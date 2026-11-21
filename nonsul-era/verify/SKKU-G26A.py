#!/usr/bin/env python3
"""SKKU-G26A (성균관대 2026학년도 수시 논술위주 수리형 1교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import sys

import numpy as np
import sympy as sp

PREFIX = "SKKU-G26A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


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


def g_pieces(f):
    """<제시문2>의 g 를 (식, 구간) 목록으로."""
    fp = sp.diff(f, x)
    return [(fp.subs(x, -1) * (x + 1) + f.subs(x, -1), -2, -1),
            (f, -1, 1),
            (fp.subs(x, 1) * (x - 1) + f.subs(x, 1), 1, 2)]


def p01():
    # (i)
    I1 = sum(sp.integrate(e, (x, lo, hi)) for e, lo, hi in g_pieces(-2 * x**2 + 1))
    ok1 = sp.simplify(I1 - sp.sympify(md_answer(1, "i"))) == 0
    # (ii) f = a x^2 + b x + c, 조건 (2)로 a,b, 조건 (1)로 c 범위, 넓이 = ∫(g+2) (g>-2 확인) 의 최소
    a, b, c = sp.symbols("a b c", real=True)
    f = a * x**2 + b * x + c
    sab = sp.solve([sp.diff(f, x).subs(x, 1) - sp.Rational(1, 3), sp.diff(f, x).subs(x, -1) + sp.Rational(1, 2)], [a, b])
    f = f.subs(sab)
    lo = max(sp.solve(sp.Eq(f.subs(x, 1), -1), c)[0], sp.solve(sp.Eq(f.subs(x, -1), -1), c)[0])
    hi = min(sp.solve(sp.Eq(f.subs(x, 1), 1), c)[0], sp.solve(sp.Eq(f.subs(x, -1), 1), c)[0])
    area = lambda cv: sum(sp.integrate(e.subs(c, cv) + 2, (x, l, h)) for e, l, h in g_pieces(f))
    above = all(min(float(e.subs(c, cv).subs(x, xv)) for e, l, h in g_pieces(f)
                    for xv in np.linspace(l, h, 201)) > -2 for cv in (lo, hi))
    grid = [lo + (hi - lo) * sp.Rational(i, 40) for i in range(41)]
    vals = [area(cv) for cv in grid]
    amin = min(vals)
    ok2 = above and vals[0] == amin and sp.simplify(amin - sp.sympify(md_answer(1, "ii"))) == 0
    # (iii) G = min(g,2), 수치 적분으로 F(c)=109/10 의 근을 (0,1] 에서 탐색
    import mpmath as mp
    cc = sp.symbols("cc", positive=True)

    def F(cv):
        pcs = g_pieces(cv * x * (x - 1))
        tot = 0
        for e, l, h in pcs:
            fn = sp.lambdify(x, e, "mpmath")
            tot += mp.quad(lambda u: min(fn(u), 2) + 2, mp.linspace(l, h, 41))
        return tot - mp.mpf(109) / 10
    cs = np.linspace(0.001, 1, 1000)
    Fv = [F(cv) for cv in cs]
    roots = []
    for i in range(len(cs) - 1):
        if Fv[i] == 0 or Fv[i] * Fv[i + 1] < 0:
            roots.append(float(mp.findroot(F, (cs[i], cs[i + 1]), solver="bisect")))
    md3 = float(sp.sympify(md_answer(1, "iii")))
    ok3 = len(roots) == 1 and abs(roots[0] - md3) < 1e-6
    report(1, ok1 and ok2 and ok3, f"(i)={I1} (ii) c∈[{lo},{hi}] min={amin} (iii) roots={roots}")


def count_hits(a, m, N, exact=False):
    """f 의 그래프(삼각형 T_1..T_N 의 윗 두 변)와 y=m(x+1) 의 교점 수."""
    r3 = sp.sqrt(3) if exact else math.sqrt(3)
    cnt = 0
    L = 0
    for n in range(1, N + 1):
        bn = 1 + (n - 1) * a
        R = L + bn
        M = (L + R) / 2 if not exact else sp.Rational(1, 2) * (L + R)
        x1 = (r3 * L + m) / (r3 - m)          # 왼쪽 변 y=√3(x-L)
        x2 = (r3 * R - m) / (r3 + m)          # 오른쪽 변 y=√3(R-x)
        h1 = bool(L <= x1 <= M) if exact else (L - 1e-12 <= x1 <= M + 1e-12)
        h2 = bool(M <= x2 <= R) if exact else (M - 1e-12 <= x2 <= R + 1e-12)
        same = (sp.simplify(x1 - x2) == 0) if exact else abs(x1 - x2) < 1e-10
        cnt += int(h1) + int(h2) - (1 if (h1 and h2 and same) else 0)
        L = R
    return cnt


def p02():
    # (i) a=2: 각 삼각형 구간에서 제시문 식을 그대로 적분
    a = 2
    S = [0]
    for n in range(1, 30):
        S.append(S[-1] + 1 + (n - 1) * a)
    tot = 0
    for n in range(1, 30):
        L, R = S[n - 1], S[n]
        if L >= 134:
            break
        fn = sp.sqrt(3) / 2 * (R - L) - sp.sqrt(3) * sp.Abs(x - sp.Rational(L + R, 2))
        hi = min(R, 134)
        M = sp.Rational(L + R, 2)
        tot += sp.integrate(fn, (x, L, min(M, hi)))
        if hi > M:
            tot += sp.integrate(fn, (x, M, hi))
    ok1 = sp.simplify(tot - sp.sympify(md_answer(2, "i"))) == 0
    # (ii) a=1, m=√3/10 정확 계산
    c2 = count_hits(sp.Integer(1), sp.sqrt(3) / 10, 60, exact=True)
    ok2 = c2 == int(md_answer(2, "ii"))
    # (iii) m=√3/30: a 격자에서 교점 30 인 집합, 양 끝은 정확 계산
    m = math.sqrt(3) / 30
    As = np.linspace(0.001, 0.03, 2901)
    good = [float(av) for av in As if count_hits(float(av), m, 3000) == 30]
    lo, hi = min(good), max(good)
    md = sp.sympify(md_answer(2, "iii"))
    L0, R0 = float(md.start), float(md.end)
    contiguous = len(good) == sum(1 for av in As if L0 < av < R0)
    ends = [count_hits(e, sp.sqrt(3) / 30, 60, exact=True) for e in (md.start, md.end)]
    ok3 = md.left_open and md.right_open and contiguous and abs(lo - L0) < 2e-5 and abs(hi - R0) < 2e-5 \
        and all(e != 30 for e in ends)
    report(2, ok1 and ok2 and ok3, f"(i)={sp.simplify(tot)} (ii)={c2} (iii) grid [{lo:.6f},{hi:.6f}] ends={ends}")


def hexpos(s, R):
    """한 변 R, 꼭짓점 (R,0) 에서 반시계로 s 만큼 이동한 점(벡터화)."""
    s = np.mod(s, 6 * R)
    i = np.floor(s / R)
    fr = s / R - i
    a0, a1 = i * np.pi / 3, (i + 1) * np.pi / 3
    return R * ((1 - fr) * np.cos(a0) + fr * np.cos(a1)), R * ((1 - fr) * np.sin(a0) + fr * np.sin(a1))


def p03():
    sP = lambda t: t**2 + 2 * t
    sQ = lambda t: 2 * t**2 + t
    # (i) 정확 좌표
    def hexpos_exact(s, R):
        s = sp.Mod(s, 6 * R)
        i = sp.floor(s / R)
        fr = s / R - i
        v = lambda j: sp.Matrix([R * sp.cos(j * sp.pi / 3), R * sp.sin(j * sp.pi / 3)])
        return (1 - fr) * v(i) + fr * v(i + 1)
    P = hexpos_exact(sP(sp.Integer(8)), 1)
    Q = hexpos_exact(sQ(sp.Integer(8)), 6)
    sin_t = sp.Abs(P[0] * Q[1] - P[1] * Q[0]) / (P.norm() * Q.norm())
    ok1 = sp.simplify(sin_t - sp.sympify(md_answer(3, "i"))) == 0
    # (ii) 실제 각도 함수의 π/3 통과 횟수
    T = np.linspace(0, 10, 4_000_001)
    px, py = hexpos(sP(T), 1.0)
    tot = 0
    det = []
    for j in range(3, 11):
        k = 5.0**j
        qx, qy = hexpos(sQ(T), k)
        th = np.arctan2(np.abs(px * qy - py * qx), px * qx + py * qy)
        dlt = th - np.pi / 3
        n = int(np.sum(np.sign(dlt[1:]) != np.sign(dlt[:-1])))
        det.append(n)
        tot += n
    ok2 = tot == int(md_answer(3, "ii"))
    # (iii) k=8: α 와 조건 (1)(2)를 만족하는 시각 수
    k = 8.0
    vx = np.cos(np.arange(6) * np.pi / 3)
    vy = np.sin(np.arange(6) * np.pi / 3)
    qs = np.linspace(0, 48, 480001)
    qx, qy = hexpos(qs, k)
    alpha = np.max(0.5 * np.abs(np.outer(qx, vy) - np.outer(qy, vx)))   # 볼록 → P 는 꼭짓점에서 최대
    def D(t):
        px_, py_ = hexpos(sP(t), 1.0)
        qx_, qy_ = hexpos(sQ(t), k)
        return alpha - 0.5 * np.abs(px_ * qy_ - py_ * qx_)
    def onCD(t):
        r = np.mod(sQ(t), 48)
        return (r >= 16 - 1e-9) & (r <= 24 + 1e-9)
    T = np.linspace(0, 6, 6_000_001)
    Dv = D(T)
    mask = onCD(T)
    idx = np.where((Dv[1:-1] <= Dv[:-2]) & (Dv[1:-1] <= Dv[2:]) & (Dv[1:-1] < 1e-3) & mask[1:-1])[0] + 1
    sols = []
    # Q 가 꼭짓점 C'(s=16,64) 또는 D'(s=24,72) 에 오는 순간도 후보(격자 국소최소에서 빠질 수 있음)
    vt = [(-1 + math.sqrt(1 + 8 * sv)) / 4 for sv in (16, 24, 64, 72)]
    for tt in vt:
        if 0 <= tt <= 6 and D(np.array([tt]))[0] < 1e-7:
            sols.append(tt)
    for i in idx:
        lo, hi = T[i - 1], T[i + 1]
        for _ in range(200):                      # 삼분 탐색(국소 단봉)
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if D(m1) < D(m2):
                hi = m2
            else:
                lo = m1
        tt = (lo + hi) / 2
        if D(tt) < 1e-7 and onCD(np.array([tt]))[0]:
            if all(abs(tt - u) > 1e-6 for u in sols):
                sols.append(tt)
    sols = sorted(float(u) for u in sols)
    ok3 = abs(alpha - 2 * math.sqrt(3)) < 1e-6 and len(sols) == int(md_answer(3, "iii"))
    report(3, ok1 and ok2 and ok3, f"(i) sin={sp.nsimplify(sin_t)} (ii) a_5^j={det} sum={tot} "
           f"(iii) alpha={alpha:.6f} t={[round(v, 5) for v in sols]}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"예외 {type(e).__name__}: {e}")
    print(f"SUMMARY {PREFIX}: {sum(RESULTS)}/{len(RESULTS)} PASS")
    sys.exit(0 if all(RESULTS) else 1)
