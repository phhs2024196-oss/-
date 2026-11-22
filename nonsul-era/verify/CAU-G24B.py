#!/usr/bin/env python3
"""CAU-G24B (2024학년도 중앙대 수시 논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G24B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """공 꺼내는 순서·내분/외분 전수 → P(|A2P| < |x1|) (답 1/3)."""
    Rt = sp.Rational
    def internal(P1, P2, a, b):
        return tuple((a * q + b * p) / (a + b) for p, q in zip(P1, P2))
    def external(P1, P2, a, b):
        return tuple((a * q - b * p) / (a - b) for p, q in zip(P1, P2))
    O, Pp = (Rt(0), Rt(0)), (Rt(12), Rt(0))
    prob = Rt(0)
    for m_, n_ in ((2, 3), (3, 2)):
        for A1, p1 in ((internal(O, Pp, m_, 1), Rt(1, m_)), (external(O, Pp, 1, m_), 1 - Rt(1, m_))):
            Q = (A1[0], Rt(8))
            opts = [(internal(A1, Q, 1, n_), Rt(2, n_))]
            if n_ != 1:
                opts.append((external(A1, Q, n_, 1), 1 - Rt(2, n_)))
            for A2, p2 in opts:
                if p2 == 0:
                    continue
                d2 = (A2[0] - 12)**2 + A2[1]**2
                if d2 < A2[0]**2:
                    prob += Rt(1, 2) * p1 * p2
    report(1, prob == Rt(1, 3), f"P={prob}")


def p02():
    """(1) f(x)=∫_{-1}^x … 를 수치적분으로 [-1,5] 격자 최댓값 (답 32/3). (2) 속력 수치적분 (답 2(3ln2-1)/√3)."""
    import mpmath as mp
    mp.mp.dps = 25
    g = lambda T: (T**2 + 5) * (1 - 2 * mp.sin(mp.pi * T / (T**2 + 5)))
    F = lambda X: mp.quad(g, [-1, X])
    grid = [mp.mpf(-1) + mp.mpf(6) * i / 600 for i in range(601)]
    vals = [F(X) for X in grid]
    i = max(range(len(vals)), key=lambda j: vals[j])
    xm = mp.findroot(g, grid[i])
    fmax = F(xm)
    ok1 = abs(xm - 1) < 1e-15 and abs(fmax - mp.mpf(32) / 3) < 1e-18 and max(vals) <= fmax + 1e-18
    # (2)
    xt = lambda T: mp.cos(T) / mp.sqrt(3)
    yt = lambda T: mp.sqrt(3) * mp.log(1 + mp.sin(T)**2 / 8)
    sp_ = lambda T: mp.sqrt(mp.diff(xt, T)**2 + mp.diff(yt, T)**2)
    L = mp.quad(sp_, [0, mp.pi / 2, mp.pi])
    target = 2 * (3 * mp.log(2) - 1) / mp.sqrt(3)
    ok2 = abs(L - target) < 1e-15
    report(2, ok1 and ok2, f"(1) 최댓점 x={mp.nstr(xm, 12)}, 최댓값={mp.nstr(fmax, 15)}  (2) 거리={mp.nstr(L, 15)}")


def p03():
    """(1) a_n 기호적분 후 Σ 정확 계산 (답 19899/10000). (2) AP+BP 최솟값 m, m² (답 98)."""
    k = sp.symbols("k", integer=True, positive=True)
    npos = sp.symbols("npos", positive=True)
    an = sp.expand(sp.simplify(sp.integrate(2 * sp.exp(2 * x) - sp.exp(x), (x, 0, sp.log(npos + 1)))))
    S = sum((sp.Rational(1, kk) + sp.Rational(1, kk + 1) + 1) / an.subs(npos, kk) for kk in range(1, 100))
    ok1 = sp.simplify(S - sp.Rational(19899, 10000)) == 0 and sp.expand(an - npos * (npos + 1)) == 0
    th = sp.symbols("th", real=True)
    P = sp.Matrix([7 * sp.cos(th), 7 * sp.sin(th)])
    f = (P - sp.Matrix([5, 0])).norm() + (P - sp.Matrix([0, 5])).norm()
    import numpy as np
    fn = sp.lambdify(th, f, "numpy")
    ts = np.linspace(0, np.pi / 2, 200001)
    vals = fn(ts)
    j = int(np.argmin(vals))
    thm = sp.nsolve(sp.diff(f, th), th, float(ts[j]), prec=30)      # 격자 최소점 근방의 임계점
    m2 = float(f.subs(th, thm))**2
    # 최소점이 sinθ=3/5(또는 4/5) 인지: 그 점에서 f² 를 정확히 계산
    ex = sp.simplify(f.subs(th, sp.asin(sp.Rational(3, 5)))**2)
    ok_pt = min(abs(sp.sin(thm) - sp.Rational(3, 5)), abs(sp.sin(thm) - sp.Rational(4, 5))) < 1e-20
    ok2 = abs(m2 - 98) < 1e-9 and sp.simplify(ex - 98) == 0 and ok_pt and vals.min() >= m2**0.5 - 1e-9
    report(3, ok1 and ok2, f"(1) a_n={an}, 합={S}  (2) m²≈{m2:.10f}, sinθ*={sp.N(sp.sin(thm), 8)}, f²(arcsin 3/5)={ex}")


def p04():
    """(1) P(k,√3) 에서 그은 두 접선의 각이 45°인 k² (답 4+2√5).
    (2) 3차원 좌표로 삼각형 ABC 구성 → 내접원 그림자 넓이 (답 2(11-4√7)π)."""
    import numpy as np
    m, kk = sp.symbols("m kk", real=True)
    def tangent_angle(kv):
        # y - √3 = m(x - k) 가 x^2 + y^2/2 = 1 에 접할 조건: 판별식 0
        X = sp.symbols("X")
        q = sp.expand(2 * X**2 + (m * (X - kv) + sp.sqrt(3))**2 - 2)
        disc = sp.discriminant(q, X)
        ms = sp.solve(disc, m)
        if len(ms) != 2:
            return None
        a1, a2 = [sp.atan(s) for s in ms]
        ang = sp.Abs(a1 - a2)
        ang = sp.Min(ang, sp.pi - ang)
        return sp.N(ang, 30)
    k2 = 4 + 2 * sp.sqrt(5)
    ang = tangent_angle(sp.sqrt(k2))
    ok_a = abs(ang - sp.pi / 4) < 1e-25
    # k² 의 유일성: k>0 범위 스캔(대칭이므로 k≥0)
    hits = []
    prev = None
    for kv in np.linspace(0.05, 6, 600):
        a = tangent_angle(sp.Float(kv))
        if a is None:
            prev = None; continue
        val = float(a) - np.pi / 4
        if prev is not None and prev * val < 0:
            hits.append(kv)
        prev = val
    ok1 = ok_a and len(hits) == 1 and abs(hits[0]**2 - float(k2)) < 0.1
    # (2) 그림자 A'B'C' 정삼각형(한 변 a, z=0), 높이 AA'=5-d, BB'=5, CC'=5+d
    a, d = sp.symbols("a d", real=True)
    Ap = sp.Matrix([0, 0, 0]); Bp = sp.Matrix([a / 2, a * sp.sqrt(3) / 2, 0]); Cp = sp.Matrix([a, 0, 0])
    A = Ap + sp.Matrix([0, 0, 5 - d]); B = Bp + sp.Matrix([0, 0, 5]); C = Cp + sp.Matrix([0, 0, 5 + d])
    eqs = [sp.Eq((C - A).dot(C - A), 48), sp.Eq(((B - A).cross(C - A)).dot((B - A).cross(C - A)), 4 * 108)]
    sols = [s for s in sp.solve(eqs, [a, d], dict=True) if s[a].is_positive]
    shadows = set()
    for s in sols:
        AA, BB, CC = [P.subs(s) for P in (A, B, C)]
        la, lb, lc = (BB - CC).norm(), (CC - AA).norm(), (AA - BB).norm()
        S = ((BB - AA).cross(CC - AA)).norm() / 2
        r = 2 * S / (la + lb + lc)
        Sp = (sp.sqrt(3) / 4) * s[a]**2
        shadows.add(sp.radsimp(sp.simplify(sp.pi * r**2 * Sp / S)))
    ok2 = len(sols) >= 1 and all(sp.simplify(v - 2 * (11 - 4 * sp.sqrt(7)) * sp.pi) == 0 for v in shadows)
    report(4, ok1 and ok2, f"(1) 각={sp.N(ang * 180 / sp.pi, 12)}°, 해 개수(k>0)={len(hits)}  (2) 해 {len(sols)}개, 그림자={shadows}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
