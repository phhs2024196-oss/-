#!/usr/bin/env python3
"""PNU-G26 (부산대 2026학년도 수시 논술 자연계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML answer 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "PNU-G26"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/PNU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 10명(0=A,1=B,2=C,3=D,4..9)을 연구 4·발표 3·지원 3에 배정하는 모든 경우 전수 열거
    R, P, S = 0, 1, 2
    c1 = c2 = total = 0
    people = range(10)
    for res in itertools.combinations(people, 4):
        rest = [x for x in people if x not in res]
        for pre in itertools.combinations(rest, 3):
            g = {}
            for x in people:
                g[x] = R if x in res else (P if x in pre else S)
            total += 1
            A, B, C, D = g[0], g[1], g[2], g[3]
            if (A == B or A == C) and B != C:
                c1 += 1
            if A == P and B in (R, S) and C != D:
                c2 += 1
    ok = total == 4200 and c1 == int(md_answer(1, "1")) and c2 == int(md_answer(1, "2"))
    report(1, ok, f"전수 {total}가지 중 [1-1] {c1}, [1-2] {c2}")


def p02():
    t, x, k, p = sp.symbols("t x k p", real=True)

    def gfun(f):
        # 법선 x절편: P(t,f(t))를 지나고 기울기 -1/f'(t) 인 직선에서 y=0 (f'(t)=0 이면 x=t)
        X = sp.symbols("X")
        fp = sp.diff(f, x).subs(x, t)
        ft = f.subs(x, t)
        sol = sp.solve(sp.Eq(-ft, -1 / fp * (X - t)), X)[0]
        return sp.expand(sp.simplify(sol))

    # (2-1) 자연수 p 를 1부터 늘려 가며 n(A_k)=2 인 k 가 존재하는 최소 p
    p1 = None
    for pv in range(1, 20):
        f = x**2 + pv * x + sp.Rational(pv, 2) - sp.Rational(1, 2)
        g = gfun(f)
        crit = [r for r in sp.solve(sp.diff(g, t), t) if r.is_real]
        if len(set(crit)) == 2:
            p1, g1 = pv, g
            break
    # n(A_k)=2 인 k: g(t)=c 의 서로 다른 실근이 2개인 c 를 찾고, 그 근 전부가 k
    ks = set()
    for r in sp.solve(sp.diff(g1, t), t):
        c = g1.subs(t, r)
        roots = set(sp.Poly(g1 - c, t).real_roots())
        if len(roots) == 2:
            ks |= roots
    # 격자 확인: 다른 k 에서는 n(A_k)=2 가 아님
    import numpy as np
    coeffs = [float(cf) for cf in sp.Poly(g1, t).all_coeffs()]
    bad = 0
    for kv in np.linspace(-4, 2, 601):
        if any(abs(kv - float(kk)) < 1e-9 for kk in ks):
            continue
        cc = coeffs[:]; cc[-1] -= np.polyval(coeffs, kv)
        rr = np.roots(cc)
        nreal = len({round(z.real, 6) for z in rr if abs(z.imag) < 1e-7})
        if nreal == 2:
            bad += 1
    parts = dict(seg.split("=", 1) for seg in md_answer(2, "1").replace(" ", "").split(";"))
    ok1 = p1 == int(parts["p1"]) and sp.nsimplify(sum(ks)) == sp.sympify(parts["sum"]) and bad == 0
    # (2-2) f = x^2+x-7 : k 격자에서 A_k 의 실근 전수 → m_k<k<M_k 인 k 의 M^2+m^2 최대
    g2 = gfun(x**2 + x - 7)
    c2 = [float(cf) for cf in sp.Poly(g2, t).all_coeffs()]
    best, bestk, krange = -1e9, None, []
    for kv in np.linspace(-3, 2, 50001):
        cc = c2[:]; cc[-1] -= np.polyval(c2, kv)
        rr = [z.real for z in np.roots(cc) if abs(z.imag) < 1e-6]
        M, m = max(rr), min(rr)
        if m < kv - 1e-9 and kv + 1e-9 < M:
            krange.append(kv)
            v = M**2 + m**2
            if v > best:
                best, bestk = v, kv
    ok2 = abs(best - float(sp.sympify(md_answer(2, "2")))) < 1e-6 and abs(min(krange) + 2) < 1e-3 and abs(max(krange) - 1) < 1e-3
    report(2, ok1 and ok2,
           f"g={g1}, p1={p1}, k={sorted(ks)}; (2-2) k범위~({min(krange):.4f},{max(krange):.4f}) max={best:.6f} at k={bestk:.4f}")


def T_geom(n, k):
    """사각형 A_{k-1}A_kB_kB_{k-1} 안의, 변 A_{k-1}A_k 를 한 변으로 하는 직사각형의 최대 넓이를
    좌표·반평면 판정 + 이분법으로 직접 구한다(공식 해설 식을 쓰지 않음)."""
    a0, a1 = (k - 1) * math.pi / (3 * n), k * math.pi / (3 * n)
    P0, P1 = (math.cos(a0), math.sin(a0)), (math.cos(a1), math.sin(a1))
    Q0, Q1 = (2, 2 * math.tan(a0)), (2, 2 * math.tan(a1))
    poly = [P0, P1, Q1, Q0]                     # 볼록 사각형(반시계)
    ex, ey = P1[0] - P0[0], P1[1] - P0[1]
    L = math.hypot(ex, ey)
    ux, uy = ey / L, -ex / L                    # 원점 반대쪽 단위 법선
    if ux * P0[0] + uy * P0[1] < 0:
        ux, uy = -ux, -uy

    def inside(pt):
        sgn = None
        for i in range(4):
            A, B = poly[i], poly[(i + 1) % 4]
            cr = (B[0] - A[0]) * (pt[1] - A[1]) - (B[1] - A[1]) * (pt[0] - A[0])
            if abs(cr) < 1e-13:
                continue
            s = cr > 0
            if sgn is None:
                sgn = s
            elif s != sgn:
                return False
        return True

    lo, hi = 0.0, 5.0
    for _ in range(80):
        h = (lo + hi) / 2
        if inside((P0[0] + h * ux, P0[1] + h * uy)) and inside((P1[0] + h * ux, P1[1] + h * uy)):
            lo = h
        else:
            hi = h
    return L * lo


def p03():
    # (3-1)
    s1 = T_geom(1, 1)
    ok1 = abs(s1 - float(sp.sympify(md_answer(3, "1")))) < 1e-10
    # (3-2) 극한값: 기호 적분 + 큰 n 에서 S_n 수렴 확인
    X = sp.symbols("X")
    lim_sym = 2 * sp.integrate(sp.sec(X), (X, 0, sp.pi / 3)) - sp.pi / 3
    target = float(sp.sympify(md_answer(3, "2")))
    okc = abs(float(lim_sym) - target) < 1e-12
    S = {n: sum(T_geom(n, kk) for kk in range(1, n + 1)) for n in (50, 100, 200, 400)}
    errs = [abs(S[n] - target) for n in (50, 100, 200, 400)]
    conv = errs[-1] < 2e-3 and all(errs[i + 1] < errs[i] for i in range(3))
    # 공식 해설 T_k 식(덧셈정리)이 기하 계산과 일치하는지 [증명-수치확인]
    formula_ok = True
    for n in (1, 3, 7, 20):
        d = math.pi / (6 * n)
        for kk in range(1, n + 1):
            th = (2 * kk - 1) * d
            Tf = 2 * math.sin(d) * (2 / math.cos(th) - math.cos(d) - math.sin(d) * math.tan(th))
            formula_ok &= abs(Tf - T_geom(n, kk)) < 1e-9
    # 샌드위치 부등식 ⑤ 확인
    sand = True
    for n in (1, 5, 30, 200):
        d = math.pi / (6 * n)
        lo = sum(1 / math.cos((2 * kk - 2) * d) for kk in range(1, n + 1))
        mid = sum(1 / math.cos((2 * kk - 1) * d) for kk in range(1, n + 1))
        up = sum(1 / math.cos(2 * kk * d) for kk in range(1, n + 1))
        sand &= lo <= mid <= up
    ok2 = okc and conv and formula_ok and sand
    report(3, ok1 and ok2,
           f"S1={s1:.12f}; lim={float(lim_sym):.10f}; S_400={S[400]:.6f} (오차 {errs[-1]:.2e}); T_k식·⑤ [증명-수치확인]")


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
