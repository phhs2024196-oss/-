#!/usr/bin/env python3
"""SJU-M23 (세종대 2023학년도 논술 모의평가) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-M23"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.strip().split(" = ", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    y, t, H = sp.symbols("y t H", positive=True)
    Vh = sp.pi * sp.integrate(20 * y - y**2, (y, 0, H))        # 높이 H 까지의 부피
    t9 = sp.solve(sp.Eq(81 * sp.pi * t, Vh.subs(H, 9)), t)[0]
    ok1 = t9 == sp.sympify(md_answer(1, "1"))
    # h(t): 81πt = V(h) 의 음함수 → h'(t), h''(t) 를 직접 미분(수치: t 의 함수 h 를 근으로 정의)
    hfun = lambda tv: mp.findroot(lambda Hv: mp.pi * (10 * Hv**2 - Hv**3 / 3) - 81 * mp.pi * tv, 9)
    h1 = mp.diff(hfun, mp.mpf(t9))
    h2 = mp.diff(hfun, mp.mpf(t9), 2)
    ok2 = abs(h1 - mp.mpf(sp.sympify(md_answer(1, "2")))) < 1e-15
    # t(y) = V(y)/(81π) 은 y 의 다항식 → d2t/dy2 기호 계산
    ty = Vh.subs(H, y) / (81 * sp.pi)
    d2t = sp.diff(ty, y, 2).subs(y, 9)
    ans = parts(md_answer(1, "3"))
    ok3 = abs(h2 - mp.mpf(sp.sympify(ans["d2y"]))) < 1e-12 and sp.simplify(d2t - sp.sympify(ans["d2t"])) == 0
    report(1, ok1 and ok2 and ok3, f"t={t9} h'={float(h1):.12f} h''={float(h2):.12f} d2t/dy2={d2t}")


def p02():
    t, X, Y = sp.symbols("t X Y", real=True)
    x = 6 * t**2 + 1; y = t**3 - 12 * t
    # (1) t>=0 에서 y 최소 (격자 + 정밀)
    ys = sp.lambdify(t, y)
    tg = np.linspace(0, 10, 100001)
    tm = tg[np.argmin(ys(tg))]
    tm = float(sp.nsolve(sp.diff(y, t), t, tm))
    Q = (x.subs(t, tm), y.subs(t, tm))
    q_md = sp.sympify(md_answer(2, "1"))
    ok1 = abs(Q[0] - q_md[0]) < 1e-9 and abs(Q[1] - q_md[1]) < 1e-9
    # (2) 길이 13 인 점 (수치 적분 + 근)
    sp_ = sp.lambdify(t, sp.sqrt(sp.diff(x, t)**2 + sp.diff(y, t)**2), "mpmath")
    L = lambda a: mp.quad(sp_, [0, a]) - 13
    a = mp.findroot(L, 1.5)
    P = (x.subs(t, a), y.subs(t, a)); m1 = (sp.diff(y, t) / sp.diff(x, t)).subs(t, a)
    eq_md = sp.sympify(md_answer(2, "2").split("=")[0], locals={"x": X, "y": Y})
    ok2 = abs(eq_md.subs({X: P[0], Y: P[1]})) < 1e-12 and abs(-eq_md.coeff(X) / eq_md.coeff(Y) - m1) < 1e-12
    # (3) A, Q 고정. L2 를 Q 를 지나는 기울기 m 의 직선으로 두고 AB=AD 를 만족하는 m 을 모두 수치로 찾음
    A = sp.solve(eq_md.subs(Y, 0), X)[0]
    def gap(m):
        Dx = Q[0] - Q[1] / m
        # B: L1 과 L2 교점
        c1, c2, c0 = float(eq_md.coeff(X)), float(eq_md.coeff(Y)), float(eq_md.subs({X: 0, Y: 0}))
        qx, qy = float(Q[0]), float(Q[1])
        Bx = -(c0 + c2 * (qy - m * qx)) / (c1 + c2 * m)   # 평행(m=-c1/c2)은 격자에서 제외
        By = qy + m * (Bx - qx)
        Ax = float(A)
        return float(np.hypot(Bx - Ax, By) - abs(Dx - Ax)), float(Dx), float(Bx), float(By)
    sols = []
    ms = np.concatenate([np.linspace(-20.013, -0.05, 4000), np.linspace(0.05, 20, 4000)])
    ms = ms[np.abs(ms + 0.75) > 0.02]
    vals = [gap(m)[0] for m in ms]
    for i in range(len(ms) - 1):
        if vals[i] * vals[i + 1] < 0 and abs(ms[i + 1] - ms[i]) < 0.1:
            lo, hi = ms[i], ms[i + 1]
            for _ in range(60):
                mid = (lo + hi) / 2
                if gap(lo)[0] * gap(mid)[0] <= 0: hi = mid
                else: lo = mid
            sols.append(gap((lo + hi) / 2))
    # 그림 조건: D 가 x축 양의 부분, B 가 제4사분면
    fig = [s for s in sols if s[1] > 0 and s[2] > 0 and s[3] < 0]
    D_md = sp.sympify(md_answer(2, "3"))
    ok3 = len(fig) == 1 and abs(fig[0][1] - float(D_md[0])) < 1e-8
    report(2, ok1 and ok2 and ok3, f"Q={Q} a={float(a):.10f} P=({float(P[0]):.6f},{float(P[1]):.6f}) AB=AD sols D_x={[round(s[1], 6) for s in sols]} (그림 조건 → {[round(s[1], 6) for s in fig]})")


def p03():
    xs = np.linspace(-6, 4, 4001)
    def gfun(a, k, X):
        f = lambda u: a * (u - k)**2
        return f(f(X))
    def n_extrema(a, k):
        g = gfun(a, k, xs)
        d = np.diff(g)
        return int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0))
    # (1) 극값 하나 ⇔ a>0 (여러 a,k 에서)
    ok1 = all((n_extrema(a, k) == 1) == (a > 0) for a in (-2, -0.5, -0.1, 0.1, 0.5, 2) for k in (-0.3, -1, -2.5))
    ok1 = ok1 and md_answer(3, "1").strip() == "아래로 볼록"
    # (2) g''>0 everywhere (a>0) & (다) 를 만족하는 k 의 최댓값
    g2ok = all(np.all(np.diff(gfun(a, k, xs), 2) > 0) for a in (0.1, 0.5, 2) for k in (-0.3, -1, -2.5))
    ks = np.linspace(-3, -1e-4, 30000)
    kmax = [max(k for k in ks if gfun(a, k, -1.0) <= gfun(a, k, 0.0) + 1e-15) for a in (0.1, 0.7, 3)]
    ans2 = parts(md_answer(3, "2"))
    ok2 = g2ok and ans2["range"] == "all_real" and all(abs(km - float(sp.sympify(ans2["kmax"]))) < 2e-4 for km in kmax)
    # (3) 각 (a,k) 에서 ∫0^p h = ∫p^2p h 의 양의 해 p 를 격자로 모두 찾음 → 최솟값
    mp.mp.dps = 20
    pmins = []
    for a in (0.3, 0.8):
        for k in (-0.5, -0.8, -1.2):
            h = lambda X: gfun(a, k, gfun(a, k, -X))
            Fp = lambda p: mp.quad(h, [0, p]) - mp.quad(h, [p, 2 * p])
            grid = [0.05 * i for i in range(1, 60)]
            fv = [Fp(p) for p in grid]
            roots = []
            for i in range(len(grid) - 1):
                if fv[i] == 0 or fv[i] * fv[i + 1] < 0:
                    roots.append(float(mp.findroot(Fp, (grid[i], grid[i + 1]), solver="bisect")))
            pmins.append((k, roots))
    ok3a = all(len(r) == 1 and abs(r[0] + k) < 1e-6 for k, r in pmins)    # 해는 p=-k 하나
    pmin = min(r[0] for k, r in pmins)                                       # k≤-1/2 범위 중 최소
    ok3 = ok3a and abs(pmin - float(sp.sympify(md_answer(3, "3")))) < 1e-6
    report(3, ok1 and ok2 and ok3, f"[증명-수치확인] 극값1개⇔a>0, g''>0, kmax~{[round(v, 4) for v in kmax]}, p roots={[(k, [round(x, 6) for x in r]) for k, r in pmins]}")


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
