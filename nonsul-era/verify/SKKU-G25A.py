#!/usr/bin/env python3
"""SKKU-G25A (성균관대 2025학년도 수시 논술우수 수리형 1교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import sys

import numpy as np
import sympy as sp

PREFIX = "SKKU-G25A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_set(no, sub, loc=None):
    v = md_answer(no, sub).strip().strip("{}")
    return [sp.sympify(p, locals=loc or {}) for p in v.split(",")]


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a = sp.symbols("a", positive=True)
    X, Y, r = sp.symbols("X Y r", real=True)
    y = X**2
    m = sp.diff(y, X).subs(X, a)
    Q = (a, a**2)
    T = (sp.solve(m * (X - a) + a**2, X)[0], 0)                       # 접선의 x절편
    S = (sp.solve(-1 / m * (X - a) + a**2, X)[0], 0)                  # 법선의 x절편
    P = (0, (-1 / m * (0 - a) + a**2))                                 # 법선의 y절편
    d = lambda A, B: sp.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    v1 = sp.simplify(d(Q, S) * d(S, T) / d(Q, T)**3)
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
    # 원 C: 중심 R=(h,r) 은 법선 위, x축에 접함(반지름 r=중심 y), Q까지 거리 r — 수치로 a 여러 값
    vals2, vals3 = [], []
    for av in (0.3, 1, 2.7, 10):
        Qn = (av, av**2)
        # 중심은 법선 위: R = Q + s*(2a,-1)/|..| 방향(아래쪽으로, x 증가)
        s = sp.symbols("s", positive=True)
        nx, ny = 2 * av / np.hypot(2 * av, 1), -1 / np.hypot(2 * av, 1)
        sol = sp.nsolve(Qn[1] + s * ny - s, s, 0.5)                     # 중심 y = 반지름 = QR
        rr = float(sol)
        Rn = (Qn[0] + rr * nx, Qn[1] + rr * ny)
        Pn = (0, av**2 + 0.5)
        PQ = np.hypot(Pn[0] - Qn[0], Pn[1] - Qn[1])
        vals2.append(0.5 * PQ + rr - Pn[1])
        assert Rn[0] > 0
    ok2 = all(abs(v - float(sp.sympify(md_answer(1, "2")))) < 1e-9 for v in vals2)
    # (iii) 기호 극한: r 은 r = a^2 - r*ny 방정식에서
    ny_s = -1 / sp.sqrt(4 * a**2 + 1)
    rs = sp.solve(sp.Eq(a**2 + r * ny_s, r), r)[0]
    lim = sp.limit(2 * sp.pi * rs / d(P, Q)**2, a, sp.oo)
    ok3 = sp.simplify(lim - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"ratio={v1} (ii)={vals2[0]:.12f} lim={lim}")


def p02():
    f = sp.Piecewise((t**2 + 2 * t, t > 0), (-t, True))
    a = sp.symbols("a", positive=True)
    sols = sp.solve(sp.Eq(sp.integrate(f, (t, -2, 0)) + sp.integrate(t**2 + 2 * t, (t, 0, a)), (32 - 3 * a) / 3), a)
    ok1 = sols == [sp.sympify(md_answer(2, "1"))]
    I2 = sp.integrate(f * sp.Abs(t - 1), (t, -2, 2))
    ok2 = sp.nsimplify(I2) == sp.sympify(md_answer(2, "2"))
    I3 = sp.integrate(f.subs(t, t**2 - 2) * sp.Abs(t - 1), (t, -2, 2))
    ok3 = sp.simplify(I3 - sp.sympify(md_answer(2, "3"))) == 0
    # (iv) g'(x) 를 미분과 적분의 관계로 만들고 부호 변화 점을 수치로 탐색
    fx = lambda v: v**2 + 2 * v if v > 0 else -v
    gp = lambda v: 2 * v**2 - 4 * v + 12 + (1 - v) * fx(v) - (4 - v) * fx(v - 3)
    xs = np.linspace(-10 + 1e-9, 10 - 1e-9, 2000001)
    gv = np.array([gp(v) for v in xs[::10]])
    xs = xs[::10]
    ext = []
    for i in range(len(xs) - 1):
        if gv[i] * gv[i + 1] < 0 or (gv[i] == 0 and i > 0 and gv[i - 1] * gv[i + 1] < 0):
            ext.append(xs[i])
    # 근 근처 모으기
    pts = []
    for e in ext:
        if not pts or abs(e - pts[-1]) > 1e-3:
            pts.append(e)
    md = sorted(float(v) for v in md_set(2, "4"))
    ok4 = len(pts) == len(md) and all(abs(p - q) < 1e-4 for p, q in zip(pts, md))
    report(2, ok1 and ok2 and ok3 and ok4, f"a={sols} I2={I2} I3={sp.nsimplify(I3)} extrema~{[round(p, 5) for p in pts]}")


def roots_ok(cf):
    """f=4x^3+3a x^2+2b x+c 가 서로 다른 세 실근 & 모든 [n,n+1] 에 근 ≤1 개인지(정확 판정)."""
    a, b, c = cf
    nr = np.roots([4, 3 * a, 2 * b, c])
    if np.any(np.abs(nr.imag) > 1e-6):                                  # 빠른 사전 거름(허근)
        return False, []
    f = 4 * x**3 + 3 * a * x**2 + 2 * b * x + c
    rts = sp.Poly(f, x).real_roots()
    if len(set(rts)) != 3 or sp.discriminant(f, x) == 0:
        return False, rts
    rv = [float(r) for r in rts]
    for n in range(int(np.floor(min(rv))) - 1, int(np.ceil(max(rv))) + 2):
        if sum(1 for r in rts if n <= r <= n + 1) > 1:
            return False, rts
    return True, rts


def p03():
    # (i) a=-1,b=-3, 정수 c 전수
    cnt = sum(1 for c in range(-200, 201) if roots_ok((-1, -3, c))[0])
    ok1 = cnt == int(md_answer(3, "1"))
    # (ii) b=0, 정수 a,c 전수, 두 근이 [0,2]
    vals, fs = set(), []
    for a in range(-20, 21):
        for c in range(-200, 201):
            ok, rts = roots_ok((a, 0, c))
            if ok and sum(1 for r in rts if 0 <= r <= 2) == 2:
                vals.add(sp.nsimplify(sp.simplify(sum(sp.Abs(r) for r in rts)).evalf(40), [sp.sqrt(3), sp.sqrt(33)]))
                fs.append(4 * x**3 + 3 * a * x**2 + c)
    md2 = md_set(3, "2")
    ok2 = len(vals) == len(md2) and all(any(abs(sp.N(v - w, 30)) < 1e-25 for w in vals) for v in md2)
    # (iii) g = ∫f + d, d 정수 전수, 서로 다른 네 실근
    gs = []
    for fpol in fs:
        G = sp.integrate(fpol, x)
        for d in range(-50, 51):
            g = G + d
            if all(cf.is_integer for cf in sp.Poly(g, x).all_coeffs()):
                rr = sp.Poly(g, x).real_roots()
                if len(set(rr)) == 4:
                    gs.append(sp.expand(g))
    md3 = md_set(3, "3", {"x": x})
    ok3 = len(gs) == len(md3) and all(any(sp.expand(g - h) == 0 for h in md3) for g in gs)
    report(3, ok1 and ok2 and ok3, f"count={cnt} sums={sorted(vals, key=float)} g={gs}")


if __name__ == "__main__":
    for name in sorted(n for n in dir() if n.startswith("p") and n[1:].isdigit()):
        globals()[name]()
    n_ok = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {n_ok}/{len(RESULTS)} PASS")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
