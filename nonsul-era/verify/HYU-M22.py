#!/usr/bin/env python3
"""HYU-M22 (한양대 2022학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HYU-M22"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    n, k = sp.symbols("n k", positive=True, integer=True)
    # 1. 적분 비교 부등식을 여러 n 에서 확인하고, 양 끝 극한(조임)을 sympy 로 계산
    import math
    lo = lambda N: 2 * (math.sqrt(N + 1) - 1)
    hi = lambda N: 1 + 2 * (math.sqrt(N) - 1)
    okb = all(lo(N) <= sum(1 / math.sqrt(j) for j in range(1, N + 1)) <= hi(N) for N in (1, 5, 50, 2000))
    L1 = sp.limit((2 * (sp.sqrt(n + 1) - 1)) / sp.sqrt(n), n, sp.oo)
    L2 = sp.limit((1 + 2 * (sp.sqrt(n) - 1)) / sp.sqrt(n), n, sp.oo)
    ok1 = okb and L1 == L2 and sp.simplify(L1 - sp.sympify(md_answer(1, "1"))) == 0
    # 2. 내부 적분을 직접 적분 → k 합 → 극한
    X = sp.symbols("X", real=True)
    def lim2(c):
        inner = sp.integrate(n * c**4 * (k / n - X), (X, (k - 1) / n, k / n))
        return sp.limit(sp.summation(sp.expand(inner), (k, 1, n)), n, sp.oo)
    A1, A2 = lim2((k - 1) / n), lim2(k / n)
    a2 = kv(md_answer(1, "2"))
    ok2 = sp.simplify(A1 - sp.sympify(a2["first"])) == 0 and sp.simplify(A2 - sp.sympify(a2["second"])) == 0
    # 3. 닫힌 꼴 합 - n/6 의 극한
    S = sp.summation((k / n)**5, (k, 1, n)) - n * sp.integrate(x**5, (x, 0, 1))
    A3 = sp.limit(sp.simplify(S), n, sp.oo)
    ok3 = sp.simplify(A3 - sp.sympify(md_answer(1, "3"))) == 0
    report(1, ok1 and ok2 and ok3, f"L={L1} lims=({A1},{A2}) A3={A3}")


def p02():
    import math
    import numpy as np
    # 1. 타원 x^2/4+y^2=1, F(-√3,0). P(2cosθ, sinθ) 에서 FP 를 접선에 대해 반사한 방향이 FP 에 수직인 θ 를 찾고
    #    반사 직선과 타원의 다른 교점 Q 로 넓이 계산(반사 성질 가정 없이 직접)
    F = np.array([-math.sqrt(3), 0.0])
    def geo(th):
        P = np.array([2 * math.cos(th), math.sin(th)])
        T = np.array([-2 * math.sin(th), math.cos(th)]); T /= np.linalg.norm(T)
        u = P - F; u /= np.linalg.norm(u)                    # 입사 방향
        r = 2 * np.dot(u, T) * T - u                          # 접선 기준 반사
        return P, r, np.dot(-u, r)                            # PF 방향(-u)과 PQ 방향(r) 내적
    from scipy.optimize import brentq
    ths = np.linspace(1e-6, 2 * math.pi - 1e-6, 20001)
    vals = [geo(t)[2] for t in ths]
    areas = set()
    for i in range(len(ths) - 1):
        if vals[i] * vals[i + 1] < 0:
            th = brentq(lambda t: geo(t)[2], ths[i], ths[i + 1], xtol=1e-15)
            P, r, _ = geo(th)
            # P + s r 가 타원 위: (Px+s rx)^2/4 + (Py+s ry)^2 = 1 의 0 아닌 근
            a = r[0]**2 / 4 + r[1]**2; b = P[0] * r[0] / 2 + 2 * P[1] * r[1]
            s = -b / a
            Q = P + s * r
            if s <= 0:
                continue
            ar = abs((P - F)[0] * (Q - F)[1] - (P - F)[1] * (Q - F)[0]) / 2
            areas.add(round(ar, 10))
    md = sorted(float(v) for v in sp.sympify(md_answer(2, "1")))
    ok1 = sorted(areas) == [round(v, 10) for v in md] if len(areas) == 2 else False
    # 2. f = 4x^2+ax+b, f(-1)=0, lim (x+1)/f = -1/7
    a, b = sp.symbols("a b", real=True)
    f = 4 * x**2 + a * x + b
    sol = sp.solve([f.subs(x, -1), sp.limit(sp.cancel((x + 1) / f.subs(b, a - 4)), x, -1) + sp.Rational(1, 7)], [a, b], dict=True)[0]
    fx = f.subs(sol)
    ok_lim = sp.limit(fx / (x**2 + 2 * x + 3), x, sp.oo) == 4
    g = sp.integrate(fx, (x, -t, t))
    cands = [g.subs(t, v) for v in [-2, 2] + [c for c in sp.solve(sp.diff(g, t), t) if -2 <= c <= 2]]
    gmin = min(cands, key=lambda e: float(e))
    ok2 = ok_lim and sp.simplify(gmin - sp.sympify(md_answer(2, "2"))) == 0
    # 3. 속력/가속도 크기
    X, Y = 2 * sp.cos(t), sp.sin(t)
    sp_ = sp.sqrt(sp.diff(X, t)**2 + sp.diff(Y, t)**2)
    ac = sp.sqrt(sp.diff(X, t, 2)**2 + sp.diff(Y, t, 2)**2)
    h = sp.lambdify(t, sp_ / ac)
    ts = np.linspace(0, 2, 400001)
    hv = h(ts)
    a3 = kv(md_answer(2, "3"))
    crit = [0, 2, float(sp.pi / 2)]
    ok3 = abs(max(h(c) for c in crit) - float(sp.sympify(a3["max"]))) < 1e-12 and \
        abs(min(h(c) for c in crit) - float(sp.sympify(a3["min"]))) < 1e-12 and \
        hv.max() <= float(sp.sympify(a3["max"])) + 1e-12 and hv.min() >= float(sp.sympify(a3["min"])) - 1e-12
    report(2, ok1 and ok2 and ok3, f"areas={sorted(areas)} f={sp.expand(fx)} gmin={gmin}")


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
