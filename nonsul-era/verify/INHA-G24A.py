#!/usr/bin/env python3
"""INHA-G24A (2024학년도 인하대 논술 자연계열 오전 일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp
from scipy import integrate as si
from scipy import optimize as so

PREFIX = "INHA-G24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) b최대=3-3ln3 (1-2) b=-e^2 (1-3) 2a+b 최대=6-2ln2 — 접선 조건을 독립 재계산."""
    a, b = sp.symbols("a b", real=True)
    # (1-1): 3x+b<=e^x  <=> b <= min(e^x-3x)
    h = sp.exp(x) - 3 * x
    xc = sp.solve(sp.diff(h, x), x)[0]
    b1 = sp.simplify(h.subs(x, xc))
    grid = np.linspace(-20, 20, 400001)
    ok1 = sp.simplify(b1 - (3 - 3 * sp.log(3))) == 0 and abs(np.min(np.exp(grid) - 3 * grid) - float(b1)) < 1e-6
    # (1-2): 기울기 a>0 일 때 b<=a-a ln a, 2a+b 최대 → 3a-a ln a 최대
    G = 3 * a - a * sp.log(a)
    ac = sp.solve(sp.diff(G, a), a)[0]
    b2 = sp.simplify((a - a * sp.log(a)).subs(a, ac))
    ok2 = sp.simplify(ac - sp.exp(2)) == 0 and sp.simplify(b2 + sp.exp(2)) == 0
    # (1-3): 두 부등식 → b<=a-a ln a, b<=6-2a-a ln a (a>0), a=0이면 2a+b<=0
    def best(av):
        xs = np.linspace(-40, 40, 160001)
        bmax = min(np.min(np.exp(xs) - av * xs), np.min(np.exp(xs - 3) + 6 - av * xs))
        return 2 * av + bmax
    res = so.minimize_scalar(lambda v: -best(v), bounds=(0.01, 10), method="bounded", options={"xatol": 1e-10})
    val3 = -res.fun
    ok3 = abs(val3 - float(6 - 2 * sp.log(2))) < 1e-5 and abs(res.x - 2) < 1e-3
    report(1, ok1 and ok2 and ok3, f"(1-1) b={b1} (1-2) a={ac}, b={b2} (1-3) 수치최대={val3:.8f} at a={res.x:.5f}")


def p02():
    """(2-1) P(ln t,t), r(t)=t(1-ln t)/sqrt(t^2+1) (2-2) ln a=-a^2 에서 C_a가 원점 통과 (2-3) 넓이."""
    T = sp.symbols("T", positive=True)
    u = sp.symbols("u", real=True)
    # 곡선 위의 점 (u,e^u) 와 직선 tx-y=0 거리 (0<t<e 이면 tu-e^u<0)
    D = (sp.exp(u) - T * u) / sp.sqrt(T**2 + 1)
    uc = sp.solve(sp.diff(D, u), u)[0]
    r = sp.simplify(D.subs(u, uc))
    ok1 = sp.simplify(uc - sp.log(T)) == 0 and sp.simplify(r - T * (1 - sp.log(T)) / sp.sqrt(T**2 + 1)) == 0
    rp = sp.simplify(sp.diff(r, T))
    ok_rp = sp.simplify(rp - (-sp.log(T) - T**2) / ((T**2 + 1) * sp.sqrt(T**2 + 1))) == 0
    # 최대점: 수치 최대화
    rf = sp.lambdify(T, r)
    res = so.minimize_scalar(lambda v: -rf(v), bounds=(1e-6, float(sp.E) - 1e-9), method="bounded", options={"xatol": 1e-12})
    a0 = res.x
    ok2a = abs(np.log(a0) + a0**2) < 1e-7
    cx, cy, rr = np.log(a0), a0, rf(a0)
    ok2b = abs(cx**2 + cy**2 - rr**2) < 1e-9          # 원점 통과
    # (2-3) 원 ∩ {x<0, y>0} 넓이(중심 포함 부분)
    def seg(xv):
        h = np.sqrt(max(rr**2 - (xv - cx) ** 2, 0.0))
        lo, hi = max(cy - h, 0.0), cy + h
        return max(hi - lo, 0.0)
    area, _ = si.quad(seg, cx - rr, 0.0, limit=400, epsabs=1e-13, epsrel=1e-13)
    formula = np.pi / 2 * a0**4 + 2 * a0**3 + np.pi / 2 * a0**2
    ok3 = abs(area - formula) < 1e-8
    report(2, ok1 and ok_rp and ok2a and ok2b and ok3,
           f"r(t)={r}, a={a0:.10f}(ln a+a^2≈0), |P|-r≈{np.hypot(cx, cy) - rr:.1e}, 넓이 수치={area:.10f} vs 식={formula:.10f}")


def p03():
    """(3-1) p=t+sin t/2, q=cos t-sin^2 t/4 (3-2) g'=-√2/2 (3-3) ∫g=π/4. 의예과 [문제 1]과 동일."""
    P, Q = sp.symbols("P Q", real=True)
    sol = sp.solve([(t - P)**2 + Q - sp.cos(t), 2 * (t - P) + sp.sin(t)], [P, Q], dict=True)[0]
    p_t, q_t = sol[P], sol[Q]
    ok1 = sp.simplify(p_t - (t + sp.sin(t) / 2)) == 0 and sp.simplify(q_t - (sp.cos(t) - sp.sin(t)**2 / 4)) == 0
    # 유일 교점 수치 확인: 여러 p 에 대해 (x-p)^2+q-cos x >=0, 0 은 한 점
    pf, qf = sp.lambdify(t, p_t), sp.lambdify(t, q_t)
    xs = np.linspace(-15, 25, 400001)
    okU = True
    for tv in np.linspace(0.05, 2 * np.pi - 0.05, 37):
        hv = (xs - pf(tv))**2 + qf(tv) - np.cos(xs)
        okU &= hv.min() > -1e-9 and np.sum(hv < 1e-6) < 400
    # (3-2)
    t0 = sp.nsolve(p_t - (sp.pi / 4 + sp.sqrt(2) / 4), t, 0.8)
    gp = sp.diff(q_t, t) / sp.diff(p_t, t)
    val2 = sp.simplify(gp.subs(t, sp.pi / 4))
    ok2 = abs(float(t0) - float(sp.pi / 4)) < 1e-12 and sp.simplify(val2 + sp.sqrt(2) / 2) == 0
    # (3-3) p: [0,2π] ↔ t: [0,2π]
    I = sp.integrate(sp.expand(q_t * sp.diff(p_t, t)), (t, 0, 2 * sp.pi))
    ok3 = sp.simplify(I - sp.pi / 4) == 0 and p_t.subs(t, 0) == 0 and sp.simplify(p_t.subs(t, 2 * sp.pi) - 2 * sp.pi) == 0
    report(3, ok1 and okU and ok2 and ok3, f"p={p_t}, q={q_t}, g'={val2}, ∫g={sp.simplify(I)}")


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
