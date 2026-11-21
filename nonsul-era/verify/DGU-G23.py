#!/usr/bin/env python3
"""DGU-G23 (2023학년도 동국대 수시 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-G23"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def reflect(dvec, tvec):
    """방향 dvec 를 접선 방향 tvec 에 대해 반사(입사각=반사각)."""
    tv = tvec / sp.sqrt(tvec.dot(tvec))
    return sp.simplify(2 * dvec.dot(tv) * tv - dvec)


def acute_with_xaxis(vec):
    return sp.nsimplify(sp.acos(sp.Abs(vec[0]) / sp.sqrt(vec.dot(vec))))


def p01():
    """포물선 반사 → 쌍곡선 위 P, 접선 l 의 x절편·γ, 반사 후 δ (답 √6, π/6, π/3)."""
    X, Y = sp.symbols("X Y", real=True)
    F = sp.Matrix([1, 0]); G = sp.Matrix([2, 2 * sp.sqrt(2)])
    par = Y**2 - 4 * X
    tG = sp.Matrix([-sp.diff(par, Y), sp.diff(par, X)]).subs({X: 2, Y: G[1]})   # 접선 방향(기울기 벡터 수직)
    d1 = reflect(G - F, tG)                                                      # G 에서 반사 후 방향
    s = sp.symbols("s", positive=True)
    hyp = X**2 / 18 - Y**2 / 4 - 1
    cand = sp.solve(hyp.subs({X: G[0] + s * d1[0], Y: G[1] + s * d1[1]}), s)
    Ps = [sp.Matrix([sp.simplify(G[0] + c * d1[0]), sp.simplify(G[1] + c * d1[1])]) for c in cand]
    Ps = [q for q in Ps if 3 * sp.sqrt(2) <= q[0] <= 9]
    P = Ps[0]
    tl = sp.Matrix([-sp.diff(hyp, Y), sp.diff(hyp, X)]).subs({X: P[0], Y: P[1]})
    nrm = sp.Matrix([sp.diff(hyp, X), sp.diff(hyp, Y)]).subs({X: P[0], Y: P[1]})
    xint = sp.simplify(P[0] + P[1] * nrm[1] / nrm[0])          # 접선 n·(Z-P)=0 에서 Y=0
    gam = acute_with_xaxis(tl)
    d2 = reflect(d1, tl)
    dlt = acute_with_xaxis(d2)
    ok = (P == sp.Matrix([3 * sp.sqrt(6), 2 * sp.sqrt(2)]) and sp.simplify(xint - sp.sqrt(6)) == 0
          and sp.simplify(gam - sp.pi / 6) == 0 and sp.simplify(dlt - sp.pi / 3) == 0 and d2[1] > 0)
    report(1, ok, f"P=({P[0]},{P[1]}), (1) x절편={xint} (2) γ={gam} (3) 반사 방향 {list(d2)} → δ={dlt}")


def p02():
    """P(X=x_i)=g(x_i)/Σg, g=e^{-ax}, x_i=i/n 의 lim E, lim V."""
    a = sp.symbols("a", positive=True)
    g = sp.exp(-a * x)
    I0 = sp.integrate(g, (x, 0, 1)); I1 = sp.integrate(x * g, (x, 0, 1)); I2 = sp.integrate(x**2 * g, (x, 0, 1))
    E = sp.simplify(I1 / I0); V = sp.simplify(I2 / I0 - E**2)
    E_ans = 1 / a - sp.exp(-a) / (1 - sp.exp(-a))
    V_ans = 1 / a**2 - sp.exp(-a) / (1 - sp.exp(-a))**2
    ok = sp.simplify(E - E_ans) == 0 and sp.simplify(V - V_ans) == 0
    # 유한 n 직접 계산(리만 합 정의 그대로)과 비교
    import mpmath as mp
    for av in (0.5, 1, 3):
        nn = 200000
        w = [mp.e**(-av * i / nn) for i in range(1, nn + 1)]
        S0 = mp.fsum(w); S1 = mp.fsum(i / nn * wi for i, wi in enumerate(w, 1)); S2 = mp.fsum((i / nn)**2 * wi for i, wi in enumerate(w, 1))
        En = S1 / S0; Vn = S2 / S0 - En**2
        ok = ok and abs(En - float(E_ans.subs(a, av))) < 1e-4 and abs(Vn - float(V_ans.subs(a, av))) < 1e-4
    report(2, ok, f"lim E={sp.simplify(E)}, lim V={sp.simplify(V)} (유한 n=2·10^5 수치 일치)")


def p03():
    """튀는 물체: (1) lim t_n (2) 곡선과 x축 사이 넓이 (3) 정사각형 단면 부피."""
    a, b, g = sp.symbols("a b g", positive=True)
    k = sp.symbols("k", integer=True, nonnegative=True)
    s = sp.symbols("s", nonnegative=True)
    vy = b / 2**k   # k번째 아치(0부터) 시작 y속도: 충돌 직전 -v → 직후 (-1/2)(-v)=v/2 (아래 시뮬레이션이 규칙 그대로 재확인)
    yk = vy * s - g * s**2 / 2                             # 아치 시작 후 경과시간 s
    dur = [r for r in sp.solve(sp.Eq(yk, 0), s) if r != 0][0]
    lim_t = sp.summation(dur, (k, 0, sp.oo))
    # 넓이·부피: x = a·(시작 시각 + s) 이므로 dx = a ds
    Ak = sp.integrate(yk * a, (s, 0, dur))
    Vk = sp.integrate(yk**2 * a, (s, 0, dur))
    A = sp.simplify(sp.summation(sp.simplify(Ak), (k, 0, sp.oo)))
    V = sp.simplify(sp.summation(sp.simplify(Vk), (k, 0, sp.oo)))
    # 직접 시뮬레이션(충돌 규칙 그대로, 30회)으로 교차 확인
    av, bv, gv = 2.0, 3.0, 9.8
    tcur, vcur, An, Vn = 0.0, bv, 0.0, 0.0
    for _ in range(60):
        T = 2 * vcur / gv
        An += av * (vcur * T**2 / 2 - gv * T**3 / 6)
        Vn += av * (vcur**2 * T**3 / 3 - vcur * gv * T**4 / 4 + gv**2 * T**5 / 20)
        tcur += T
        vcur = -0.5 * (-vcur)
    sub = {a: av, b: bv, g: gv}
    ok = (sp.simplify(lim_t - 4 * b / g) == 0 and sp.simplify(A - 16 * a * b**3 / (21 * g**2)) == 0
          and sp.simplify(V - 128 * a * b**5 / (465 * g**3)) == 0
          and abs(tcur - float(lim_t.subs(sub))) < 1e-9 and abs(An - float(A.subs(sub))) < 1e-9 and abs(Vn - float(V.subs(sub))) < 1e-9)
    report(3, ok, f"(1) {lim_t} (2) {A} (3) {V} (충돌 60회 수치 시뮬레이션 일치)")




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
