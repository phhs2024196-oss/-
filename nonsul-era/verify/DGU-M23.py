#!/usr/bin/env python3
"""DGU-M23 (2023학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M23"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """원에 내접하는 ABCD: 둘레 22, ∠A=60°, BD=8 → CB+CD 범위, 넓이 최대일 때 CB+CD."""
    import numpy as np
    X = sp.symbols("X", positive=True)
    s1 = 22 - X                                          # AB+AD
    # 코사인법칙: AB^2+AD^2-AB·AD=64 → (AB+AD)^2-3AB·AD=64 ; ∠C=120° → (CB+CD)^2-CB·CD=64
    p1 = (s1**2 - 64) / 3
    p2 = X**2 - 64
    conds = [s1 > 0, p1 > 0, s1**2 - 4 * p1 >= 0, p2 > 0, X**2 - 4 * p2 >= 0]
    rng = sp.reduce_inequalities(conds, X)
    S = sp.sqrt(3) / 4 * (p1 + p2)                       # 넓이 = ½ab sin60° + ½cd sin120°
    lo, hi = sp.Integer(8), 16 * sp.sqrt(3) / 3
    crit = [c for c in sp.solve(sp.diff(S, X), X) if lo < c <= hi]
    best = max(crit + [hi], key=lambda c: float(S.subs(X, c)))
    # 수치: 변 길이 (a,b,c,d) 를 직접 만들어 존재하는 x 범위 확인
    xs = np.linspace(0.01, 21.99, 220001)
    ok_x = []
    for xv in xs[::50]:
        sv = 22 - xv; P1 = (sv**2 - 64) / 3; P2 = xv**2 - 64
        if sv > 0 and P1 > 0 and P2 > 0 and sv**2 - 4 * P1 >= 0 and xv**2 - 4 * P2 >= 0:
            ok_x.append(xv)
    ok = (rng == sp.And(sp.Lt(8, X), sp.Le(X, hi)) or (sp.simplify(rng.as_set().inf - 8) == 0 and sp.simplify(rng.as_set().sup - hi) == 0 and rng.as_set().left_open and not rng.as_set().right_open))
    ok = ok and abs(min(ok_x) - 8) < 0.01 and abs(max(ok_x) - float(hi)) < 0.01 and sp.simplify(best - hi) == 0
    report(1, ok, f"CB+CD 범위: {rng}, S(x)={sp.expand(S)}, 넓이 최대 x={best}, S_max={sp.nsimplify(sp.simplify(S.subs(X, best)))}")


def p02():
    """y>=0 속력 1, y<0 속력 2, A(0,√7)→B(2,-1) 최소 시간 (답 5√2/2)."""
    import numpy as np
    X = sp.symbols("X", real=True)
    T = sp.sqrt(X**2 + 7) + sp.sqrt((2 - X)**2 + 1) / 2
    crit = sp.solve(sp.diff(T, X), X)
    real = [c for c in crit if c.is_real]
    tmin = min((sp.nsimplify(sp.simplify(T.subs(X, c))) for c in real), key=float)
    xs = np.linspace(-20, 20, 4000001)
    Tn = np.sqrt(xs**2 + 7) + np.sqrt((2 - xs)**2 + 1) / 2
    # 대표 오답: 선분 AB(최단 거리) 경로
    xab = sp.solve(sp.Eq(sp.sqrt(7) + (-1 - sp.sqrt(7)) * X / 2, 0), X)[0]
    wrong = T.subs(X, xab)
    ok = sp.simplify(tmin - 5 * sp.sqrt(2) / 2) == 0 and abs(Tn.min() - float(tmin)) < 1e-9 and abs(xs[Tn.argmin()] - 1) < 1e-4
    report(2, ok, f"임계점 {real}, 최소 시간 {tmin} (수치 {Tn.min():.10f}); 선분 AB 경로 시간 {float(wrong):.6f}")


def p03():
    """x=2cos²t, y=5c/4 - c sin t: 만날 때마다 속도 일치 → c, 거리 처음 최대 시각, Q의 속도·가속도."""
    c = sp.symbols("c", positive=True)
    tt = sp.symbols("t", real=True)
    xP = 2 * sp.cos(tt)**2; yQ = sp.Rational(5, 4) * c - c * sp.sin(tt)
    # 조건: 같은 시각에 위치·속도 모두 같은 t 가 존재. s=sin t, cos t 로 분리
    sv, cv = sp.symbols("s co", real=True)
    pos = sp.expand((xP - yQ).subs({sp.cos(tt): cv, sp.sin(tt): sv}))
    vel = sp.expand((sp.diff(xP, tt) - sp.diff(yQ, tt)).subs({sp.cos(tt): cv, sp.sin(tt): sv}))
    sols = sp.solve([pos, vel, sv**2 + cv**2 - 1], [c, sv, cv], dict=True)
    cs = sorted({sp.nsimplify(d[c]) for d in sols if d[c].is_positive and d[sv].is_real and d[cv].is_real})
    # 모든 만남에서 속도 일치하는지 확인(만남 = pos=0 인 모든 t)
    good = []
    for cc in cs:
        meet = sp.solve(pos.subs(c, cc).subs(cv**2, 1 - sv**2), sv)
        meet = [m for m in meet if m.is_real and -1 <= m <= 1]
        if all(sp.simplify(vel.subs(c, cc).subs(sv, m)) == 0 for m in meet):   # vel 은 co·(…) 꼴
            good.append(cc)
    cc = good[0]
    D = sp.simplify(sp.Abs(xP - yQ.subs(c, cc)))
    import numpy as np
    ts = np.linspace(0, 4 * np.pi, 4000001)
    Dn = np.abs(2 * np.cos(ts)**2 - (1.25 * float(cc) - float(cc) * np.sin(ts)))
    Dmax = Dn.max()
    tfirst = ts[np.argmax(Dn > Dmax - 1e-9)]
    t_star = sp.nsimplify(round(tfirst / np.pi * 2) / 2) * sp.pi
    vQ = sp.diff(yQ.subs(c, cc), tt).subs(tt, t_star)
    aQ = sp.diff(yQ.subs(c, cc), tt, 2).subs(tt, t_star)
    # 대표 오답: 첫 임계점 중 극대 t=π/2 에서의 거리
    ok = (cc == 2 and sp.simplify(D.subs(tt, t_star) - sp.Rational(9, 2)) == 0 and t_star == 3 * sp.pi / 2
          and abs(tfirst - 3 * np.pi / 2) < 1e-3 and vQ == 0 and aQ == -2)
    report(3, ok, f"후보 c={cs} → 조건 만족 c={cc}; D(t)={D}, 최대 {sp.Rational(9,2)} 처음 t={t_star}; Q 속도 {vQ}, 가속도 {aQ} (t=π/2 극대 D={D.subs(tt, sp.pi/2)})")




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
