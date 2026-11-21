#!/usr/bin/env python3
"""DGU-M25 (2025학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1) g(π/2)=-4/3 (2) 최솟값 -(3π+14)/8, 최댓값 (π-14)/16 (3) -4π."""
    import numpy as np
    X, T = sp.symbols("X T", real=True)
    f = sp.sin(X)**2 + 4 * X + 3
    fp = sp.diff(f, X)
    # 두 접선: y=f'(T)(x-T)+f(T),  y=(x-f(T))/f'(T)+T  (역함수 미분법)
    xx = sp.symbols("xx")
    g = sp.solve(sp.Eq(fp.subs(X, T) * (xx - T) + f.subs(X, T), (xx - f.subs(X, T)) / fp.subs(X, T) + T), xx)[0]
    g = sp.simplify(g)
    ok1 = sp.simplify(g.subs(T, sp.pi / 2) + sp.Rational(4, 3)) == 0
    gn = sp.lambdify(T, g, "numpy")
    grid = np.linspace(0, np.pi, 400001)
    vals = gn(grid)
    mn, mx = vals.min(), vals.max()
    cmin = -(3 * sp.pi + 14) / 8
    cmax = (sp.pi - 14) / 16
    ok2 = abs(mn - float(cmin)) < 1e-9 and abs(mx - float(cmax)) < 1e-9         and sp.simplify(g.subs(T, 3 * sp.pi / 4) - cmin) == 0 and sp.simplify(g.subs(T, sp.pi / 4) - cmax) == 0
    # (3) 치환 x=f(t): ∫g(f^{-1}(x))dx = ∫ g(t) f'(t) dt
    I = sp.integrate(sp.simplify(g * (fp.subs(X, T) - 1)), (T, 0, sp.pi))
    from scipy.integrate import quad
    from scipy.optimize import brentq
    fn = sp.lambdify(X, f); finv = lambda y: brentq(lambda s: fn(s) - y, -1, 5)
    num = quad(lambda y: gn(finv(y)), fn(0), fn(np.pi), limit=200)[0] - quad(gn, 0, np.pi, limit=200)[0]
    ok3 = sp.simplify(I + 4 * sp.pi) == 0 and abs(num + 4 * np.pi) < 1e-7
    report(1, ok1 and ok2 and ok3, f"g(t)={g}; (2) 격자 min={mn:.10f}, max={mx:.10f} (3) 기호={sp.simplify(I)}, 수치={num:.10f}")


def p02():
    """전개도를 접어 정육면체 좌표를 만든 뒤 (1) π/3 (2)(3)(4) π/2."""
    # 전개도 좌표(한 변 1): 가로 네 칸 [0,1],[1,2],[2,3],[3,4] × [0,1], 둘째 칸 위·아래에 한 칸씩
    # A=(0,1), B=(1,0), C=(2,1), D=(3,1), E=(3,0), F=(4,1), H=(4,0), G=(3.5,0.5)
    def fold(p):
        x0, y0 = p
        if 1 <= x0 <= 2:            # 둘째 칸 = 바닥면 z=0
            return sp.Matrix([x0, y0, 0])
        if x0 < 1:                  # 첫째 칸: x=1 을 축으로 세움
            return sp.Matrix([1, y0, 1 - x0])
        if x0 <= 3:                 # 셋째 칸: x=2 를 축으로 세움
            return sp.Matrix([2, y0, x0 - 2])
        return sp.Matrix([5 - x0, y0, 1])   # 넷째 칸: 윗면
    R = sp.Rational
    net = {"A": (0, 1), "B": (1, 0), "C": (2, 1), "D": (3, 1), "E": (3, 0), "F": (4, 1), "H": (4, 0), "G": (R(7, 2), R(1, 2))}
    P = {k: fold(v) for k, v in net.items()}
    # 접기 검증: 꼭짓점이 모두 [1,2]×[0,1]×[0,1] 정육면체의 꼭짓점(정수 좌표)인지
    cube_ok = all(1 <= P[k][0] <= 2 and 0 <= P[k][1] <= 1 and 0 <= P[k][2] <= 1 and all(c.is_integer for c in P[k]) for k in "ABCDEFH")
    def ang(p, q, r, s):
        u, v = P[q] - P[p], P[s] - P[r]
        return sp.acos(sp.Abs(u.dot(v)) / (u.norm() * v.norm()))
    a1, a2, a3, a4 = ang("A", "B", "B", "C"), ang("B", "C", "E", "F"), ang("C", "D", "D", "H"), ang("C", "G", "E", "F")
    ok = cube_ok and P["A"] == P["F"] and (a1, a2, a3, a4) == (sp.pi / 3, sp.pi / 2, sp.pi / 2, sp.pi / 2)
    report(2, ok, f"A=F={list(P['A'])}, 각=({a1}, {a2}, {a3}, {a4})")


def p03():
    """(1) |v_A|=sinθ/(1-sinθ) (2) θ=π/3, 3+2√3 (3) θ=π/6."""
    import numpy as np
    th, t = sp.symbols("theta t", real=True)
    Px, Py = sp.cos(th), sp.sin(th)
    Qx, Qy = sp.cos(th + t), sp.sin(th + t)
    a = sp.simplify((Px * Qy - Qx * Py) / (Qy - Py))          # 직선 PQ 의 x절편
    va = sp.diff(a, t).subs(t, sp.pi / 2 - th)
    claim = sp.sin(th) / (1 - sp.sin(th))
    ok1 = all(abs(abs(float(va.subs(th, v))) - float(claim.subs(th, v))) < 1e-12 for v in np.linspace(0.01, float(sp.pi / 3), 50))
    vn = sp.lambdify(th, claim, "numpy")
    grid = np.linspace(0, np.pi / 3, 100001)
    ok2 = abs(grid[vn(grid).argmax()] - np.pi / 3) < 1e-9 and sp.simplify(claim.subs(th, sp.pi / 3) - (3 + 2 * sp.sqrt(3))) == 0
    d2 = sp.simplify((a.subs(t, sp.pi / 2 - th))**2 + 1)       # Q=(0,1), A=(a,0)
    sols = [s for s in sp.solve(sp.Eq(d2, 4 * claim), th) if 0 <= s <= sp.pi / 3]
    ok3 = sols == [sp.pi / 6]
    report(3, ok1 and ok2 and ok3, f"a(t)={a}; (1) 수치 50점 일치 (2) 최대 θ=π/3 → {sp.nsimplify(claim.subs(th, sp.pi/3))} (3) d²={d2}, θ={sols}")


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
