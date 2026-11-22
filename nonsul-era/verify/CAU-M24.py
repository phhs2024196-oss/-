#!/usr/bin/env python3
"""CAU-M24 (2024학년도 중앙대 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """공 4번 복원추출 전수(16가지 순서) → E[x+y] (답 272/81). 이동은 순서대로 직접 적용."""
    import itertools
    E = sp.Integer(0)
    for seq in itertools.product("BW", repeat=4):
        px, py = sp.Integer(0), sp.Integer(0)
        pr = sp.Integer(1)
        for c in seq:
            if c == "B":
                px, pr = 2 * px + 2, pr * sp.Rational(4, 6)
            else:
                py, pr = 3 * py - 3, pr * sp.Rational(2, 6)
        E += pr * (px + py)
    report(1, E == sp.Rational(272, 81), f"E[x+y]={E}")


def p02():
    """(1) [0,2π] 위 f 의 최댓점 a 에서 sin a (답 -√15/4). (2) P(t,-t), 0≤t≤2 에서 tan∠APB 최댓값 (답 (2+√6)/2)."""
    f = x - 3 * sp.sin(x) + sp.sin(2 * x)
    crit = [c for c in sp.solveset(sp.diff(f, x), x, sp.Interval(0, 2 * sp.pi))]
    cands = crit + [sp.Integer(0), 2 * sp.pi]
    best = max(cands, key=lambda c: float(f.subs(x, c)))
    s1 = sp.nsimplify(sp.N(sp.sin(best), 30), [sp.sqrt(15)])
    ok1 = sp.simplify(s1 + sp.sqrt(15) / 4) == 0
    # (2) 벡터로 직접 tan 계산: tan = |cross|/dot
    ax, ay, bx, by = 0, 1, 2, 1
    px, py = t, -t
    u = (ax - px, ay - py); v = (bx - px, by - py)
    cr = u[0] * v[1] - u[1] * v[0]; dt = u[0] * v[0] + u[1] * v[1]
    tn = sp.simplify((-(cr)) / dt) if sp.N((-cr).subs(t, 1)) > 0 else sp.simplify(cr / dt)
    import numpy as np
    g = sp.lambdify(t, tn, "numpy")
    ts = np.linspace(0, 2, 200001)
    vals = g(ts)
    num_max = float(vals.max())
    cs = [c for c in sp.solveset(sp.diff(tn, t), t, sp.Interval(0, 2))]
    exact = max([tn.subs(t, c) for c in cs] + [tn.subs(t, 0), tn.subs(t, 2)], key=lambda e: float(e))
    ok2 = sp.simplify(exact - (2 + sp.sqrt(6)) / 2) == 0 and abs(num_max - float(exact)) < 1e-8
    report(2, ok1 and ok2, f"(1) sin a={s1}  (2) tanθ={sp.simplify(tn)}, max={sp.nsimplify(exact)} (격자 {num_max:.10f})")


def p03():
    """(1) I_n 수치적분 vs ln(a_n)/n, 극한 ln2. (2) 함수방정식+조건(나)로 f 결정 → Σf'(k) (답 440)."""
    import mpmath as mp
    ok_int = True
    for N in (2, 3, 5, 8):
        an = mp.mpf(2) ** N / N**2
        val = mp.quad(lambda X: 1 / (N * X * (1 + X**N)), [1 / an, an])
        ok_int &= abs(val - mp.log(an) / N) < 1e-12
    nn = sp.symbols("nn", positive=True)
    L = sp.limit(sp.log(2**nn / nn**2) / nn, nn, sp.oo)
    ok1 = ok_int and sp.simplify(L - sp.log(2)) == 0
    # (2) f(x)=x^3/3+x^2/2+c x-1 이 (가)를 만족하는지 확인, (나)에서 c 결정
    y, c = sp.symbols("y c")
    F = lambda z: z**3 / 3 + z**2 / 2 + c * z - 1
    fe = sp.expand(F(x + y) - (F(x) + F(y) + x * y * (x + y + 1) + 1))
    # (나): (f(x)+1)/x → 0 이어야 하므로 c = lim (f(x)+1)/x = 0
    csol = sp.solve(sp.limit((F(x) + 1) / x, x, 0), c)
    f = F(x).subs(c, csol[0])
    # |f(x)+1| ≤ x^2 를 0 근방에서 수치 확인
    import numpy as np
    xs = np.linspace(-0.5, 0.5, 10001)
    fn = sp.lambdify(x, f + 1, "numpy")
    ok_bd = bool(np.all(np.abs(fn(xs)) <= xs**2 + 1e-15))
    S = sum(sp.diff(f, x).subs(x, k) for k in range(1, 11))
    ok2 = fe == 0 and csol == [0] and ok_bd and S == 440
    report(3, ok1 and ok2, f"(1) lim I_n={L}  (2) c={csol}, Σf'(k)={S}")


def p04():
    """(1) 구 외접 원뿔대(z=0~2) 부피=4π 조건으로 원뿔 결정 → A/B (답 (23+5√21)/2).
    (2) 한 변 d 정팔각형 좌표에서 |CA→-BE→|=2 → 넓이 (답 8(3√2-1)/17)."""
    H, R = sp.symbols("H R", positive=True)
    # 단면: 꼭짓점 (0,H), 밑 (R,0). 중심 (0,1)에서 모선까지 거리 = 1
    eq1 = sp.Eq((R * (H - 1))**2, H**2 + R**2)
    r_top = R * (H - 2) / H
    V = sp.pi * 2 / 3 * (R**2 + R * r_top + r_top**2)
    sols = sp.solve([eq1, sp.Eq(V, 4 * sp.pi)], [H, R], dict=True)
    ratios = []
    for s in sols:
        if s[H].is_real and float(s[H]) > 2:
            ratios.append(sp.nsimplify(sp.N((s[R] / r_top.subs(s))**2, 40), [sp.sqrt(21)]))
    ok1 = len(ratios) >= 1 and all(sp.simplify(q - (23 + 5 * sp.sqrt(21)) / 2) == 0 for q in ratios)
    # (2)
    d = sp.symbols("d", positive=True)
    h = d / sp.sqrt(2)
    # 그림: 위 A,H / 왼쪽 B,C / 아래 D,E / 오른쪽 G,F
    A = sp.Matrix([h, h + d + h]); Hh = sp.Matrix([h + d, 2 * h + d])
    B = sp.Matrix([0, h + d]); C = sp.Matrix([0, h])
    D = sp.Matrix([h, 0]); E = sp.Matrix([h + d, 0])
    w = (A - C) - (E - B)
    dsol = [s for s in sp.solve(sp.Eq(w.dot(w), 4), d) if s.is_positive][0]
    area = (2 * h + d)**2 - 4 * h**2 / 2
    area = sp.radsimp(sp.simplify(area.subs(d, dsol)))
    ok2 = sp.simplify(area - 8 * (3 * sp.sqrt(2) - 1) / 17) == 0
    report(4, ok1 and ok2, f"(1) A/B={ratios}  (2) 넓이={area}")


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
