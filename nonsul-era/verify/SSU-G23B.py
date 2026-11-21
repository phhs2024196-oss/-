#!/usr/bin/env python3
"""숭실대 2023학년도 논술(자연2) 문제 1~4 (SSU-G23B-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import math
import re
import sys

import mpmath as mp
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

PREFIX = "SSU-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    """yaml 의 subparts[sub].answer.value 원문(문자열)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return str(d["subparts"][sub]["answer"]["value"])


def tex(s):
    """간단한 LaTeX 식 → sympy (\\dfrac, \\sqrt, \\pi, ^, 암묵적 곱)."""
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\dfrac", "\\frac")
    for _ in range(10):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", " pi ").replace("^", "**").replace("{", "(").replace("}", ")")
    return parse_expr(s, local_dict={"x": x, "t": t, "pi": sp.pi, "sqrt": sp.sqrt},
                      transformations=standard_transformations + (implicit_multiplication_application,))


import numpy as np


def golden(fun, lo, hi, it=200, maximize=False):
    g = (lambda u: -fun(u)) if maximize else fun
    r = (mp.sqrt(5) - 1) / 2
    a, b = mp.mpf(lo), mp.mpf(hi)
    c, d = b - r * (b - a), a + r * (b - a)
    for _ in range(it):
        if g(c) < g(d):
            b, d = d, c
            c = b - r * (b - a)
        else:
            a, c = c, d
            d = a + r * (b - a)
    return (a + b) / 2


def p01():
    """교점 P_m 을 직접 풀어 자취 확인 → 자취 위 세 점(실제 m 값에서 나온 점)으로 만든 삼각형 중 한 각이 π/6 인 것의 넓이 최대 탐색."""
    m = sp.symbols("m", real=True)
    X, Y = sp.symbols("X Y", real=True)
    sol = sp.solve([X + 2 * m * Y, 2 * m * X - Y - 2 * m], [X, Y], dict=True)[0]
    Px, Py = sp.simplify(sol[X]), sp.simplify(sol[Y])
    assert sp.simplify(Px**2 - Px + Py**2) == 0          # 원 (x-1/2)^2+y^2=1/4 위
    fx, fy = sp.lambdify(m, Px, "numpy"), sp.lambdify(m, Py, "numpy")
    ms = np.tan(np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 1801))   # 실수 m 전체를 훑음
    pts = np.stack([fx(ms), fy(ms)], 1)
    # A, B 를 실제 m 값에서 나온 점(격자)으로 고르고, ∠BAC=π/6 이 되도록 C 를 정확히 잡는다
    # (원주각 π/6 ⇔ 현 BC 의 중심각 π/3: C 는 B 를 중심 (1/2,0) 둘레로 ±π/3 회전, A 가 큰 호 위에 있어야 함)
    cen = np.array([0.5, 0.0])
    P = pts[::2]
    ang = np.arctan2(P[:, 1], P[:, 0] - 0.5)
    best = 0.0
    for sgn in (1, -1):
        angC = ang + sgn * np.pi / 3
        Cx, Cy = 0.5 + 0.5 * np.cos(angC), 0.5 * np.sin(angC)
        okC = Cx < 1 - 1e-12                       # C 도 어떤 실수 m 에서 나와야 함((1,0) 제외)
        for i in range(len(P)):
            A = P[i]
            u1x, u1y = P[:, 0] - A[0], P[:, 1] - A[1]
            u2x, u2y = Cx - A[0], Cy - A[1]
            n1, n2 = np.hypot(u1x, u1y), np.hypot(u2x, u2y)
            good = okC & (n1 > 1e-9) & (n2 > 1e-9)
            cosA = np.where(good, (u1x * u2x + u1y * u2y) / (n1 * n2 + 1e-300), -2)
            good &= np.abs(cosA - np.cos(np.pi / 6)) < 1e-9      # A 가 큰 호 위(각이 정확히 π/6)
            if good.any():
                ar = 0.5 * np.abs(u1x * u2y - u1y * u2x)[good]
                best = max(best, ar.max())
    md = float(tex(load_answer(1)))
    # 격자 최대는 참값 이하 근사 + 꼭짓점 하나라도 (1,0) 이 아닌 실제 점으로 참값 달성 가능(대칭 배치) 확인
    s = np.pi / 3                                # 꼭짓점 각 배치: 중심각 기준
    th = np.array([np.pi / 2, np.pi / 2 + (2 * np.pi - s) / 2, np.pi / 2 + (2 * np.pi - s) / 2 + s])
    V = np.stack([0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th)], 1)
    exact_area = 0.5 * abs((V[1, 0] - V[0, 0]) * (V[2, 1] - V[0, 1]) - (V[2, 0] - V[0, 0]) * (V[1, 1] - V[0, 1]))
    realizable = all(v[0] < 1 - 1e-9 for v in V)  # x=1 인 점 (1,0) 만 m 으로 만들 수 없음
    ok = abs(best - md) < 1e-5 and best <= md + 1e-12 and abs(exact_area - md) < 1e-12 and realizable
    report(1, ok, f"자취 x^2-x+y^2=0 확인, 격자 최대 넓이≈{best:.6f}, 대칭 배치 넓이={exact_area:.12f}, md={load_answer(1)}")


import functools


@functools.lru_cache(maxsize=None)
def nroots(k):
    """|2x|/(x^2+1)=k 의 서로 다른 실근 개수(정확)."""
    k = sp.sympify(k)
    cnt = set()
    for sgn in (1, -1):        # x≥0: 2x = k(x^2+1),  x<0: -2x = k(x^2+1)
        for r in sp.solve(sp.Eq(sgn * 2 * x, k * (x**2 + 1)), x):
            if r.is_real and ((sgn == 1 and r >= 0) or (sgn == -1 and r < 0)):
                cnt.add(r)
    return len(cnt)


def p02():
    """(1) k 표본점에서 실근 개수를 직접 세어 md 의 구간별 식과 비교. (2) g 의 조건을 연속성 정의로 직접 판정해 g(0) 하한 탐색."""
    a1 = load_answer(2, 0)
    pieces = re.findall(r"(\d+)\\ \(([^)]*)\)", a1)
    def md_f(kv):
        for val, cond in pieces:
            c = cond.replace("=", "==").replace("<==", "<=").replace(">==", ">=")
            if eval(c, {"k": kv}):
                return int(val)
    samples = [-3, -1, sp.Rational(-1, 100), 0, sp.Rational(1, 100), sp.Rational(1, 2), sp.Rational(99, 100), 1,
               sp.Rational(101, 100), 2, 5]
    ok1 = all(nroots(kv) == md_f(kv) for kv in samples) and len(pieces) == 5
    report(2, ok1, f"(1) 표본 k 의 근 개수 {[nroots(kv) for kv in samples]}, md={a1}")
    # (2) f 의 값: 근 개수 함수 F(k). 좌·우극한은 k±1e-9 로.
    F = lambda kv: nroots(kv)
    eps = sp.Rational(1, 10**9)
    jumps = [(F(c - eps), F(c), F(c + eps)) for c in (0, 1)]          # (좌, 값, 우)
    p_, q_, r_, s_ = sp.symbols("p q r s", real=True)
    G = x**4 + p_ * x**3 + q_ * x**2 + r_ * x + s_
    eqs = []
    for L, V, R in jumps:                                              # g∘f 연속 ⇔ g(좌)=g(값)=g(우)
        eqs += [G.subs(x, L) - G.subs(x, V), G.subs(x, R) - G.subs(x, V)]
    solg = sp.solve(eqs, [p_, q_, r_], dict=True)[0]
    Ga = sp.expand(G.subs(solg))                                       # s_ 가 자유 → g(0)=s_
    def feasible(sv):
        g = Ga.subs(s_, sv)
        vals = [g.subs(x, nn) for nn in range(-10, 11)]
        if any(v < 0 for v in vals):                                   # (ii)
            return False
        fvals = {F(v) for v in vals}                                   # (iv): f(g(n)) 가 모든 정수 n 에서 같아야 연속
        return len(fvals) == 1
    lo, hi = sp.Integer(0), sp.Integer(100)
    assert not feasible(lo) and feasible(hi)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if feasible(mid) else (mid, hi)
    c_md = tex(load_answer(2, 1))
    ok2 = abs(hi - c_md) < 1e-9 and not feasible(c_md) and feasible(c_md + eps)
    report(2, ok2, f"(2) g = {sp.factor(Ga - s_)} + g(0); 가능한 g(0) 의 하한≈{sp.N(hi, 12)} (하한 자체는 불가), md={load_answer(2, 1)}")


def p03():
    """기름 부피 보존: 윗면 높이 u 마다 아랫면 높이 w 를 수치로 풀어 두께 u-w 의 최댓값 탐색(윗면이 8 아래로 내려가는 경우 포함)."""
    def A(xx):
        return mp.pi * ((8 - xx) / 2 + 1) if xx <= 8 else mp.pi * (xx - 7)
    def vol(w, u):
        pts = [w, u] if (u <= 8 or w >= 8) else [w, mp.mpf(8), u]
        return sum(mp.quad(A, [pts[i], pts[i + 1]]) for i in range(len(pts) - 1))
    V = vol(mp.mpf(8), mp.mpf(12))
    def thick(u):
        if vol(mp.mpf(0), u) < V:
            return None
        lo, hi = mp.mpf(0), u
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if vol(mid, u) > V else (lo, mid)
        return u - lo
    grid = [(thick(mp.mpf(k) / 100), k) for k in range(400, 1201)]
    grid = [(th, k) for th, k in grid if th is not None]
    th0, k0 = max(grid)
    us = golden(lambda u: thick(u), (k0 - 1) / mp.mpf(100), (k0 + 1) / mp.mpf(100), it=60, maximize=True)
    tmax = thick(us)
    md = tex(load_answer(3))
    ok = abs(tmax - mp.mpf(str(sp.N(md, 30)))) < 1e-8 and abs(V - 12 * mp.pi) < 1e-20
    report(3, ok, f"기름 부피={mp.nstr(V / mp.pi, 10)}π, 최대 두께≈{mp.nstr(tmax, 12)} (윗면 높이 {mp.nstr(us, 10)}), md={load_answer(3)}")


def p04():
    """(1) Q=(1,2), R=Q 에서 ℓ 에 내린 수선의 발 → 코사인법칙으로 각 OQR. (2) t'(1)=0 으로 a 를 풀고, 그 a 에서 x=1 이 전역 최소인지 수치 확인."""
    Q = sp.Matrix([1, 2])
    assert 3 - 1**4 == 2
    # 수선의 발: ℓ: 3x+y-10=0
    xr, yr = sp.symbols("xr yr", real=True)
    R = sp.solve([3 * xr + yr - 10, (yr - 2) - sp.Rational(1, 3) * (xr - 1)], [xr, yr])
    Rv = sp.Matrix([R[xr], R[yr]])
    O = sp.Matrix([0, 0])
    d2 = lambda U, W: sum((U - W).applyfunc(lambda z: z**2))
    cosq = (d2(Q, O) + d2(Q, Rv) - d2(O, Rv)) / (2 * sp.sqrt(d2(Q, O) * d2(Q, Rv)))
    sinq = sp.sqrt(1 - cosq**2)
    tanq = sp.simplify(sinq / cosq)
    ok1 = sp.simplify(tanq - tex(load_answer(4, 0))) == 0
    report(4, ok1, f"(1) R={tuple(Rv)}, cos∠OQR={sp.simplify(cosq)}, tan={tanq}, md={load_answer(4, 0)}")
    a = sp.symbols("a", positive=True)
    T = sp.sqrt(x**2 + (3 - x**4) ** 2) / a + sp.Abs(3 * x + (3 - x**4) - 10) / sp.sqrt(10)
    T_ = sp.sqrt(x**2 + (3 - x**4) ** 2) / a - (3 * x + (3 - x**4) - 10) / sp.sqrt(10)
    asol = [s for s in sp.solve(sp.diff(T_, x).subs(x, 1), a) if s.is_positive]
    av = asol[0]
    Tf = sp.lambdify(x, T.subs(a, av), "mpmath")
    xs = [mp.mpf(k) / 1000 for k in range(-2000, 2001)]
    xmin = min(xs, key=Tf)
    ok2 = len(asol) == 1 and sp.simplify(av - tex(load_answer(4, 1))) == 0 and abs(xmin - 1) < 2e-3
    report(4, ok2, f"(2) a={av}, 그 a 에서 T(x) 전역 최소 위치≈{mp.nstr(xmin, 6)}, md={load_answer(4, 1)}")


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
