#!/usr/bin/env python3
"""서울과학기술대 2021학년도 수시모집 논술전형(자연계열 4차) 문제 1~3 (SNUT-G21D-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교."""
import sys

import sympy as sp

PREFIX = "SNUT-G21D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi, "oo": sp.oo}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


for _nm in ("alpha", "l", "r"):
    SYM[_nm] = sp.Symbol(_nm, real=True)
SYM["l"] = sp.Symbol("l", positive=True)
SYM["r"] = sp.Symbol("r", positive=True)


def p01():
    """1.1 a 를 격자로 찾아 b 결정, 1.2 두 곡선 사이 넓이 수치적분, 1.3 직각삼각형 좌표로 l2/l1 최대 탐색."""
    import numpy as np
    import mpmath as mp
    # 1.1  f=a sin^2θ + a sin(θ+π/2) + b + 1.  g=f-b-1 의 (최대-최소)=9 가 되는 a 를 격자 탐색
    th = np.linspace(0, 2 * np.pi, 400001)
    best = None
    for k in range(1, 401):
        av = k / 4
        g = av * np.sin(th) ** 2 + av * np.sin(th + np.pi / 2)
        if abs((g.max() - g.min()) - 9) < 1e-6:
            bv = 1 - g.min()
            if abs(g.max() + bv + 1 - 11) < 1e-6:
                best = (av, bv)
    ans1 = load_answer(1, 0)
    ok1 = best is not None and abs(best[0] * best[1] - float(ans1)) < 1e-6 and best == (4.0, 5.0)
    # 1.2  넓이 = ∫_1^{e^2} |log_{ln a} x - log_{1/ln a} x| dx, 교점 x=1 확인
    mp.mp.dps = 25
    ok2 = True
    for av in (3.0, 5.0, 20.0, 100.0):
        base = mp.log(av)
        lg1 = lambda X: mp.log(X) / mp.log(base)
        lg2 = lambda X: mp.log(X) / mp.log(1 / base)
        meet1 = abs(lg1(1) - lg2(1)) < 1e-20
        area = mp.quad(lambda X: abs(lg1(X) - lg2(X)), [1, mp.e ** 2])
        form = float(load_answer(1, 1).subs(a, av))
        ok2 &= meet1 and abs(float(area) - form) < 1e-12
    # 1.3  B=(0,0), C=(10cosθ,0), A=(0,10sinθ). 내접원 반지름 = 넓이/반둘레
    ths = np.linspace(1e-6, np.pi / 2 - 1e-6, 2000001)
    AB, BC, AC = 10 * np.sin(ths), 10 * np.cos(ths), 10.0
    s_ = (AB + BC + AC) / 2
    rin = (AB * BC / 2) / s_
    ratio = 2 * np.pi * rin / (AB + BC + AC)
    i = ratio.argmax()
    A_deg, C_deg = np.degrees(np.pi / 2 - ths[i]), np.degrees(ths[i])
    ans3 = load_answer(1, 2)
    ok3 = abs(np.radians(C_deg) - float(ans3[1])) < 1e-5 and abs(np.radians(A_deg) - float(ans3[0])) < 1e-5
    report(1, ok1 and ok2 and ok3, f"[1.1] (a,b)={best}, ab={best[0]*best[1] if best else None}; [1.2] 4개의 a 에서 수치적분=공식 {ok2}; [1.3] 최대 θ=C={C_deg:.4f}°, A={A_deg:.4f}°")


def p02():
    """f(x)=tanh(√7 k x)/√7 (k=f'(0)) 가 조건 (가)(다)를 만족함을 이용한 시험함수로 2.1~2.4 검증, 2.3 은 A,B,C 를 연립해 구함."""
    import mpmath as mp
    mp.mp.dps = 30
    s7 = mp.sqrt(7)
    ok = True
    for kk in (1, 0.6, 1.3):
        f = lambda X, kk=kk: mp.tanh(s7 * kk * X) / s7
        # (다) 함수방정식 확인
        eq = max(abs(f(u + v) - (f(u) + f(v)) / (1 + 7 * f(u) * f(v))) for u in (-1.2, 0.3, 2.0) for v in (-0.7, 0.5, 1.1))
        alpha = f(1)
        ok &= eq < 1e-14 and abs(f(0)) < 1e-30
        # 2.1, 2.2
        ok &= abs(f(-1) - float(load_answer(2, 0).subs(SYM["alpha"], float(alpha)))) < 1e-12
        ok &= abs(mp.quad(f, [-1, 1])) < 1e-25 and load_answer(2, 1) == 0
    # 2.3: f'(x) = f'(0)(A f^2 + B f + C): k=1 에서 세 점으로 연립
    f = lambda X: mp.tanh(s7 * X) / s7
    fp = lambda X: mp.diff(f, X)
    xs3 = [0.2, 0.5, 0.9]
    M = mp.matrix([[f(X) ** 2, f(X), 1] for X in xs3])
    sol = mp.lu_solve(M, mp.matrix([fp(X) / fp(0) for X in xs3]))
    ans3 = [float(v) for v in sp.sympify(load_answer(2, 2, raw=True), locals=SYM)]
    ok &= all(abs(float(sol[i]) - ans3[i]) < 1e-12 for i in range(3))
    # 2.4: f'(0)=1
    alpha = f(1)
    ans4 = load_answer(2, 3)
    f2_formula = float(ans4[0].subs(SYM["alpha"], alpha))
    I = mp.quad(lambda X: (1 + 7 * alpha ** 2) * f(X) ** 2, [0, 2])
    I_formula = float(ans4[1].subs(SYM["alpha"], alpha))
    ok &= abs(float(f(2)) - f2_formula) < 1e-15 and abs(float(I) - I_formula) < 1e-14
    report(2, ok, f"[2.1] f(-1)=-α, [2.2] ∫=0, [2.3] (A,B,C)={[round(float(v),10) for v in sol]}, [2.4] f(2)={float(f(2)):.10f}, ∫={float(I):.10f} (yaml {f2_formula:.10f}, {I_formula:.10f})")


def p03():
    """3.1 격자 전수(l=4..16, 가능한 r 모두): 원(곡선)과 S(네 변)가 만나는지 거리 범위로 판정, 3.2~3.4 확률."""
    from fractions import Fraction
    import math
    f31 = load_answer(3, 0)
    f32 = load_answer(3, 1)
    ok1 = ok2 = True
    n_checked = 0
    for ll in range(4, 17):
        for rr in range(1, ll):
            if not rr < ll / 2 - 1:
                continue
            corners = [(0, 0), (ll - 1, 0), (ll - 1, ll - 1), (0, ll - 1)]
            meet = 0
            for aa in range(ll):
                for bb in range(ll):
                    # S 위 점들까지 거리의 범위 [dmin, dmax]: 변은 연결이므로 dmin<=r<=dmax 이면 원과 만남
                    dmax = max(math.hypot(aa - cx, bb - cy) for cx, cy in corners)
                    dmin = min(aa, bb, ll - 1 - aa, ll - 1 - bb)   # 중심이 정사각형 내부/둘레이므로 최단거리는 가장 가까운 변까지의 거리
                    if dmin <= rr <= dmax:
                        meet += 1
            non = ll * ll - meet
            ok1 &= non == f31.subs({SYM["l"]: ll, SYM["r"]: rr})
            ok2 &= sp.Rational(meet, ll * ll) == f32.subs({SYM["l"]: ll, SYM["r"]: rr})
            n_checked += 1
    p = sp.Rational(48, 64)
    p_f = f32.subs({SYM["l"]: 8, SYM["r"]: 1})
    X = sp.Rational(1) - sp.binomial(4, 4) * p ** 4
    ok3 = p_f == p and X == load_answer(3, 2) == sp.Rational(175, 256)
    # 3.4: 평균·분산, 표준화, 표값 합. 표 4자리가 erf 와 일치하는지, 정규근사가 정확한 이항확률과 가까운지 확인
    mu, var = 1200 * p, 1200 * p * (1 - p)
    zlo, zhi = (894 - mu) / sp.sqrt(var), (918 - mu) / sp.sqrt(var)
    table = {sp.Rational(4, 10): 0.1554, sp.Rational(6, 10): 0.2257, sp.Rational(8, 10): 0.2881, 1: 0.3413, sp.Rational(12, 10): 0.3849}
    erf_tab = all(abs(round(0.5 * math.erf(float(z) / math.sqrt(2)), 4) - v) < 1e-12 for z, v in table.items())
    approx = sp.Rational(1554, 10000) + sp.Rational(3849, 10000)
    exact = sum(math.comb(1200, k) * 0.75 ** k * 0.25 ** (1200 - k) for k in range(894, 919))
    ok4 = mu == 900 and var == 225 and zlo == sp.Rational(-2, 5) and zhi == sp.Rational(6, 5) and erf_tab and approx == load_answer(3, 3) and abs(exact - float(approx)) < 0.03
    report(3, ok1 and ok2 and ok3 and ok4, f"[3.1] (l,r) {n_checked}쌍 전수 일치 {ok1}; [3.2] 일치 {ok2}, p(8,1)={p_f}; [3.3] P(X≤3)={X}; [3.4] μ={mu}, σ²={var}, z=({zlo},{zhi}), 표근사={approx}, 정확한 이항확률≈{exact:.4f}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G21D: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
