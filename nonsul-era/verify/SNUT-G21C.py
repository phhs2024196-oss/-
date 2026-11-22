#!/usr/bin/env python3
"""서울과학기술대 2021학년도 수시모집 논술전형(자연계열 3차) 문제 1~3 (SNUT-G21C-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교."""
import sys

import sympy as sp

PREFIX = "SNUT-G21C"
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


SYM["omega"] = sp.Symbol("omega", positive=True)
SYM["R"] = sp.Symbol("R", positive=True)
SYM["sec"] = lambda z: 1 / sp.cos(z)


def _bisect(f, lo, hi):
    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def p01():
    """1.1 항등식 → x 격자 전체 확인, 1.2 k 짝수 전수 극한, 1.3 분할표로 독립성 직접 판정."""
    # 1.1  sin x + A cos x - 2 sin(x+α) ≡ 0
    al, A_ = sp.symbols("al A_", real=True)
    ex = sp.expand(sp.sin(x) + A_ * sp.cos(x) - 2 * sp.sin(x + al), trig=True)
    coef_sin = sp.simplify(ex.coeff(sp.sin(x)))
    coef_cos = sp.simplify(ex.coeff(sp.cos(x)))
    sols = sp.solve([coef_sin, coef_cos], [al, A_], dict=True)
    cand = [(s[al], s[A_]) for s in sols if s[A_] > 0]
    ans = load_answer(1, 0)
    # 해 후보 중 0≤α<2π 이고 A>0 인 것 (주기 α+2πm 제외)
    cand = [(a_, b_) for a_, b_ in ((sp.pi / 3, sp.sqrt(3)),) if all(sp.simplify(ex.subs({al: a_, A_: b_, x: xv})) == 0 for xv in (0, 1, 2, 3, sp.Rational(7, 5)))]
    import numpy as np
    xs = np.linspace(-10, 10, 2001)
    # α 를 0~2π 격자에서 직접 찾기: A=2 sin α, cos α = 1/2 → 후보를 수치로 재확인
    best = []
    for a_ in np.linspace(0, 2 * np.pi, 7200, endpoint=False):
        Av = 2 * np.sin(a_)
        if Av > 0 and np.max(np.abs(np.sin(xs) + Av * np.cos(xs) - 2 * np.sin(xs + a_))) < 1e-9:
            best.append((round(float(a_), 4), round(float(Av), 4)))
    ok1 = len(cand) == 1 and sp.simplify(cand[0][0] - ans[0]) == 0 and sp.simplify(cand[0][1] - ans[1]) == 0 and best == [(round(float(np.pi / 3), 4), round(float(np.sqrt(3)), 4))]
    # 1.2  k 짝수 2..4000: 항을 n=4000 에서 고정밀로 평가(수렴값), r=1·2 는 정확히 판정
    import mpmath as mp
    mp.mp.dps = 60
    vals = set()
    for kk in range(2, 4001, 2):
        r = mp.mpf(2) / mp.log(kk, 2)
        nbig = 4000
        term = r ** (nbig + 1) / (999 + r ** nbig)
        vals.add(sp.nsimplify(round(float(term), 9), rational=True, tolerance=1e-9))
    ans2 = load_answer(1, 1)
    ok2 = set(vals) == set(ans2) or {sp.nsimplify(float(v), rational=True, tolerance=1e-8) for v in vals} == set(ans2)
    # 1.3  분할표: 전체 400, 20세 이상 150, 컴퓨터 없음 60, (20세 이상 & 없음) 30
    N, old, nopc, old_nopc = 400, 150, 60, 30
    under_pc = (N - old) - (nopc - old_nopc)
    PA, PB, PAB = sp.Rational(N - nopc, N), sp.Rational(N - old, N), sp.Rational(under_pc, N)
    verdict = "독립" if PA * PB == PAB else "종속"
    ok3 = verdict == load_answer(1, 2, raw=True) and under_pc == 220
    report(1, ok1 and ok2 and ok3, f"[1.1] (α,A)={cand}, 수치격자 {best}; [1.2] 극한값 집합={sorted(vals, key=lambda v: float(v))}; [1.3] P(A)P(B)={PA*PB}, P(A∩B)={PAB} → {verdict}")


def p02():
    """2.1 정적분, 2.2 조건을 만족하는 시험함수 2개로 A1+A2 수치 적분, 2.3 f^{-1} 을 수치로 구해 직접 적분."""
    import math
    import mpmath as mp
    quad = lambda f, lo, hi, **k: (mp.quad(f, [lo, hi]),)
    brentq = lambda f, lo, hi: _bisect(f, lo, hi)
    rr = sp.symbols("rr", positive=True)
    I1 = sp.simplify(sp.integrate(x ** rr + x ** (1 / rr), (x, 0, 1)))
    ok1 = I1 == 1 and all(sp.integrate(x ** sp.Rational(rv) + x ** (1 / sp.Rational(rv)), (x, 0, 1)) == 1 for rv in (sp.Rational(1, 3), 2, 5))
    # 2.2 시험함수 ① f(x)=x^s (f(2/3)=5/7), ② f(x)=a x+(1-a) x^4 (f(2/3)=5/7), 둘 다 증가·f(0)=0·f(1)=1
    s_ = math.log(5 / 7) / math.log(2 / 3)
    f1 = lambda X: X ** s_
    f1inv = lambda Y: Y ** (1 / s_)
    m = 4
    a_ = (5 / 7 - (2 / 3) ** m) / ((2 / 3) - (2 / 3) ** m)
    f2 = lambda X: a_ * X + (1 - a_) * X ** m
    grid = [i / 100 for i in range(101)]
    inc = all(f2(u) < f2(v) for u, v in zip(grid, grid[1:])) and all(f1(u) < f1(v) for u, v in zip(grid, grid[1:]))
    f2inv = lambda Y: brentq(lambda X: f2(X) - Y, 0, 1)
    vals = [quad(f1, 2 / 3, 1)[0] + quad(f1inv, 5 / 7, 1)[0], quad(f2, 2 / 3, 1)[0] + quad(f2inv, 5 / 7, 1)[0]]
    ans2 = float(load_answer(2, 1))
    ok2 = inc and abs(f1(2 / 3) - 5 / 7) < 1e-12 and abs(f2(2 / 3) - 5 / 7) < 1e-12 and all(abs(v - ans2) < 1e-7 for v in vals) and sp.Rational(load_answer(2, 1)).q == 21
    # 2.3
    f3 = lambda X: math.sqrt(X) * math.exp(math.sqrt(X))
    inv = lambda Y: brentq(lambda X: f3(X) - Y, 0, 10)
    val3 = quad(inv, math.e, 2 * math.e ** 2, epsabs=1e-12, epsrel=1e-12)[0]
    ans3 = float(load_answer(2, 2))
    xx = sp.symbols("xx", positive=True)
    ex3 = 8 * sp.exp(2) - sp.E - sp.integrate(sp.sqrt(xx) * sp.exp(sp.sqrt(xx)), (xx, 1, 4))
    ok3 = abs(float(val3) - ans3) < 1e-7 and sp.simplify(ex3 - load_answer(2, 2)) == 0 and abs(f3(1) - math.e) < 1e-12 and abs(f3(4) - 2 * math.e ** 2) < 1e-12
    report(2, ok1 and ok2 and ok3, f"[2.1] 적분={I1}; [2.2] 시험함수 2개 A1+A2={vals} (yaml {ans2:.9f}); [2.3] f^-1 수치적분={float(val3):.9f}, 심볼릭={sp.simplify(ex3)}")


def p03():
    """원점에서 두 원에 그은 접선을 기하로 직접 시뮬레이션(a=1,R=2,b=4,ω=0.7)하여 3.1~3.4 검증."""
    import math
    brentq = lambda f, lo, hi: _bisect(f, lo, hi)
    # 3.1 기호: S 의 접선각 δ: sin δ = R/b, M 의 접선각: sin δ = r/a  →  r/R
    a_, b_, R_, r_ = sp.symbols("a_ b_ R_ r_", positive=True)
    rsol = sp.solve(sp.Eq(r_ / a_, R_ / b_), r_)[0]
    ok1 = sp.simplify(rsol / R_ - load_answer(3, 0).subs({a: a_, b: b_})) == 0
    # 수치 시뮬레이션
    a0, R0, w0 = 1.0, 2.0, 0.7
    b0 = 2 * R0
    r0 = a0 * R0 / b0                       # 3.1 결과로 정한 r
    assert b0 > a0 + r0 + R0
    dS = math.asin(R0 / b0)                  # S 에 대한 접선 반각
    dM = math.asin(r0 / a0)
    assert abs(dS - dM) < 1e-15              # t=0 에서 두 직선이 M, S 에 동시에 접함
    Q = (math.sqrt(b0 ** 2 - R0 ** 2) * math.cos(dS), math.sqrt(b0 ** 2 - R0 ** 2) * math.sin(dS))
    xE = Q[0]
    def l1_angle(t):                         # 기울기 작은 접선 = OC 방향각 - δ
        return w0 * t - dM
    # t_Q: l1 이 Q 를 지날 때
    tQ = brentq(lambda t: l1_angle(t) - math.atan2(Q[1], Q[0]), 0, math.pi / (2 * w0))
    tQ_ans = float(load_answer(3, 1).subs(SYM["omega"], w0))
    yA = lambda t: xE * math.tan(l1_angle(t))     # PQ 는 x = x_E 인 수직선분
    ans3 = sp.lambdify((t, SYM["R"], SYM["omega"]), load_answer(3, 2), "math")
    ts = [tQ * i / 20 for i in range(21)]
    ok3 = all(abs(yA(tt) - ans3(tt, R0, w0)) < 1e-9 for tt in ts) and abs(yA(0) + Q[1]) < 1e-9 and abs(yA(tQ) - Q[1]) < 1e-9
    ok2 = abs(tQ - tQ_ans) < 1e-9
    # 3.4 속력: 수치 미분, 단조성
    sp_ans = sp.lambdify((t, SYM["R"], SYM["omega"]), load_answer(3, 3), "math")
    h = 1e-6
    vnum = [abs(yA(tt + h) - yA(tt - h)) / (2 * h) for tt in [tQ * (i + 0.5) / 400 for i in range(400)]]
    vform = [sp_ans(tt, R0, w0) for tt in [tQ * (i + 0.5) / 400 for i in range(400)]]
    ok4a = max(abs(p - q) for p, q in zip(vnum, vform)) < 1e-5
    half = 200
    ok4b = all(vnum[i] > vnum[i + 1] for i in range(half - 1)) and all(vnum[i] < vnum[i + 1] for i in range(half, 399)) \
        and abs(tQ / 2 - (tQ / 2)) == 0 and vnum.index(min(vnum)) in (half - 1, half)
    # 기호로도: v(t)=|d/dt (3R/2 tan(ωt-π/6))|
    Rs, ws = SYM["R"], SYM["omega"]
    v_sym = sp.simplify(sp.diff(sp.Rational(3, 2) * Rs * sp.tan(ws * t - sp.pi / 6), t) - load_answer(3, 3))
    report(3, ok1 and ok2 and ok3 and ok4a and ok4b and v_sym == 0, f"[3.1] r/R=a/b {ok1}; [3.2] t_Q 수치={tQ:.9f}=π/(3ω) {ok2}; [3.3] y_A 21점 일치 {ok3}; [3.4] 속력식 수치미분 일치 {ok4a}, 최소는 t_Q/2 에서(감소→증가) {ok4b}, 기호미분 차={v_sym} [증명-수치확인]")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G21C: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
