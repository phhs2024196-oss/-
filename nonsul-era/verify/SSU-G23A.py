#!/usr/bin/env python3
"""숭실대 2023학년도 논술(자연1) 문제 1~4 (SSU-G23A-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import math
import re
import sys

import mpmath as mp
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

PREFIX = "SSU-G23A"
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
    """γ 를 정수 격자(-300..300)로 훑어 변곡점 x좌표가 양의 정수이고 두 삼각형이 모두 예각인 f 를 전수 탐색."""
    s5 = sp.sqrt(5)
    O, B = (0, 0), (2 + s5, 0)
    def acute(P, Q, R):
        d2 = lambda U, V: sp.expand((U[0] - V[0]) ** 2 + (U[1] - V[1]) ** 2)
        a2, b2, c2 = d2(Q, R), d2(P, R), d2(P, Q)
        area2 = sp.expand((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1]))
        return area2 != 0 and all(sp.N(u + v - w, 30) > 0 for u, v, w in ((a2, b2, c2), (b2, c2, a2), (c2, a2, b2)))
    found = []
    for g3 in range(-600, 601):          # γ = g3/3 : 변곡점 (4+γ)/3 이 정수가 되려면 γ 는 정수여야 함을 겸사겸사 확인
        gam = sp.Rational(g3, 3)
        f = sp.expand((x - 2 + s5) * (x - 2 - s5) * (x - gam))
        pin = sp.solve(sp.diff(f, x, 2), x)[0]
        if not (pin.is_integer and pin > 0) or gam in (2 + s5, 2 - s5):
            continue
        P, C = (pin, f.subs(x, pin)), (gam, 0)
        if acute(O, P, B) and acute(O, P, C):
            found.append(f)
    md = sp.expand(tex(load_answer(1)))
    ok = len(found) == 1 and sp.expand(found[0] - md) == 0
    report(1, ok, f"조건 만족 f 개수={len(found)}: {found}, md={load_answer(1)}")


def p02():
    """h∈[0,1] 에서 이동 시간 T(h) 를 직접 구성(수평 구간 x=t²/2, 원호 길이/√3)하여 최대화."""
    def T(h):
        xq = 1 + mp.sqrt(1 - h ** 2)            # y=h 와 원 (x-1)^2+y^2=1 (1≤x≤2) 의 교점
        t1 = mp.findroot(lambda s: s ** 2 / 2 - xq, 2)   # x(t)=∫0^t u du
        arc = mp.atan2(h, xq - 1)               # 중심 (1,0), 반지름 1 → 호의 길이 = 중심각
        return t1 + arc / mp.sqrt(3)
    grid = [(T(mp.mpf(k) / 1000), k) for k in range(0, 1001)]
    k0 = max(grid)[1]
    hs = golden(T, max(0, k0 - 1) / mp.mpf(1000), min(1000, k0 + 1) / mp.mpf(1000), maximize=True)
    ans = tex(load_answer(2))
    ok = abs(hs - mp.mpf(str(sp.N(ans, 40)))) < 1e-10
    report(2, ok, f"h*={mp.nstr(hs, 15)}, T_max={mp.nstr(T(hs), 12)}, md={load_answer(2)}")


def p03():
    """(1) 거리제곱 최소를 sympy 로, (2) 우주선 시작점 a 를 바꿔 가며 min_t 거리 를 최소화(이중 최적화)."""
    r2 = sp.sqrt(2)
    cx, cy = 2 * r2 * t - 4 * r2, t**2 - 2 * t + 3
    # 직선과 곡선이 만나지 않음 확인
    assert sp.solve(sp.Eq(cy, r2 * cx - 1), t) == [] or all(not s.is_real for s in sp.solve(sp.Eq(cy, r2 * cx - 1), t))
    sx, sy = -2 * r2 + r2 * t, -5 + 2 * t
    assert sp.simplify(sy.subs(t, 0) - (r2 * sx.subs(t, 0) - 1)) == 0
    s = sp.expand((cx - sx) ** 2 + (cy - sy) ** 2)
    crit = [c for c in sp.solve(sp.diff(s, t), t) if c.is_real]
    tm = min(crit, key=lambda c: s.subs(t, c))
    dist = sp.sqrt(s.subs(t, tm))
    a1 = load_answer(3, 0)
    m1 = re.fullmatch(r"t=(.*),\\ (.*)", a1)
    ok1 = sp.simplify(tex(m1.group(1)) - tm) == 0 and sp.simplify(tex(m1.group(2)) - dist) == 0
    report(3, ok1, f"(1) t={tm}, 거리={dist}, md={a1}")
    aa = sp.symbols("a", real=True)
    S = sp.expand((cx - (aa + r2 * t)) ** 2 + (cy - (r2 * aa - 1 + 2 * t)) ** 2)
    dS = sp.lambdify((t, aa), sp.diff(S, t), "mpmath")
    Sf = sp.lambdify((t, aa), S, "mpmath")
    def mind(a):
        cs = [c for c in sp.Poly(sp.diff(S, t).subs(aa, sp.nsimplify(float(a))), t).nroots(n=30) if abs(sp.im(c)) < 1e-20]
        best = min(cs, key=lambda c: Sf(mp.mpf(str(sp.re(c))), a))
        return Sf(mp.mpf(str(sp.re(best))), a), mp.mpf(str(sp.re(best)))
    a_star = golden(lambda a: mind(a)[0], -10, 10, it=120)
    dmin, t_star = mind(a_star)
    a2 = load_answer(3, 1)
    m2 = re.fullmatch(r"\((.*),(.*)\),\\ t=(.*)", a2)
    X0, Y0, T0 = (tex(m2.group(i)) for i in (1, 2, 3))
    ok2 = (abs(a_star - float(X0)) < 1e-6 and abs(r2 * 0 + float(r2) * a_star - 1 - float(Y0)) < 1e-6
           and abs(t_star - float(T0)) < 1e-5)
    report(3, ok2, f"(2) a*={mp.nstr(a_star, 10)} → 시작점 ({mp.nstr(a_star, 8)}, {mp.nstr(mp.sqrt(2) * a_star - 1, 8)}), "
                   f"t*={mp.nstr(t_star, 10)}, 최소거리={mp.nstr(mp.sqrt(dmin), 10)}, md={a2}")


def p04():
    """조건 (iii)을 만족하는 연속함수 g 를 구간마다 서로 다른 모양(조각별 일차)으로 실제 구성해 두 적분을 정확(유리수) 계산."""
    from fractions import Fraction as Fr
    import random
    random.seed(1)
    N = 2025
    a = [None] + [k * (k - 1) for k in range(1, N + 3)]
    S = [0] * (N + 3)
    for k in range(1, N + 3):
        S[k] = S[k - 1] + a[k]
    # g: 각 구간 [-S_k, -S_{k-1}] (k≥2) 에서 양 끝 0, 내부 꼭짓점 위치 p 가 무작위인 삼각형 모양(넓이 1). 경계에서 0 → 연속
    peak = {k: Fr(random.randint(1, 9), 10) for k in range(2, N + 3)}
    def g(u):
        u = Fr(u)
        if u == 0:
            return Fr(0)
        k = next(k for k in range(2, N + 3) if -S[k] <= u <= -S[k - 1])
        L, R = -S[k], -S[k - 1]
        P = L + peak[k] * (R - L)
        H = Fr(2, R - L)
        return H * (u - L) / (P - L) if u <= P else H * (R - u) / (R - P)
    def g_breaks(k):
        L, R = -S[k], -S[k - 1]
        return [L, L + peak[k] * (R - L), R]
    def integ_g(lo, hi):     # 조각별 일차이므로 꺾인 점 기준 사다리꼴 합이 정확
        pts = sorted({lo, hi} | {b for k in range(2, N + 3) for b in g_breaks(k) if lo < b < hi}) if hi - lo < 10**12 else None
        return sum((pts[i + 1] - pts[i]) * (g(pts[i]) + g(pts[i + 1])) / 2 for i in range(len(pts) - 1))
    # (1): 조건 (iii) ∫_0^{a_n} f(x-S_n)dx (x-S_n≤0 이므로 f=g) 확인 후 ∫_{-S_{n-1}}^{-S_n} g
    vals1 = set()
    for nn in range(2, 12):
        assert integ_g(Fr(-S[nn]), Fr(-S[nn] + a[nn])) == 1
        vals1.add(-integ_g(Fr(-S[nn]), Fr(-S[nn - 1])))
    ok1 = len(vals1) == 1 and all(v == int(tex(load_answer(4, 0))) for v in vals1)
    report(4, ok1, f"(1) n=2..11 에서 ∫_(-S_(n-1))^(-S_n) g = {vals1}, md={load_answer(4, 0)}")
    # (2): f 를 (ii) 로 정의하고 [n-1,n] 마다 꺾인 점에서 사다리꼴 합(정확)
    total = Fr(0)
    for nn in range(1, 2024):
        u = lambda X: -S[nn] + (nn - 1 - X) * a[nn + 1]
        xs = {Fr(nn - 1), Fr(nn)}
        for b in g_breaks(nn + 1):
            X = Fr(nn - 1) - (Fr(b) + S[nn]) / a[nn + 1]
            if nn - 1 < X < nn:
                xs.add(X)
        xs = sorted(xs)
        total += sum((xs[i + 1] - xs[i]) * (g(u(xs[i])) + g(u(xs[i + 1]))) / 2 for i in range(len(xs) - 1))
    # 연속성 확인(정수점에서 좌우 일치)
    cont = all(g(-S[k] + (k - 1 - k) * a[k + 1]) == g(-S[k + 1] + (k - k) * a[k + 2]) for k in range(1, 50))
    md = tex(load_answer(4, 1))
    ok2 = cont and sp.Rational(total.numerator, total.denominator) == md
    report(4, ok2, f"(2) ∫_0^2023 f = {total}, 정수점 연속={cont}, md={load_answer(4, 1)}")


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
