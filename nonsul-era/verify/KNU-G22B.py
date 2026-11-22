#!/usr/bin/env python3
"""KNU 2022학년도 수시 논술(AAT) 자연계열Ⅱ 수학 (KNU-G22B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-G22B"
WIDTH = 2
RESULTS = []
x, c = sp.symbols("x c", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]




# ── 문제 1: 조건 (1)~(4)를 모두 만족하는 구체적 모형을 만들어 수치로 확인 ──
#  θ(x)=πx+p(x) (p 는 주기 1/2), f=sinθ, g=½sin((π-2θ)/3). p 는 p(0)=π/2, p(-1/12)=5π/6, p(±1/4)=4π/5 를 지나는 삼각다항식.
def model():
    import mpmath as mp
    mp.mp.dps = 30
    pi = mp.pi
    c0, c1, s1 = 13 * pi / 20, -3 * pi / 20, -31 * pi / (60 * mp.sqrt(3))
    th = lambda t: pi * t + c0 + c1 * mp.cos(4 * pi * t) + s1 * mp.sin(4 * pi * t)
    f = lambda t: mp.sin(th(t))
    g = lambda t: mp.sin((pi - 2 * th(t)) / 3) / 2
    return mp, f, g


def p01():
    mp, f, g = model()
    msgs, ok = [], True
    grid = mp.linspace(-3, 3, 241)
    c1 = max(abs(f(t + 0.25) ** 2 + f(t - 0.25) ** 2 - 1) for t in grid)
    c2 = max(abs(f(t) * f(t + 0.5) - 3 * g(t) + 16 * g(t) ** 3) for t in grid)
    inner = mp.linspace(-0.25 + 1e-6, -1e-6, 400)
    c3 = all(-0.5 < g(t) < 0 for t in inner)
    c4 = abs(f(mp.mpf(5) / 12) + mp.sqrt(2) / 2) < 1e-20 and abs(g(0)) < 1e-20 and f(0) != 0
    model_ok = c1 < 1e-20 and c2 < 1e-20 and c3 and c4
    ok &= model_ok
    msgs.append(f"모형 조건(1)~(4) 충족={model_ok}")
    # 1-1 [증명-수치확인]: |f'(n)| = |f'(0)|, n=-5..5
    d0 = abs(mp.diff(f, 0))
    ok &= all(abs(abs(mp.diff(f, n)) - d0) < 1e-15 for n in range(-5, 6))
    msgs.append("1-1 |f'(n)|=|f'(0)| (n=-5..5) [증명-수치확인]")
    # 1-2: 문제 조건만으로 — f(-1/12)=±√2/2 두 경우의 삼차방정식 근 중 (-1/2,0) 에 있는 것
    a = sp.symbols("a")
    roots = set()
    for sgn in (1, -1):
        prod = sgn * sp.sqrt(2) / 2 * (-sp.sqrt(2) / 2)
        roots |= {r for r in sp.solve(sp.Eq(3 * a - 16 * a ** 3, prod), a) if r.is_real and -sp.Rational(1, 2) < r < 0}
    a12 = sp.sympify(load_answer(1, 1))
    ok &= roots == {a12} and abs(g(-mp.mpf(1) / 12) - float(a12)) < 1e-20
    msgs.append(f"1-2 후보근={roots}, 모형 g(-1/12)={mp.nstr(g(-mp.mpf(1)/12), 8)} md={a12}")
    # 1-3: 모형에서 적분 수치값 → 유리수 식별 → p+q
    I = mp.quad(lambda t: mp.diff(f, t) * g(t) * (f(t) - 2 * f(t) ** 3), [-mp.mpf(1) / 12, 0])
    q = sp.nsimplify(float(I), tolerance=1e-12, rational=True)
    a13 = int(load_answer(1, 2))
    ok &= q.p + q.q == a13
    msgs.append(f"1-3 적분={mp.nstr(I, 12)}≈{q}, p+q={q.p + q.q} md={a13}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    a0, a1, a2 = sp.symbols("a0 a1 a2", positive=True)
    # 2-1: 접선 기울기각이 두 할선 기울기각의 평균 ⇔ tan(α_n+α_{n+1}) = tan 2β_{n+1} 을 a2 에 대해 풀기
    t1, t2, tb = a0 + a1, a1 + a2, 2 * a1
    sol = sp.solve(sp.Eq((t1 + t2) / (1 - t1 * t2), 2 * tb / (1 - tb ** 2)), a2)
    p, q = sp.symbols("p q")
    form = (4 * a1 ** 3 + q * a1 - a0) / (1 + p * a0 * a1)
    pq = sp.solve(sp.Poly(sp.numer(sp.together(sol[0] - form)), a0, a1).coeffs(), [p, q], dict=True)
    md = [sp.sympify(v) for v in load_answer(2, 0).split(";")]
    ok &= len(sol) == 1 and len(pq) == 1 and [pq[0][p], pq[0][q]] == md
    msgs.append(f"2-1 (p,q)=({pq[0][p]},{pq[0][q]}) md={md}")
    # 실제 기하(예각 비교)로 확인: 수열을 만들어 접선과 두 할선의 예각 비교
    import math
    rec = sp.lambdify((a0, a1), sol[0])
    seq = [0.3, 0.5]
    for _ in range(12):
        seq.append(rec(seq[-2], seq[-1]))
    ang = lambda m1, m2: abs(math.atan(m1) - math.atan(m2))
    geo = all(abs(ang(seq[i] + seq[i + 1], 2 * seq[i + 1]) - ang(seq[i + 1] + seq[i + 2], 2 * seq[i + 1])) < 1e-9
              for i in range(10))
    ok &= geo
    # 2-2: s_n 의 닫힌 형태와 s_n>1, 1/a_n 감소 [증명-수치확인]
    s_expr = sp.simplify((sol[0] - a1) / (a1 - a0))
    ok &= sp.simplify(s_expr - (1 + 4 * a1 ** 2) / (1 + 4 * a0 * a1)) == 0
    s_vals = [(seq[i + 2] - seq[i + 1]) / (seq[i + 1] - seq[i]) for i in range(len(seq) - 2)]
    dd = seq[1] - seq[0]
    ok &= all(v > 1 for v in s_vals) and all(seq[k] >= seq[0] + k * dd - 1e-12 for k in range(len(seq)))
    msgs.append(f"2-1 기하(예각 일치)={geo}; 2-2 s_n>1, a_n≥a_1+(n-1)d 확인 [증명-수치확인]")
    # 2-3: r_n - s_n 의 닫힌 형태 [증명-수치확인]
    rs = sp.simplify(a1 / a0 - s_expr - (a1 - a0) / (a0 * (1 + 4 * a0 * a1)))
    ok &= rs == 0
    msgs.append("2-3 r_n-s_n=(a_{n+1}-a_n)/(a_n(1+4a_na_{n+1})) 항등 [증명-수치확인]")
    # 2-4: a_{n+1}=t a_n 일 때 U/D 를 적분으로 계산, t→lim r_n=lim s_n=3 (2-3)
    t, u, xx = sp.symbols("t u xx", positive=True)
    lo, hi = u, t * u
    chord = lo ** 2 + (lo + hi) * (xx - lo)
    U = sp.integrate(chord - xx ** 2, (xx, lo, hi))
    D = sp.integrate(xx ** 2, (xx, lo, hi))
    ratio = sp.simplify(U / D)
    a24 = sp.sympify(load_answer(2, 3))
    ok &= sp.simplify(sp.nsimplify(ratio.subs(t, 3)) - a24) == 0
    msgs.append(f"2-4 U/D(t)={sp.factor(ratio)} → t=3: {ratio.subs(t, 3)} md={a24}")
    report(2, ok, "; ".join(msgs))


def p03():
    msgs, ok = [], True
    a = sp.symbols("a", real=True)
    fR = (7 - a) * x - 0 + 4 * sp.cos(x) - 2 * (sp.pi + 1)
    fL = -(a + 1) * x ** 2 - (7 - a) * sp.log(1 - x) - 2 * (sp.pi - 1)
    # 미분가능성(경계) 확인
    ok &= sp.simplify(fR.subs(x, 0) - fL.subs(x, 0)) == 0 and sp.simplify(sp.diff(fR, x).subs(x, 0) - sp.diff(fL, x).subs(x, 0)) == 0
    # 3-1: a 를 격자로 훑어 f'>0 (x≥0 은 최솟값 7-a-4, x<0 은 x∈[-10^6,0) 표본 + 점근) 판정 → 경계
    import numpy as np
    def good(av):
        if 7 - av - 4 <= 0:
            return False
        xs = -np.logspace(-6, 6, 4000)
        v = -2 * (av + 1) * xs + (7 - av) / (1 - xs)
        return bool(np.all(v > 0)) and av + 1 >= 0      # x→-∞ 에서 -2(a+1)x + O(1/x): a+1≥0 필요
    grid = [k / 100 for k in range(-300, 800)]
    okset = [g for g in grid if good(g)]
    k1, k2 = min(okset), max(okset) + 0.01
    md = [sp.sympify(v) for v in load_answer(3, 0).split(";")]
    ok &= abs(k1 - float(md[0])) < 1e-9 and abs(k2 - float(md[1])) < 1e-9 and not good(float(md[1]))
    msgs.append(f"3-1 격자 허용 a∈[{k1},{k2}) md={md}")
    # 3-2: h(a) 를 수치로 a=1±ε 에서 풀어 중앙차분
    import mpmath as mp
    mp.mp.dps = 30
    def h(av):
        return mp.findroot(lambda t: (7 - av) * t + 4 * mp.cos(t) - 2 * (mp.pi + 1), 1)
    ok &= abs(h(1) - mp.pi / 3) < 1e-20
    dh = (h(1 + mp.mpf(10) ** -8) - h(1 - mp.mpf(10) ** -8)) / (2 * mp.mpf(10) ** -8)
    a32 = sp.sympify(load_answer(3, 1))
    ok &= abs(dh - mp.mpf(str(sp.N(a32, 30)))) < 1e-10
    msgs.append(f"3-2 h(1)=π/3, 수치 h'(1)={mp.nstr(dh, 12)} md={a32}")
    # 3-3(1): a=1 에서 t 가 접점(원점 통과)인 조건 w=f-xf'=0 의 근을 (-2,6π) 에서 촘촘한 부호변화로 셈
    f1 = sp.Piecewise((fL.subs(a, 1), x < 0), (fR.subs(a, 1), True))
    wL = sp.lambdify(x, (fL - x * sp.diff(fL, x)).subs(a, 1), "mpmath")
    wR = sp.lambdify(x, (fR - x * sp.diff(fR, x)).subs(a, 1), "mpmath")
    w = lambda t: wL(t) if t < 0 else wR(t)
    pts = [mp.mpf(-2) + (6 * mp.pi + 2) * k / 60000 for k in range(1, 60000)]
    vals = [w(t) for t in pts]
    roots = [mp.findroot(w, (pts[i], pts[i + 1]), solver="bisect") for i in range(len(pts) - 1) if vals[i] * vals[i + 1] < 0]
    ok &= len(roots) == 5 and (14 - 6 * mp.log(3) - 2 * mp.pi) > 0
    msgs.append(f"3-3(1) (-2,6π) 근 {len(roots)}개 {[mp.nstr(r, 6) for r in roots]} [증명-수치확인]")
    # 3-3(2): 양의 근 c_n 을 n=1..40 쌍까지 구해 c_{2n}-c_{2n-1} 와 경계 부등식 확인
    cs = []
    for n in range(1, 41):
        cs.append(mp.findroot(wR, (2 * n * mp.pi + mp.mpf(10) ** -12, (2 * n + mp.mpf(1) / 2) * mp.pi), solver="bisect"))
        cs.append(mp.findroot(wR, ((2 * n + mp.mpf(1) / 2) * mp.pi, (2 * n + 1) * mp.pi - mp.mpf(10) ** -12), solver="bisect"))
    # (0,2π] 에 양의 근이 없음(3-3(1))이므로 위 목록이 c_1,c_2,... 순서
    bounds = all(2 * n * mp.pi < cs[2 * n - 2] < 2 * n * mp.pi + mp.mpf(1) / n and
                 (2 * n + 1) * mp.pi - mp.mpf(1) / n < cs[2 * n - 1] < (2 * n + 1) * mp.pi for n in range(1, 41))
    gap = cs[-1] - cs[-2]
    ok &= bounds and abs(gap - mp.pi) < 2 / mp.mpf(40)
    msgs.append(f"3-3(2) 경계 부등식(n≤40)={bounds}, c_80-c_79={mp.nstr(gap, 10)}→π [증명-수치확인]")
    report(3, ok, "; ".join(msgs))


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
