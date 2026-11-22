#!/usr/bin/env python3
"""KNU 2022학년도 수시 논술(AAT) 자연계열Ⅰ 수학 (KNU-G22A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-G22A"
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


def p01():
    msgs, ok = [], True
    tang = 2 * c * x - c ** 2
    low = -x ** 2 + 4 * x - 3
    # 1-1: 교점 방정식의 판별식 < 0 인 c 의 집합
    disc = sp.discriminant(sp.expand(tang - low), x)
    sol = sp.solve_univariate_inequality(disc < 0, c, relational=False)
    al, be = sol.left, sol.right
    md = [sp.sympify(v) for v in load_answer(1, 0).split(";")]
    ok &= sp.simplify(al - md[0]) == 0 and sp.simplify(be - md[1]) == 0 and sol.left_open and sol.right_open
    msgs.append(f"1-1 ({al},{be}) md={md}")
    # 1-2: 곡선 위 점 b 에서 직선 2cx-y-c^2=0 까지 거리의 최솟값(수치: 여러 c 에서 b 최소화)
    b = sp.symbols("b", real=True)
    fmd = sp.sympify(load_answer(1, 1), locals={"c": c})
    good = True
    for cv in [al + (be - al) * k / 7 for k in range(1, 7)]:
        dist = sp.Abs(2 * cv * b - low.subs(x, b) - cv ** 2) / sp.sqrt(4 * cv ** 2 + 1)
        bs = [float(al) - 5 + i * 0.001 for i in range(10001)]
        dmin = min(float(dist.subs(b, t)) for t in bs[::10])
        good &= abs(dmin - float(fmd.subs(c, cv))) < 1e-3
    ok &= good
    msgs.append(f"1-2 격자 최솟값 일치={good}")
    # 1-3: 격자에서 S 를 직접 판정(점 (X,Y) 가 어떤 c∈[α,β] 의 접선 위 & 두 곡선 사이) — 세로 단면 길이 적분
    #  고정 X 에서 {2cX-c^2 : c∈[α,β]} 는 구간(연속함수 상) → 끝 = 그 함수의 최대·최소(격자 c 로 계산)
    import numpy as np
    A, B = float(al), float(be)
    cs = np.linspace(A, B, 4001)
    Xs = np.linspace(A, B, 4001)
    tot = 0.0
    for X in Xs:
        vals = 2 * cs * X - cs ** 2
        lo, hi = max(vals.min(), -X ** 2 + 4 * X - 3), min(vals.max(), X ** 2)
        tot += max(hi - lo, 0)
    area = tot * (B - A) / (len(Xs) - 1)
    a13 = sp.sympify(load_answer(1, 2))
    ok &= abs(area - float(a13)) < 2e-3
    msgs.append(f"1-3 수치 넓이={area:.5f} md={a13}={float(a13):.5f}")
    report(1, ok, "; ".join(msgs))


def semicircle_counts(n):
    """S_n 을 실제 좌표로 만들고 공선 판정으로 직선 수·삼각형 수를 센다(고정밀)."""
    import mpmath as mp
    mp.mp.dps = 40
    N = 2 ** n
    pts = [(mp.mpf(-1) + 2 * mp.mpf(k) / N, mp.mpf(0)) for k in range(N + 1)] + \
          [(mp.cos(mp.pi * k / N), mp.sin(mp.pi * k / N)) for k in range(1, N)]
    col = lambda P, Q, R: abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) < mp.mpf(10) ** -30
    lines = set()
    for i, j in itertools.combinations(range(len(pts)), 2):
        lines.add(frozenset(k for k in range(len(pts)) if col(pts[i], pts[j], pts[k])))
    tri = sum(1 for t in itertools.combinations(range(len(pts)), 3) if not col(*[pts[i] for i in t]))
    return len(lines), tri


def p02():
    msgs, ok = [], True
    cnt = {n: semicircle_counts(n) for n in (1, 2, 3, 4)}
    md = [int(v) for v in load_answer(2, 0).split(";")]
    ok &= list(cnt[3]) == md
    msgs.append(f"2-1 열거 (a3,b3)={cnt[3]} md={md}")
    # c_n: n=1..4 는 열거값으로, 일반 n 은 공선점이 지름뿐이라는 열거 관찰을 일반화한 조합식으로 계산
    cs = {n: sp.Rational(9 * b, 7 * (a - 1)) + sp.Rational(5, 7) for n, (a, b) in cnt.items()}
    ok &= all(cs[n] == 2 ** n for n in cs)
    m = sp.symbols("m", positive=True, integer=True)
    an = sp.binomial(2 ** (m + 1), 2) - sp.binomial(2 ** m + 1, 2) + 1
    bn = sp.binomial(2 ** (m + 1), 3) - sp.binomial(2 ** m + 1, 3)
    cgen = sp.simplify(sp.expand_func(9 * bn / (7 * (an - 1)) + sp.Rational(5, 7)))
    ok &= sp.simplify(cgen - 2 ** m) == 0
    msgs.append(f"c_n(n≤4 열거)={list(cs.values())}, 일반식 c_n={cgen}")
    # 2-2: n=100 에서 넓이 이등분 원점 직선을 직접 풀기
    n = 100
    cn = lambda k: cgen.subs(m, k)
    A, B, C = (cn(n), 0), (cn(n + 1), 0), (0, cn(n + 2))
    d = sp.symbols("d", positive=True)
    def meet(P, Q):  # 직선 y=dx 와 선분 PQ 의 교점 x
        t = sp.symbols("t")
        tt = sp.solve(sp.Eq(P[1] + t * (Q[1] - P[1]), d * (P[0] + t * (Q[0] - P[0]))), t)[0]
        return P[0] + tt * (Q[0] - P[0]), P[1] + tt * (Q[1] - P[1]), tt
    Px, Py, tp = meet(A, C)
    Qx, Qy, tq = meet(B, C)
    areaPQC = sp.Abs((Px - C[0]) * (Qy - C[1]) - (Qx - C[0]) * (Py - C[1])) / 2
    areaABC = sp.Rational(1, 2) * (B[0] - A[0]) * C[1]
    sols = [s for s in sp.solve(sp.Eq(((Px - C[0]) * (Qy - C[1]) - (Qx - C[0]) * (Py - C[1])) ** 2 / 4, (areaABC / 2) ** 2), d)
            if s.is_positive and 0 < tp.subs(d, s) < 1 and 0 < tq.subs(d, s) < 1]
    dv = sp.nsimplify(sp.simplify(sols[0]))
    lg = sp.simplify(sp.log(sp.simplify(Qx.subs(d, dv) - Px.subs(d, dv)), 8))
    a1, a2 = sp.sympify(load_answer(2, 1)), sp.sympify(load_answer(2, 2))
    ok &= len(sols) == 1 and sp.simplify(lg - a1) == 0 and sp.simplify(dv - a2) == 0
    msgs.append(f"2-2(1) log8(q-p)={lg} md={a1}; 2-2(2) d={dv} md={a2}")
    report(2, ok, "; ".join(msgs))


def p03():
    msgs, ok = [], True
    al, be, ga = sp.symbols("alpha beta gamma", real=True)
    fp = (x - al) * (x - be) * (x - ga)
    f = sp.Rational(1, 4) * (x - (al + be + ga) / 3) * fp - sp.Rational(1, 3) * (x - al) ** 2 + 2
    # 조건: f 의 도함수가 실제로 fp (항등식) — 계수 비교
    eqs = sp.Poly(sp.expand(sp.diff(f, x) - fp), x).all_coeffs()
    sols = sp.solve(eqs, [be, ga], dict=True)
    real = [s for s in sols if all(v.is_real is not False for v in s.values())]
    # α≤β≤γ 를 만족하는 해(α 임의 수치로 확인)
    ordered = [s for s in real if (s[be] - al).subs(al, 1) >= 0 and (s[ga] - s[be]).subs(al, 1) >= 0]
    ok &= len(ordered) == 1 and sp.simplify(ordered[0][be] - al) == 0 and sp.simplify(ordered[0][ga] - al) != 0
    msgs.append(f"3-1 항등식 해={ordered} → α=β≠γ [증명-수치확인]")
    gmd = sp.sympify(load_answer(3, 1))
    gv = sp.simplify(ordered[0][ga] - al)
    ok &= gv == gmd
    msgs.append(f"3-2 γ-α={gv} md={gmd}")
    F = f.subs(ordered[0])
    av = [r for r in sp.solve(sp.Eq(F.subs(x, al), F.subs(x, 2 * al)), al) if r != 0]
    a1 = sp.sympify(load_answer(3, 2))
    ok &= av == [a1]
    msgs.append(f"3-3(1) α={av} md={a1}")
    Fa = F.subs(al, av[0])
    crit = sp.solve(sp.diff(Fa, x), x)
    kmax = min(Fa.subs(x, r) for r in crit)          # 사차(양의 최고차) → 전역 최솟값
    a2 = sp.sympify(load_answer(3, 3))
    ok &= sp.simplify(kmax - a2) == 0
    msgs.append(f"3-3(2) min f={kmax} md={a2}")
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
