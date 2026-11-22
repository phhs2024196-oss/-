#!/usr/bin/env python3
"""서울과기대 2026학년도 모의논술고사 문제 1~3 (SNUT-M26-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SNUT-M26"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
SYM = {"x": x, "t": t, "n": n, "k": k, "pi": sp.pi, "e": sp.E}


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


def same(a, b, tol=1e-9):
    """기호 동일 또는 수치 근사 동일."""
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    return abs(complex(sp.N(a - b, 30))) < tol


import itertools
import math
from fractions import Fraction


def p01():
    # [1.1] 정12각형 꼭짓점 좌표로 직각 삼각형 전수(내적 0)
    pts = [(math.cos(2*math.pi*i/12), math.sin(2*math.pi*i/12)) for i in range(12)]
    def right(a, b, c):
        for p, q, r in ((a, b, c), (b, a, c), (c, a, b)):
            v1 = (q[0]-p[0], q[1]-p[1]); v2 = (r[0]-p[0], r[1]-p[1])
            if abs(v1[0]*v2[0] + v1[1]*v2[1]) < 1e-9:
                return True
        return False
    cnt = sum(1 for tri in itertools.combinations(pts, 3) if right(*tri))
    ok1 = cnt == int(load_answer(1, 0))
    report(1, ok1, f"[1.1] 정12각형 전수 직각삼각형 수={cnt}, yaml={load_answer(1,0)}")
    # [1.2]
    a, d = sp.symbols("a d")
    S = lambda nn: sp.Rational(nn, 2)*(2*a + (nn - 1)*d)
    sol = sp.solve([sp.Eq(S(4), 214), sp.Eq(S(6), 303)], [a, d], dict=True)[0]
    seq = [sol[a] + (i - 1)*sol[d] for i in range(1, 41)]
    tot = sum(abs(v) for v in seq)
    ok2 = same(tot, load_answer(1, 1))
    report(1, ok2, f"[1.2] a={sol[a]}, d={sol[d]}, Σ|a_n|(n≤40)={tot}, yaml={load_answer(1,1)}")
    # [1.3] 급수 조건을 만족하는 서로 다른 두 수열로 극한 계산(Fraction 부분합)
    def run(c):
        N = 200000
        base = lambda n_: Fraction(2*n_*n_ + 2*n_ - 1, n_*n_ + n_)
        sa = Fraction(0); ssum = Fraction(0); an = None
        for n_ in range(1, N + 1):
            an = base(n_) + c(n_)
            sa += base(n_) + c(n_) - 2
        return float(an + sa)
    # c_n 의 합이 5 가 되도록: (i) 5/(n(n+1)) → 합 5, (ii) 5·2^{-n} → 합 5(무한)
    v1 = run(lambda n_: Fraction(5, n_*(n_ + 1)))
    v2 = run(lambda n_: Fraction(5, 2**n_) if n_ < 70 else Fraction(0))
    ans = float(load_answer(1, 2))
    ok3 = abs(v1 - ans) < 1e-3 and abs(v2 - ans) < 1e-3
    report(1, ok3, f"[1.3] 예시 수열 2개(N=200000)에서 값={v1:.5f}, {v2:.5f}, yaml={ans}")
    # [1.4]
    X = sp.symbols("X", positive=True)
    dist2 = X**2 + (1 - X**2)**2
    crit = [c for c in sp.solve(sp.diff(dist2, X), X) if c > 0]
    xt = [c for c in crit if sp.diff(dist2, X, 2).subs(X, c) > 0][0]          # 최소 거리 → 접점
    r2 = sp.simplify(dist2.subs(X, xt)); r = sp.sqrt(r2)
    circ = sp.sqrt(r2 - X**2)
    S1 = sp.integrate(1 - X**2 - circ, (X, 0, xt))
    S2 = sp.integrate(1 - X**2 - circ, (X, xt, r)) + sp.integrate(1 - X**2, (X, r, 1))
    tot = sp.simplify(S1 + S2)
    ok4 = same(tot, load_answer(1, 3))
    report(1, ok4, f"[1.4] 접점 x={xt}, r={r}, S1={sp.simplify(S1)}, S2={sp.simplify(S2)}, 합={tot}≈{float(tot):.9f}, yaml={load_answer(1,3)}")


def p02():
    import random
    import numpy as np
    raw = load_answer(2, 0, raw=True)
    g = [sp.sympify(z.split("=")[1], locals=SYM) for z in raw.split(";")]
    random.seed(7)
    okg = True
    for _ in range(300):
        nn = random.randint(1, 6)
        xv = random.uniform(-40, 40)
        f = abs(xv + 2*nn*math.pi) + abs(xv - 2*nn*math.pi) + 1 - 4*nn*math.pi
        L = 2*nn*math.pi
        pick = g[0] if xv < -L else (g[1] if xv <= L else g[2])
        okg &= abs(float(pick.subs({x: xv, n: nn})) - f) < 1e-9
    report(2, okg, f"[2.1] 무작위 300점에서 구간별 식이 원식과 일치={okg}, yaml={raw}")
    # [2.2] 교점의 개수: d=f-cos ≥ 0 의 영점(국소 최소 ≈0) 개수
    def zeros(nn):
        L = 2*nn*math.pi
        xs = np.linspace(-L - 10, L + 10, 400001)
        fv = np.abs(xs + L) + np.abs(xs - L) + 1 - 2*L
        d = fv - np.cos(xs)
        assert d.min() > -1e-12
        idx = [i for i in range(1, len(xs) - 1) if d[i] <= d[i-1] and d[i] <= d[i+1] and d[i] < 1e-6]
        # 군집화
        cl = []
        for i in idx:
            if not cl or xs[i] - cl[-1] > 0.5:
                cl.append(xs[i])
        return len(cl)
    expr = sp.sympify(load_answer(2, 1, raw=True).split("=")[1], locals=SYM)
    bad = [nn for nn in range(1, 7) if zeros(nn) != int(expr.subs(n, nn))]
    report(2, not bad, f"[2.2] n=1..6 수치 교점 수={[zeros(nn) for nn in range(1,7)]}, yaml a_n={expr}, 불일치 n={bad}")
    # [2.3]
    tot = sum(sp.Rational(1, 1)/((expr.subs(n, i))*(expr.subs(n, i + 1))) for i in range(1, 27))
    report(2, same(tot, load_answer(2, 2)), f"[2.3] Σ_(n=1)^26 1/(a_n a_(n+1))={tot}, yaml={load_answer(2,2)}")


def p03():
    a, b = sp.symbols("a b", real=True)
    f = lambda z: a*z**3
    g = lambda z: 3*z**2 + b
    sol = sp.solve([sp.Eq(sp.diff(f(x), x).subs(x, 2), sp.diff(g(x), x).subs(x, 2)), sp.Eq(f(2), g(2))], [a, b], dict=True)[0]
    raw = load_answer(3, 0, raw=True)
    ya, yb = [sp.sympify(z.split("=")[1]) for z in raw.split(";")]
    report(3, len(sol) == 2 and same(sol[a], ya) and same(sol[b], yb), f"[3.1] a={sol[a]}, b={sol[b]}, yaml=({ya},{yb})")
    F = x**3; G = 3*x**2 - 4
    line = sp.diff(F, x).subs(x, 2)*(x - 2) + F.subs(x, 2)
    rawl = load_answer(3, 1, raw=True)
    l_str, p_str = [z.strip() for z in rawl.split(";")]
    yline = sp.sympify(l_str.split("=")[1], locals=SYM)
    ypt = sp.sympify(p_str)   # 튜플
    inter = [(r, F.subs(x, r)) for r in sp.solve(sp.Eq(F, line), x)]
    q3 = [(r, v) for r, v in inter if r < 0 and v < 0]
    ok2 = same(line, yline) and len(q3) == 1 and tuple(q3[0]) == tuple(ypt)
    report(3, ok2, f"[3.2] 접선 y={sp.expand(line)}, 교점들={inter}, 제3사분면={q3}, yaml={l_str} ; {p_str}")
    # [3.3] 영역 {l ≤ y ≤ min(F,G)}, x∈[-4,2] 에서 넓이(구간별 기호 적분, 격자로 위치 확인)
    xs = [sp.Rational(i, 10) for i in range(-39, 20)]
    ordered = all(sp.Min(F, G).subs(x, v) - line.subs(x, v) >= 0 for v in xs)
    # 위쪽 경계는 F,G 교점(-1,2)에서 교대
    cross = sorted(sp.solve(sp.Eq(F, G), x))
    area = sp.integrate(F - line, (x, -4, cross[0])) + sp.integrate(G - line, (x, cross[0], 2))
    # 수치 확인: 격자 적분
    import numpy as np
    xx = np.linspace(-4, 2, 600001)
    Fv = xx**3; Gv = 3*xx**2 - 4; lv = 12*xx - 16
    num = float(np.sum(np.maximum(0, np.minimum(Fv, Gv) - lv))*(xx[1] - xx[0]))
    ok3 = ordered and same(area, load_answer(3, 2)) and abs(num - float(area)) < 1e-3
    report(3, ok3, f"[3.3] F,G 교점={cross}, 구간합 넓이={area}, 격자 적분≈{num:.4f}, yaml={load_answer(3,2)}")


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
