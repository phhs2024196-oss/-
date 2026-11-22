#!/usr/bin/env python3
"""가천대 2023학년도 수시 논술(실전) 자연C 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-G23C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p07():
    # f 는 다항식, F' = f, F = f + 2x^3 - 5x^2 - 5x  →  f = f' + 6x^2 - 10x - 5. 차수 0~4 일반 다항식으로 풀이
    cs = sp.symbols("c0:5")
    f = sum(c * x ** i for i, c in enumerate(cs))
    sol = sp.solve(sp.Poly(f - (sp.diff(f, x) + 6 * x ** 2 - 10 * x - 5), x).all_coeffs(), cs, dict=True)
    fx = f.subs(sol[0])
    roots = sp.solve(sp.Eq(fx, 4 * x ** 2 + 3), x)
    val = sp.simplify(sum(r ** 2 for r in roots))
    report(7, val == 7 and len(sol) == 1, f"f={sp.expand(fx)}, 근={roots}, α²+β²={val}")


def p08():
    k = sp.symbols("k")
    f = -x ** 3 + 3 * x ** 2 + 9 * x + k
    crit = sp.solve(sp.diff(f, x), x)
    a = [c for c in crit if sp.diff(f, x, 2).subs(x, c) < 0][0]
    b = [c for c in crit if sp.diff(f, x, 2).subs(x, c) > 0][0]
    kk = sp.solve(f.subs(x, b), k)[0]
    report(8, (a, b, kk) == (3, -1, 5), f"극대 a={a}, 극소 b={b}, f(b)=0 → k={kk}")


def p09():
    import numpy as np
    xs = np.linspace(0, 2, 200001)
    circ_top = np.sqrt(np.clip(1 - (xs - 1) ** 2, 0, None))

    def meets(base):  # y=base^x (x≥0, y>0) 와 원의 위쪽 반원(0≤y≤√(1-(x-1)^2)) 교차 여부: 곡선이 원판 안/경계로 들어오는가
        ys = base ** xs
        return bool(np.any(ys <= circ_top + 1e-12))

    good = [a for a in range(2, 200) if meets(3 * a ** 0.5 / 10) and not meets(3 * a / 7)]
    # 기호 경계: 3√a/10 ≤ 1, 3a/7 > 1
    sym = [a for a in range(2, 200) if sp.Rational(3, 10) * sp.sqrt(a) <= 1 and sp.Rational(3, 7) * a > 1]
    report(9, len(good) == 9 and good == sym, f"수치 교차판정 a={good}, 밑 조건 a={sym}, 개수={len(good)}")


def p10():
    a, b, c, p = sp.symbols("a b c p")
    g_in = a * x ** 2 + b * x + c                      # |x|≠1 에서 g = f/(x^2-1)
    f = (x ** 2 - 1) * g_in
    eqs = [sp.Eq(g_in.subs(x, 1), 1 + p),               # x=1 연속
           sp.Eq(g_in.subs(x, -1), -1 + p),             # x=-1 연속
           sp.Eq(1 + p + 1, 0)]                         # 극한 존재 → g(1)+1=0
    s1 = sp.solve(eqs, [p, b, a + c], dict=True)[0]
    psol, bsol = s1[p], s1[b]
    csol = sp.solve(sp.Eq(a + bsol + c, -1), c)[0]      # c = -a - b - 1 형태
    lim = sp.limit(((g_in + 1) / f).subs({b: bsol, c: csol}), x, 1)
    asol = sp.solve(sp.Eq(lim, 1), a)[0]
    csol = csol.subs(a, asol)
    gp = g_in.subs({a: asol, b: bsol, c: csol, x: psol})
    ok = (psol, bsol, csol, gp) == (-2, 1, sp.Rational(-1, 2), sp.Rational(-17, 2))
    report(10, ok, f"① p={psol}, ② b={bsol}, ③ c={csol} (a={asol}), ④ g(p)={gp}")


def p11():
    a = sp.symbols("a", positive=True)
    Ax = sp.solve(sp.Eq(4 + sp.log(x + 2, a), 5), x)[0]
    By = (a ** (x + 2) + 4).subs(x, -1)
    P = (-1, 5)
    area = sp.Rational(1, 2) * sp.Abs(Ax - P[0]) * sp.Abs(By - P[1])
    sols = [s for s in sp.solve(sp.Eq(area.subs(sp.Abs(Ax + 1), Ax + 1).subs(sp.Abs(By - 5), By - 5), 2), a) if s > 1]
    report(11, sols == [3], f"A=({Ax},5), B=(-1,{By}), 넓이=½(a-1)², a={sols}")


def p12():
    s = sp.symbols("s", real=True)
    expr = 1 - s ** 2 + 4 * s                       # cos²x+4sinx (k 제외)
    mx = max(expr.subs(s, v) for v in [-1, 1] + [r for r in sp.solve(sp.diff(expr, s), s) if -1 <= r <= 1])
    kmax = -mx
    report(12, kmax == -4, f"max_(s∈[-1,1]) (1-s²+4s)={mx} → k 최댓값={kmax}")


def p13():
    a = sp.symbols("a", positive=True)
    asol = [v for v in sp.solve(sp.sqrt(4 + a) - 5, a) if v > 0][0]
    f = (sp.sqrt(x ** 2 + asol) - 5) / ((x - 2) * (x ** 2 + 4))
    f2 = sp.limit(f, x, 2)
    val = sp.simplify(asol / f2)
    report(13, val == 420, f"a={asol}, f(2)={f2}, a/f(2)={val}")


def p14():
    a, b = sp.symbols("a b")
    f = lambda s: a * s + b  # noqa: E731
    g = x * sp.integrate(t * f(t) - 11, (t, 0, x)) - sp.integrate(t ** 2 * f(t), (t, 0, x))
    gp = sp.diff(g, x)
    sol = sp.solve([gp.subs(x, 2), g.subs(x, 2) + 20], [a, b], dict=True)[0]
    gps = sp.factor(gp.subs(sol))
    crit = sp.solve(gps, x)
    g2 = sp.diff(g, x, 2).subs(sol)
    mins = [c for c in crit if g2.subs(x, c) > 0]
    others = [c for c in mins if c != 2]
    ok = sol == {a: 12, b: 6} and 2 in mins and others == [sp.Rational(-11, 4)]
    report(14, ok, f"f(x)={sol[a]}x+{sol[b]}, g'={gps}, 극소점={mins} → k={others}")


def p15():
    nn = sp.symbols("n", positive=True, integer=True)
    r = 1 / sp.Integer(3) ** nn
    m = sp.Rational(-2, 3)
    c = sp.symbols("c", positive=True)               # 제1사분면 접선 y = m x + c (c>0), 원점 거리 = r
    cn = sp.solve(sp.Eq(c / sp.sqrt(1 + m ** 2), r), c)[0]
    xint = cn / (-m)
    an = sp.simplify(sp.Rational(1, 2) * cn * xint)
    a1 = an.subs(nn, 1)
    S = sp.nsimplify(sum(an.subs(nn, k) for k in range(1, 11)))
    target_l = sp.sqrt(13) / sp.Integer(3) ** (nn + 1)
    target_S = sp.Rational(13, 96) * (1 - sp.Rational(1, 3) ** 20)
    ok = (sp.simplify(cn - target_l) == 0 and sp.simplify(an - sp.Rational(13, 12) / sp.Integer(9) ** nn) == 0
          and a1 == sp.Rational(13, 108) and S == target_S)
    report(15, ok, f"① y절편={sp.simplify(cn)}, ② a_n={an}, ③ a1={a1}, ④ 합={S}")


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
