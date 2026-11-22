#!/usr/bin/env python3
"""가천대 2026학년도 수시 논술(실전) 자연계열 D세트 수학(문제 7~15, 보고서 문항카드 64~72) — sympy 검산.
정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산한 뒤, 그 값을 YAML answer.value 문자열과 대조한다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G26D"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items/GCU/PAST"
x, t, h, n = sp.symbols("x t h n", real=True)


def ans(seq):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return str(d["subparts"][0]["answer"]["value"])


def tex(v):
    v = sp.nsimplify(v)
    if v.is_Rational and v.q != 1:
        s = f"\\frac{{{abs(v.p)}}}{{{v.q}}}"
        return ("-" if v < 0 else "") + s
    return str(v)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def has(seq, *vals):
    a = ans(seq).replace(" ", "")
    return all(tex(v) in a for v in vals)


def p07():
    a, b, c = sp.symbols("a b c", positive=True)
    bv = sp.solve(sp.Eq(2 * sp.pi * 3 * b, 6 * sp.pi), b)[0]
    cc = sp.symbols("cc")
    sol = sp.solve([sp.Eq(-a + cc, -2), sp.Eq(a * sp.sin(sp.pi / 2 / (3 * bv)) + cc, sp.Rational(3, 2))], [a, cc], dict=True)[0]
    f = sol[a] * sp.sin(x / (3 * bv)) + sol[cc]
    import math
    vals = [float(f.subs(x, 6 * math.pi * k / 2000)) for k in range(2001)]
    ok_min = abs(min(vals) + 2) < 1e-6 and sp.periodicity(f, x) == 6 * sp.pi
    val = sol[a] * bv * sol[cc]
    report(7, ok_min and has(7, val), f"b={bv}, a={sol[a]}, c={sol[cc]}, abc={val}, 주기·최솟값 확인={ok_min}")


def p08():
    a, b, c = sp.symbols("a b c")
    f = a * (x + 1)
    g = b * (x + c) ** 2
    eqs = sp.Poly(sp.expand(f ** 2 - 4 * g - 4 * x - 3), x).coeffs() + sp.Poly(sp.expand(f - sp.diff(g, x) - 1), x).coeffs()
    sols = [s for s in sp.solve(eqs, [a, b, c], dict=True) if s[b] != 0]
    assert len(sols) == 1
    s = sols[0]
    fx, gx = f.subs(s), g.subs(s)
    # 교점: log2 f = log2 sqrt(g), 진수 조건 f>0, g>0
    roots = [r for r in sp.solve(sp.Eq(fx ** 2, gx), x) if fx.subs(x, r) > 0 and gx.subs(x, r) > 0
             and sp.simplify(fx.subs(x, r) - sp.sqrt(gx.subs(x, r))) == 0]
    report(8, len(roots) == 1 and has(8, roots[0]), f"a,b,c={s[a]},{s[b]},{s[c]}, 교점 x={roots}")


def p09():
    r, m = sp.symbols("r m", positive=True)
    a1 = sp.sqrt(3)
    A = lambda k: a1 * r ** (k - 1)  # noqa: E731
    expr = sp.simplify(A(5) * A(7) / (A(4) * A(6) * A(8)))
    rv = sp.solve(sp.Eq(sp.log(expr, 3), 9), r)
    rv = [sp.nsimplify(v) for v in rv if v.is_positive]
    assert len(rv) == 1
    rv = rv[0]
    ms = [k for k in range(1, 200) if sp.simplify(A(k).subs(r, rv) - sp.Rational(1, 3 ** 9)) == 0]
    expo = sp.nsimplify(sp.log(rv, 3))
    ok = expo == sp.Rational(-19, 10) and ms == [6] and "-\\frac{19}{10}" in ans(9) and "m=6" in ans(9)
    report(9, ok, f"공비=3^({expo}), m={ms}")


def p10():
    tt = sp.symbols("tt", positive=True)
    O, A, B = (0, 0), (1, 1), (tt, sp.sqrt(tt))
    T = sp.Abs((A[0] - O[0]) * (B[1] - O[1]) - (B[0] - O[0]) * (A[1] - O[1])) / 2
    S = sp.integrate(x ** 2, (x, tt, 1))
    Tn = sp.simplify(T.subs(tt, sp.Rational(1, 4)))  # 0<t<1 에서 |t - sqrt t| = sqrt t - t 확인
    assert Tn == (sp.sqrt(sp.Rational(1, 4)) - sp.Rational(1, 4)) / 2
    L2 = sp.limit(S / ((sp.sqrt(tt) - tt) / 2), tt, 1, "-")
    report(10, L2 == sp.Integer(ans(10)), f"T=(sqrt t - t)/2, S={sp.factor(S)}, lim={L2}")


def count_roots_numeric(fun, lo, hi, N=20000):
    import mpmath as mp
    xs = [lo + (hi - lo) * i / N for i in range(N + 1)]
    vals = [fun(v) for v in xs]
    cnt = 0
    for i in range(N):
        if vals[i] == 0:
            cnt += 1
        elif vals[i] * vals[i + 1] < 0:
            cnt += 1
    return cnt


def p11():
    f = sp.Rational(5, 16) * x ** 4 - sp.Rational(5, 2) * x ** 2 + 4

    def g(tv):
        tv = sp.nsimplify(tv)
        rts = set()
        for s in (tv, -tv):
            for r in sp.Poly(f - s, x).real_roots():
                rts.add(r)
        return len(rts)

    right, left, at = g(1 + sp.Rational(1, 10 ** 6)), g(1 - sp.Rational(1, 10 ** 6)), g(1)
    val = right + left + at
    report(11, val == sp.Integer(ans(11)), f"g(1+)={right}, g(1-)={left}, g(1)={at}, 합={val}")


def p12():
    found = []
    # 극한 조건으로 f=36x^2+ax+b(정수) 로 좁힌 뒤, 넓은 정수 범위 전수 탐색으로 (가)(나)(다) 직접 확인
    for a in range(-100, 101):
        for b in range(-100, 101):
            f = 36 * x ** 2 + a * x + b
            if f.subs(x, sp.Rational(-1, 3)) != 0:
                continue
            # 극값(부호 변화): -1/3 이 중근이 아니어야 함
            if sp.diff(f, x).subs(x, sp.Rational(-1, 3)) == 0:
                continue
            # (가)(나): t>=0 에서 (6t+1)^2 < f(t) < (6t+2)^2  ⇔ 차이 다항식이 t>=0 에서 양수
            d1 = sp.Poly(sp.expand(f - (6 * x + 1) ** 2), x)
            d2 = sp.Poly(sp.expand((6 * x + 2) ** 2 - f), x)
            ok = True
            for d in (d1, d2):
                cs = d.all_coeffs()
                # 일차 이하: t>=0 에서 양수 ⇔ 상수항>0, 일차계수>=0
                c1, c0 = (cs[-2] if len(cs) >= 2 else 0), cs[-1]
                if not (c0 > 0 and c1 >= 0):
                    ok = False
            if ok:
                found.append((a, b, f.subs(x, -1)))
    mn = min(v for _, _, v in found)
    report(12, mn == sp.Integer(ans(12)), f"조건 만족 (a,b,f(-1))={found}, 최솟값={mn}")


def p13():
    res = None
    for av in [sp.Rational(k, 4) for k in range(1, 81)]:
        def F(v):
            return v ** 3 if v < 0 else av * v * (v - 2)

        def g(tv):
            c = F(tv)
            cnt = 0
            if c < 0:
                cnt += 1
            for rr in sp.solve(sp.Eq(av * x * (x - 2), c), x):
                if rr.is_real and rr >= 0:
                    cnt += 1
            return cnt

        eps = sp.Rational(1, 10 ** 4)
        cands = sorted(set([-sp.real_root(av, 3), 0, 1, 2]))
        ks = [k for k in cands if abs(g(k + eps) - g(k - eps)) == 2]
        if sum(ks) == 1:
            res = av
            break
    c = sp.symbols("c", positive=True)
    cv = [v for v in sp.solve(sp.Eq((2 * res * c - 2 * res) * (0 - c) + res * c * (c - 2), -9), c) if v > 0][0]
    hx = (2 * res * cv - 2 * res) * (x - cv) + res * cv * (cv - 2)
    area = sp.integrate(res * x * (x - 2) - hx, (x, 0, cv))
    report(13, area == sp.Integer(ans(13)), f"a={res}, k 합=1, 접점 c={cv}, h(x)={sp.expand(hx)}, 넓이={area}")


def p14():
    import math
    good = []
    for a in range(-30, 31):
        def F(v, a=a):
            return (math.cos(v + 5 * math.pi / 6) ** 2 - a * math.cos(v + 4 * math.pi / 3)
                    + a * math.sin(v + 5 * math.pi / 6) - 2 * a - 4)
        # 정확한 근 개수: t=cos(x+pi/3) 이차방정식 근을 이용
        tt = sp.symbols("tt")
        cnt = set()
        for r in sp.solve(tt ** 2 - 2 * a * tt + 2 * a + 3, tt):
            if r.is_real and -1 <= r <= 1:
                for th in {sp.acos(r), 2 * sp.pi - sp.acos(r)}:
                    xv = th - sp.pi / 3
                    if 0 <= xv <= sp.Rational(4, 3) * sp.pi:
                        assert abs(F(float(xv))) < 1e-9
                        cnt.add(round(float(xv), 12))
        if len(cnt) == 2:
            good.append(a)
    ok = good == [-3, -2] and all(f"a={v}" in ans(14).replace(" ", "") for v in good)
    report(14, ok, f"근 2개인 정수 a={good} (원식에 근 대입 확인)")


def p15():
    a, b = sp.symbols("a b")
    f = x ** 3 + a * x ** 2 + b * x + 14
    sol = sp.solve([sp.Eq(sp.diff(f, x).subs(x, 2), -4), sp.Eq(f.subs(x, 2), -4 * 2 + 24)], [a, b], dict=True)[0]
    fs = f.subs(sol)
    crit = sorted(sp.solve(sp.diff(fs, x), x))
    mx = fs.subs(x, crit[0]) if sp.diff(fs, x, 2).subs(x, crit[0]) < 0 else None
    mn = fs.subs(x, crit[1]) if sp.diff(fs, x, 2).subs(x, crit[1]) > 0 else None
    report(15, has(15, mx, mn), f"a={sol[a]}, b={sol[b]}, 극대 f({crit[0]})={mx}, 극소 f({crit[1]})={mn}")


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
