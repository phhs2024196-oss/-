#!/usr/bin/env python3
"""가천대 2026학년도 수시 논술(실전) 자연계열 C세트 수학(문제 7~15, 보고서 문항카드 55~63) — sympy 검산.
정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산한 뒤, 그 값을 YAML answer.value 문자열과 대조한다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G26C"
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
    a_of_x = sp.expand_trig(1 - 5 * sp.cos(sp.pi / 2 - x) ** 2 - 4 * sp.cos(x))  # 방정식에서 a 분리
    g = sp.simplify(a_of_x.subs(sp.sin(x) ** 2, 1 - sp.cos(x) ** 2)).subs(sp.cos(x), t)
    cands = [g.subs(t, v) for v in [-1, 1] + [r for r in sp.solve(sp.diff(g, t), t) if -1 <= r <= 1]]
    mx, mn = max(cands), min(cands)
    # 수치 격자로도 치역 확인
    import math
    vals = [1 - 5 * math.sin(k / 1000) ** 2 - 4 * math.cos(k / 1000) for k in range(0, 6284)]
    ok = abs(max(vals) - float(mx)) < 1e-5 and abs(min(vals) - float(mn)) < 1e-5 and has(7, mx, mn)
    report(7, ok, f"a=5t^2-4t-4 on [-1,1]: max={mx}, min={mn} (격자 확인)")


def p08():
    a = sp.Rational(47, 45)
    for k in range(1, 2025):
        a += (-1) ** k * sp.Rational(2 * k + 1, k * (k + 1))
    p, q = a.q, a.p
    val = sp.sqrt(p) + q
    report(8, val == sp.Integer(ans(8)), f"a_2025={a}, p={p}, q={q}, sqrt(p)+q={val}")


def p09():
    a = sp.symbols("a", positive=True)
    f = a * x ** 3
    tan_at_4 = sp.diff(f, x).subs(x, 5) * (4 - 5) + f.subs(x, 5)
    av = sp.solve(sp.Eq(tan_at_4, 2), a)[0]
    fx = f.subs(a, av)
    g = sp.cbrt(x / av)  # 역함수(x>=0 부분)
    # 제1사분면: f 와 g 사이 넓이를 직접 적분, 원점 대칭으로 2배
    xs = sorted(sp.solve(sp.Eq(fx, x), x))
    area_q1 = sp.integrate(g - fx, (x, 0, xs[-1]))
    area = sp.nsimplify(2 * area_q1)
    report(9, area == sp.Integer(ans(9)), f"a={av}, 교점={xs}, 제1사분면 g-f 적분={sp.nsimplify(area_q1)}, 전체={area}")


def p10():
    a = sp.log(27, 4)
    b = sp.log(25, 9)
    P, Q = sp.symbols("P Q")
    pv = sp.solve(sp.Eq(sp.log(10, 5), 1 + 1 / (P * a * b)), P)[0]
    qv = sp.solve(sp.Eq(sp.log(10, 6), (1 + pv * a * b) / (1 + Q * a)), Q)[0]
    pv, qv = sp.nsimplify(sp.simplify(pv)), sp.nsimplify(sp.simplify(qv))
    report(10, has(10, pv, qv), f"p={pv}, q={qv}")


def p11():
    total = 0
    for k in range(1, 4):
        px = sp.solve(sp.Eq(sp.log(x - 1, sp.Rational(1, 3)), k), x)[0]
        qx = sp.solve(sp.Eq(sp.log(3 - x, sp.Rational(1, 9)), k), x)[0]
        # R_k = (2, k-1): R_1 은 두 그래프 교점, R_{k} 는 선분 P_{k-1}Q_{k-1} 위 x=2
        assert sp.log(2 - 1, sp.Rational(1, 3)) == 0 and sp.log(3 - 2, sp.Rational(1, 9)) == 0
        ak = sp.Rational(1, 2) * (qx - px) * 1
        bk = 1 - sp.Rational(1, 2 * 3 ** k) - ak
        total += bk
    val = sp.nsimplify(1458 * total)
    report(11, val == sp.Integer(ans(11)), f"S_3={sp.nsimplify(total)}, 1458S_3={val}")


def p12():
    a, b = sp.symbols("a b")
    F = x ** 4 + a * x ** 3 + b * x ** 2 + 52 * x
    f = sp.diff(F, x)
    eqs = [sp.Eq(F.subs(x, 1), 0), sp.Eq(b, sp.integrate(t * sp.diff(f, x).subs(x, t), (t, 2, 3)))]
    sol = sp.solve(eqs, [a, b], dict=True)[0]
    f2 = f.subs(sol).subs(x, 2)
    # 원식 확인: ∫_1^x f = F 이고 b 가 정의대로인지
    fs = f.subs(sol)
    chk = sp.simplify(sp.integrate(fs.subs(x, t), (t, 1, x)) - F.subs(sol)) == 0
    report(12, chk and has(12, sol[a], f2), f"a={sol[a]}, b={sol[b]}, f(2)={f2}, 항등식 확인={chk}")


def p13():
    # lim x/(f-x)=1/2 를 만족하는 임의의 f: f(x)=3x+c x^2 (c 여러 값)로 확인
    vals = set()
    for c in (0, 1, -2, 5):
        f = 3 * x + c * x ** 2
        assert sp.limit(x / (f - x), x, 0) == sp.Rational(1, 2)
        vals.add(sp.limit((35 * x * f + 5 * f ** 2) / (2 * f ** 2 + 3 * x * f - 2 * x ** 2), x, 0))
    v = vals.pop() if len(vals) == 1 else None
    report(13, v is not None and v == sp.Integer(ans(13)), f"극한값={v} (f=3x+cx^2, c=0,1,-2,5 모두 동일)")


def p14():
    tot = 0
    for k in range(1, 10):
        f = -sp.log((k + 1) * x, 2)
        g = sp.log((k + 1) / (k * x), sp.Rational(1, 2))
        L = sp.log(k, 2)
        # 수치로 h=max(f,g) 와 y=L 의 교점 탐색(이분법)
        hf = sp.lambdify(x, sp.Max(f, g) - L, "mpmath")
        import mpmath as mp
        roots = []
        grid = [mp.mpf(10) ** (e / 50) for e in range(-200, 101)]
        for u, v in zip(grid, grid[1:]):
            if hf(u) * hf(v) <= 0:
                roots.append(mp.findroot(hf, (u, v), solver="bisect"))
        assert len(roots) == 2, (k, roots)
        exact = [sp.Rational(1, k * (k + 1)), k + 1]
        assert all(abs(r - e) < 1e-12 for r, e in zip(sorted(roots), exact))
        tot += exact[1] - exact[0]
    val = sp.Rational(10, 3) * tot
    report(14, val == sp.Integer(ans(14)), f"sum l(k)={tot}, 10/3*sum={val} (교점 수치 확인)")


def p15():
    a, b = sp.symbols("a b")
    f = x ** 2 + a * x + b
    S = sum(sp.limit((f.subs(x, k + 1 + h) - f.subs(x, k + 1 - 2 * h)) / (5 * h), h, 0) for k in range(1, 16))
    av = sp.solve(sp.Eq(S, 90), a)[0]
    bv = sp.solve(sp.Eq(f.subs(x, 2), sp.diff(f, x).subs(x, 2)).subs(a, av), b)[0]
    val = f.subs({a: av, b: bv}).subs(x, av + bv)
    report(15, val == sp.Integer(ans(15)), f"a={av}, b={bv}, f(a+b)={val}")


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
