#!/usr/bin/env python3
"""HIU-G24S (홍익대 2024학년도 수시 논술 자연계열 세종) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G24S"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, r = sp.symbols("a r", real=True)
    n, k = sp.symbols("n k", integer=True, positive=True)
    sols = [s for s in sp.solve([a * r**2 - 45, a * r**7 - 9 * a * r**5], [a, r], dict=True)
            if s[a] > 0 and s[r] > 0]
    s = sols[0]
    Sn = sp.summation(s[a] * s[r]**(k - 1), (k, 1, n))
    md = sp.sympify(md_answer(1, "1"), locals={"n": n})
    ok = len(sols) == 1 and all(sp.simplify(Sn.subs(n, v) - md.subs(n, v)) == 0 for v in range(1, 12))
    report(1, ok, f"a={s[a]} r={s[r]} S_n={sp.simplify(Sn)}")


def p02():
    th = sp.symbols("theta", real=True)
    import math
    # tan<0, cos=-3/5 인 θ ∈ [0,2π)
    cands = [v for v in sp.solve(sp.Eq(sp.cos(th), sp.Rational(-3, 5)), th)]
    cands = [sp.nsimplify(v) for v in cands] + [2 * sp.pi - v for v in cands]
    good = [v for v in cands if float(sp.tan(v)) < 0]
    vals = {sp.nsimplify(sp.simplify(sp.sin(v) + 1 / sp.tan(v))) for v in good}
    ok = len(vals) == 1 and sp.simplify(list(vals)[0] - sp.sympify(md_answer(2, "1"))) == 0
    report(2, ok, f"theta={[round(float(v), 6) for v in good]} value={vals}")


def p03():
    b = sp.symbols("b", real=True)
    f = 3**x
    g = sp.Rational(1, 3)**(x - 1) + 2
    # u = 3^x (>0) 치환: g = 3*3^(-x) + 2 → u = 3/u + 2
    u = sp.symbols("u", positive=True)
    eq_u = sp.together(u - (3 / u + 2))
    assert abs(float(g.subs(x, 0.37)) - float((3 / u + 2).subs(u, 3**0.37))) < 1e-12  # 치환 확인
    us = sp.solve(sp.numer(eq_u), u)
    xs = [sp.log(v, 3) for v in us]
    x0 = sp.simplify(xs[0])
    assert abs(float(f.subs(x, x0) - g.subs(x, x0))) < 1e-12
    bval = sp.solve(sp.Eq(9**x0 + b, f.subs(x, x0)), b)[0]
    ok = len(xs) == 1 and sp.simplify(bval - sp.sympify(md_answer(3, "1"))) == 0
    report(3, ok, f"intersection x={x0}, y={f.subs(x, x0)}, b={bval}")


def p04():
    C = sp.symbols("C", real=True)
    left = sp.limit(sp.Abs(2 * x + 3), x, 1, "-")
    Cs = sorted(sp.solve(sp.Eq(sp.Abs(7 - C), left), C))
    C1, C2 = Cs
    g = x**3 - C2 * x + C1
    pts = [p for p in sp.solve(sp.diff(g, x), x) if -3 <= p <= 3] + [-3, 3]
    mn = min(g.subs(x, p) for p in pts)
    ok = sp.simplify(mn - sp.sympify(md_answer(4, "1"))) == 0
    # 격자 확인
    ok = ok and all(float(g.subs(x, -3 + 6 * i / 1000)) >= float(mn) - 1e-9 for i in range(1001))
    report(4, ok, f"C1={C1} C2={C2} min={mn}")


def p05():
    t, X = sp.symbols("t X", positive=True)
    y = 10 - X**2
    tang = sp.diff(y, X).subs(X, t) * (X - t) + y.subs(X, t)
    f = sp.solve(tang, X)[0]
    g = tang.subs(X, 0)
    h = sp.expand(sp.simplify(t * f * g))
    val = sp.diff(h, t).subs(t, 1)
    ok = sp.simplify(val - sp.sympify(md_answer(5, "1"))) == 0
    report(5, ok, f"f={f} g={g} h={h} h'(1)={val}")


def p06():
    k = sp.symbols("k", positive=True)
    X = sp.symbols("X")
    found = []
    for kv in sp.solve(sp.Eq((sp.log(9 * k, 3))**2 - 2 * sp.log(k**2, 3), 13), k):
        kv = sp.nsimplify(sp.simplify(kv))
        roots = sp.solve(X**2 - X * sp.log(9 * kv, 3) + sp.log(kv**2, 3), X)
        roots = [sp.simplify(r) for r in roots]
        if all(r.is_real and r > 0 for r in roots):
            found.append(kv)
    ok = len(found) == 1 and sp.simplify(found[0] - sp.sympify(md_answer(6, "1"))) == 0
    report(6, ok, f"k candidates with positive roots={found}")


def p07():
    t = sp.symbols("t", real=True)
    c = sp.symbols("c0:5")
    # 차수 4 이하의 일반 다항식으로 두고 항등식을 풀어 f 결정(차수는 계수 비교로 자동 결정)
    f = sum(c[i] * x**i for i in range(5))
    ft = f.subs(x, t)
    a_ = sp.integrate(ft, (t, -1, 1))
    b_ = sp.integrate(t * ft, (t, -1, 1))
    eq = sp.expand(f + sp.integrate(ft, (t, 0, x)) - (x**3 + 2 * x**2 + x * a_ + 3 * b_))
    sol = sp.solve(sp.Poly(eq, x).all_coeffs(), c, dict=True)
    fs = f.subs(sol[0])
    val = sp.integrate(fs, (x, 1, 2))
    ok = len(sol) == 1 and sp.simplify(val - sp.sympify(md_answer(7, "1"))) == 0
    report(7, ok, f"f={sp.expand(fs)} integral={val}")


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
