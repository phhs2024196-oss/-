#!/usr/bin/env python3
"""가천대 2024학년도 논술전형 자연E 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-G24E"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int) -> str:
    """YAML 의 answer.value (LaTeX 문자열)."""
    it = yaml.safe_load((ROOT / "items/GCU/PAST" / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def tex(s: str):
    """간단한 LaTeX(\\dfrac, \\sqrt, 정수) → sympy."""
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    for _ in range(5):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
        s = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", s)
        s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"((\1)/(\2))", s)
        s = re.sub(r"\\frac(\d)\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("\\pi", "pi").replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)\s*(sqrt|pi|\()", r"\1*\2", s)
    return sp.nsimplify(sp.sympify(s))


def same(val, s) -> bool:
    return sp.simplify(sp.nsimplify(val) - tex(s)) == 0


import mpmath as mp


def parts(s):
    return [tex(seg.split("=")[-1]) for seg in s.split(",\\ ")]


def p07():
    mp.mp.dps = 40
    good = [n_ for n_ in range(1, 5000) if (mp.log(mp.mpf(n_) / 8, 2) * mp.log(mp.mpf(n_) / 64, 2)) < 40]
    report(7, same(max(good), ans(7)) and good == list(range(1, max(good) + 1)), f"만족 자연수 1..{max(good)}, 최댓값={max(good)}")


def p08():
    A, r = sp.symbols("A r", real=True)
    vals = set()
    for s in sp.solve([sp.Eq(A * A * r**3, sp.Rational(3, 2)), sp.Eq(A**2 * r**13, 54)], [A, r], dict=True):
        if s[A].is_real and s[r].is_real:
            vals.add(sp.simplify((s[A] * s[r]**4)**2))
    report(8, len(vals) == 1 and same(vals.pop(), ans(8)), "실수 등비수열 해 전부에서 a5^2 동일")


def _roots_abs_sine(kv):
    out = set()
    for c in {kv, -kv}:
        v = (c - 1) / sp.Integer(6)  # sin x = v
        if -1 <= v <= 1:
            for r in (sp.pi - sp.asin(v), 2 * sp.pi + sp.asin(v)):
                if sp.pi <= r < 2 * sp.pi:
                    out.add(sp.simplify(r))
    return sorted(out, key=lambda z: float(z))


def p09():
    good = []
    for kv in [sp.Rational(j, 4) for j in range(1, 30)]:
        rts = _roots_abs_sine(kv)
        if len(rts) == 3:
            good.append((kv, rts))
    kv, (al, be, ga) = good[0]
    val = sp.simplify(kv * (be + ga) / al)
    report(9, len(good) == 1 and same(val, ans(9)), f"세 근이 되는 k={[g[0] for g in good]}, α={al}, β+γ={sp.simplify(be + ga)}, 값={val}")


def p10():
    C = x**4 - 5 * x**3 - 9 * x**2 + 2 * x - 3
    f = sp.diff(C, x)
    crit = [c for c in sp.solve(sp.diff(f, x), x) if c > 0]
    tm = min(crit, key=lambda c: f.subs(x, c))
    assert sp.limit(f, x, sp.oo) == sp.oo and f.subs(x, tm) < f.subs(x, 0)  # t→0+ 보다 작음
    line = sp.expand(f.subs(x, tm) * (x - tm) + C.subs(x, tm))
    got = sp.sympify(ans(10).split("=")[1].replace("x", "*x"), locals={"x": x})
    report(10, sp.simplify(line - got) == 0, f"t={tm}, 최소 기울기={f.subs(x, tm)}, 접점=({tm},{C.subs(x, tm)}), 접선 y={line}")


def p11():
    A = [n_ for n_ in range(-50, 51) if sp.Rational(1, 4)**(-n_**2 + 5 * n_ + 3) < sp.Integer(4)**(n_ - 6)]
    kk = sp.symbols("kk")
    prod_exp = sum(-a_**2 + 2 * a_ + kk for a_ in A)  # 3^{-a^2}(1/3)^{-2a-k}=3^{-a^2+2a+k}
    elems = {sp.simplify(-a_**2 + 2 * a_) for a_ in A}
    kv = sp.solve(sp.Eq(prod_exp, 0), kk)[0]
    report(11, len(elems) == len(A) and same(kv, ans(11)), f"A={A}, 곱의 지수={prod_exp}, k={kv}")


def p12():
    a, b = sp.symbols("a b", real=True)
    f = x * (x - 1) * (x**2 + a * x + b)
    s = sp.solve(sp.Eq(sp.limit(f / (x - 1), x, 1), 4), b)[0]
    fp0 = sp.diff(f, x).subs(x, 0).subs(b, s)
    rng = sp.solve_univariate_inequality(sp.Abs(fp0) <= 4, a, relational=False)
    f2 = sp.expand(f.subs(b, s).subs(x, 2))
    lo, hi = f2.subs(a, rng.inf), f2.subs(a, rng.sup)
    got = parts(ans(12))
    report(12, sorted([lo, hi], reverse=True) == got, f"b=3-a, f'(0)={fp0}, a∈{rng}, f(2)={f2} ∈ [{lo},{hi}]")


def _kmin(g):
    rts = sorted(set(float(r) for r in sp.Poly(g, x).real_roots()))
    km = -sp.oo
    for r in rts:
        if g.subs(x, sp.nsimplify(r) - sp.Rational(1, 10**6)) < 0 or g.subs(x, sp.nsimplify(r) + sp.Rational(1, 10**6)) < 0:
            km = r
    return km


def p13():
    best = []
    for a in range(-12, 13):
        for bb in range(-12, 13):
            f = x**3 + a * x**2 + bb * x  # f(0)=0
            g = sp.expand(f + 2 * sp.diff(f, x))
            if g.subs(x, 0) != 0:
                continue
            if abs(_kmin(g)) < 1e-9 and all(g.subs(x, v) >= 0 for v in (0, 1, 5, 50)):
                best.append(f.subs(x, 2))
    report(13, same(min(best), ans(13)), f"정수 계수 전수(|계수|≤12) 조건 만족 f(2) 후보 최솟값={min(best)}")


def p14():
    a, b = sp.symbols("a b")
    f = x**2 + a * x + b
    g = sp.integrate(sp.diff(f, x).subs(x, t), (t, 0, x)) + (x**2 + x + 1) * f + 1
    s = sp.solve([sp.Eq(g.subs(x, 0), -1), sp.Eq(sp.diff(g, x).subs(x, 0), 0)], [a, b], dict=True)[0]
    gs = g.subs(s)
    ismin = sp.diff(gs, x, 2).subs(x, 0) > 0
    report(14, ismin and same(gs.subs(x, 1), ans(14)), f"f={f.subs(s)}, g''(0)>0 확인, g(1)={gs.subs(x, 1)}")


def p15():
    c4, c3, c2, c1, c0, a = sp.symbols("c4 c3 c2 c1 c0 a", real=True)
    # (가) 우함수 → 홀수 차 계수 0; (나) 유한 극한 → x^4 계수 0, x^2 계수 1, 극한값 b=x 계수
    fgen = c4 * x**4 + c2 * x**2 + c0
    lead = sp.Poly(sp.expand(fgen - x**2), x).all_coeffs()
    assert lead[0] == c4  # c4=0, c2=1 이어야 극한 유한(수동 확인 아닌 계수 비교)
    found = set()
    grid = [sp.Rational(j, 4) for j in range(-8, 9)]
    for av in grid:
        for cv in [sp.Rational(j, 4) for j in range(-16, 17)]:
            f = x**2 + cv
            bv = sp.limit((f - x**2) / (x - av), x, sp.oo)
            for kk in (1, 2, 3):
                if sp.limit(sp.Abs(f / (x - av)**kk), x, av) == 1:
                    found.add((av, bv))
    got = {tuple(tex(v) for v in p.strip("()").split(",")) for p in re.findall(r"\(([^()]*?(?:\([^()]*\))*[^()]*)\)", ans(15).replace("\\left", "").replace("\\right", ""))}
    report(15, found == {(sp.nsimplify(p[0]), sp.nsimplify(p[1])) for p in got}, f"격자 탐색 순서쌍={sorted(found)}")


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
