#!/usr/bin/env python3
"""가천대 2024학년도 논술전형 자연C 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-G24C"
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


def parts(s):
    """'p=\\dfrac29,\\ q=1,\\ ...=3' → 각 '=' 오른쪽 값 목록."""
    return [tex(seg.split("=")[-1]) for seg in s.split(",\\ ")]


def p07():
    th = sp.symbols("th", real=True)
    sols = sp.solve(sp.Eq(sp.sin(th) + sp.cos(th), sp.Rational(1, 2)), th)
    vals = {sp.nsimplify(sp.simplify(sp.Abs(sp.sin(s) - sp.cos(s))), [sp.sqrt(7)]) for s in sols}
    ok = len(sols) > 0 and all(abs(float(u - tex(ans(7)))) < 1e-12 for u in vals)
    report(7, ok, f"해 {len(sols)}개 모두 |sin-cos|={sorted(map(float, vals))}, 답={ans(7)}")


def p08():
    a, b = sp.symbols("a b")
    g = (x**3 + a * x + b) / (x + 1)
    s = sp.solve([sp.Eq(g.subs(x, 2), 1), sp.Eq((x**3 + a * x + b).subs(x, -1), 0)], [a, b], dict=True)[0]
    fm1 = sp.limit(g.subs(s), x, -1)
    report(8, same(fm1, ans(8)), f"(a,b)=({s[a]},{s[b]}), f(-1)={fm1}")


def p09():
    c = sp.symbols("c0:5")
    f = sum(c[i] * x**i for i in range(5))
    G = (x + 1)**2 * f - x**4 - 4 * x**3 - 6 * x**2 - 4 * x
    eqs = sp.Poly(sp.expand(sp.diff(G, x) - 2 * (x + 1) * f), x).all_coeffs() + [G.subs(x, 0) + 2]
    s = sp.solve(eqs, c, dict=True)[0]
    f1 = f.subs(s).subs(x, 1)
    report(9, same(f1, ans(9)), f"f={sp.expand(f.subs(s))}, f(1)={f1}")


def p10():
    ok = True
    for kk in (sp.Integer(1), sp.Rational(1, 2), sp.Integer(2)):
        f = x**4 + sp.Rational(8, 3) * kk * x**3 - 6 * kk**2 * x**2 + 3
        lo, mid, hi = -45 * kk**4 + 3, -sp.Rational(7, 3) * kk**4 + 3, sp.Integer(3)
        tests = sorted({lo - 1, lo, (lo + mid) / 2, mid, (mid + hi) / 2, hi, hi + 1, hi + 5})
        for c in tests:
            n_roots = len(set(sp.real_roots(sp.Poly(f - c, x))))
            pred2 = (c > 3) or (lo < c < mid)
            pred3 = c in (hi, mid)
            ok &= (n_roots == 2) == pred2 and (n_roots == 3) == pred3
    s = ans(10)
    ok &= all(w in s for w in ("a>3", "-45k^4+3<a<-\\dfrac73k^4+3", "b=3", "b=-\\dfrac73k^4+3"))
    report(10, ok, "k=1,1/2,2 에서 y=c 교점 수 전수 대조: 두 점 ⇔ c>3 또는 f(-3k)<c<f(k), 세 점 ⇔ c∈{3, f(k)}")


def p11():
    a, b = sp.symbols("a b")
    f1, f2, f3 = a * x**3 + 8, -4 * x + b, b / 8 * x**2 - 6 * x + 9 * a
    s = sp.solve([f1.subs(x, 0) - f2.subs(x, 0), f2.subs(x, 1) - f3.subs(x, 1)], [a, b], dict=True)[0]
    f1, f2, f3 = (g.subs(s) for g in (f1, f2, f3))
    per = sp.simplify(f1.subs(x, -2) - f3.subs(x, 3))  # 주기 5 연결 연속성
    I1 = sp.integrate(f1, (x, -2, 0)) + sp.integrate(f2, (x, 0, 1)) + sp.integrate(f3, (x, 1, 3))
    I2 = sp.integrate(f1.subs(x, x - 5), (x, 3, 5))  # f(x)=f(x-5)
    tot = I1 + I2
    report(11, same(tot, ans(11)) and per == 0, f"a={s[a]}, b={s[b]}, ∫_-2^3={I1}, ∫_3^5={I2}, 합={tot}")


def p12():
    a = sp.symbols("a", positive=True)
    p_, q_ = sp.symbols("p q")
    # [-1,1]에서 f=p x+q x^3 (기함수), f(1)=2a(x=1 대입), ∫_0^1 f=3
    s = sp.solve([p_ + q_ - 2 * a, p_ / 2 + q_ / 4 - 3], [p_, q_], dict=True)[0]
    base = (p_ * x + q_ * x**3).subs(s)

    def F(lo, hi, shift):  # [lo,hi] ⊂ [-1+2m,1+2m]: f(x)=base(x-2m)+4a m
        return sp.integrate(base.subs(x, x - 2 * shift) + 4 * a * shift, (x, lo, hi))
    total = F(1, 2, 1) + F(2, 3, 1) + F(3, 4, 2)
    sol = sp.solve(sp.Eq(total, 45), a)
    av = sol[0]
    inc = all(sp.diff(base.subs(a, av), x).subs(x, xv) > 0 for xv in (-1, -0.5, 0, 0.5, 1))
    area3 = sp.integrate(base.subs(a, av) + 2 * av, (x, -1, 0))
    report(12, len(sol) == 1 and same(av, ans(12)) and inc and av > sp.Rational(3, 2),
           f"∫_1^4 f={sp.expand(total)}, a={av}, 구체 f=({base.subs(a, av)}) 증가 확인, ③넓이={area3}=2a-3")


def p13():
    lo, hi = 9 * x / 2 - 2, (3 * x - 2)**2
    cand = [r for r in sp.solve(sp.Eq(lo, 4), x) if r > 0 and sp.simplify(hi.subs(x, r) - 4) == 0]
    a = cand[0]
    r = sp.limit(lo / x**3, x, sp.oo), sp.limit(hi / x**3, x, sp.oo)
    val = (a - r[0]) / (2 * r[0] + 3)
    bad = sp.solve_univariate_inequality(lo > hi, x, relational=False)
    report(13, len(cand) == 1 and r[0] == r[1] == 0 and same(val, ans(13)),
           f"a={a}, lim f/x=0, 값={val} [주의: 하한>상한인 구간 {bad} 존재 — 조건 모순 의심]")


def p14():
    X, Y = sp.symbols("X Y", positive=True)
    s = sp.solve([sp.Eq(1 / (9 * X) + 1 / (2 * Y), sp.Rational(1, 2)), sp.Eq(9 * X * sp.log(2), 2 * Y * sp.log(12))], [X, Y], dict=True)
    s = [d for d in s if d[X].is_real][0]
    p = sp.nsimplify(sp.simplify(s[X] / sp.log(24, 2)))
    q = sp.nsimplify(sp.simplify(s[Y] / sp.log(24, 12)))
    v = sp.nsimplify(sp.simplify(9 * s[X] * sp.log(2, 24) + s[Y] * sp.log(12, 24)))
    got = parts(ans(14))
    report(14, [p, q, v] == got, f"x={sp.N(s[X])}, y={sp.N(s[Y])}, p={p}, q={q}, 값={v}")


def p15():
    al = sp.solve(sp.Eq(sp.Rational(1, 2) * 4 * t, sp.Rational(3, 2) * sp.pi), t)[0]
    cands = [th for th in (al, 2 * sp.pi - al) if sp.cos(th) / sp.tan(th) < 0]
    th = cands[0]
    s2, t3 = sp.sin(2 * th), sp.tan(3 * th)
    got = parts(ans(15))
    report(15, len(cands) == 1 and [s2, t3] == got, f"α={al}, θ={th}, sin2θ={s2}, tan3θ={t3}")


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
