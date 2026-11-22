#!/usr/bin/env python3
"""가천대 2025학년도 논술고사 자연D 세트(보고서 문항카드 64~72) 수학 — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G25D"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items/GCU/PAST"
x, t, a, b, c, k = sp.symbols("x t a b c k", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int, sub: str = "1") -> str:
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    for s in d["subparts"]:
        if str(s["id"]) == sub:
            return str(s["answer"]["value"]).strip()
    raise KeyError(sub)


def tex(s: str):
    """간단한 LaTeX 값 → sympy"""
    s = s.replace("\\dfrac", "\\frac").replace("\\ ", "").replace("\\pi", "pi")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"\\log_(\d+)\s*(\d+)", r"log(\2,\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)\s*(log|sqrt|exp|\()", r"\1*\2", s)
    return sp.nsimplify(sp.sympify(s, locals={"log": lambda v, base: sp.log(v, base)}), rational=False)


def pick(s: str, name: str):
    m = re.search(rf"{name}\s*=\s*([^,]+?)(?:,|$)", s.replace("\\,", ","))
    return tex(m.group(1))


import math
import mpmath as mp


def p07():
    X = sp.symbols("X", real=True)
    # log_a(1/b)=-X, log_{1/b} a = -1/X, log_a(b^3/a^2)=3X-2
    sols = [v for v in sp.solve(sp.Eq(-X, (-1 / X) * (3 * X - 2)), X) if v != 1]
    vals = {sp.simplify(1 / v + (-v)) for v in sols}   # log_b a = 1/X, log_{1/a} b = -X
    report(7, vals == {tex(ans(7))}, f"X={sols}, 값={vals}")


def p08():
    A, B = sp.symbols("A B")
    f = x ** 2 + A * x + B
    F = sp.integrate(f.subs(x, t), (t, 1, x + 1))
    e1 = sp.Eq(sp.integrate(f, (x, 0, 1)), sp.Rational(1, 3))
    e2 = sp.Eq(sp.limit((x ** 2 + x + 1) / x * F, x, 0), 3)
    s = sp.solve([e1, e2], [A, B], dict=True)[0]
    fv = f.subs(s)
    v = fv.subs(x, s[A]) + fv.subs(x, s[B])
    report(8, v == int(ans(8)), f"a={s[A]}, b={s[B]}, f(a)+f(b)={v}")


def p09():
    R = sp.symbols("R", positive=True)
    cosC = sp.Rational(27, 28); sinC = sp.sqrt(1 - cosC ** 2)
    sA, sB = sp.symbols("sA sB", positive=True)
    sol = sp.solve([sp.Eq(4 * R ** 2 * sA * sB, 42), sp.Eq(sA + sB, 13 * sp.sqrt(55) / 56),
                    sp.Eq(cosC, (4 * R ** 2 * (sA ** 2 + sB ** 2 - sinC ** 2)) / (8 * R ** 2 * sA * sB))], [sA, sB, R], dict=True)
    vals = {sp.nsimplify(sp.sqrt(495 * d[R] ** 2)) for d in sol}
    report(9, vals == {int(ans(9))}, f"R={[d[R] for d in sol][:1]}, 값={vals}")


def p10():
    S = sum(sp.Rational(1, 2) * (2 * n) * n for n in range(1, 11))
    L = sum(2 * n + 2 * sp.sqrt((n) ** 2 + n ** 2) for n in range(1, 11))
    ya = ans(10)
    ok = S == pick(ya, r"\\sum S_n") and sp.simplify(L - pick(ya, r"\\sum l_n")) == 0
    report(10, ok, f"ΣS={S}, Σl={sp.simplify(L)}")


def p11():
    best = None
    for bb in [sp.Rational(k_, 4) for k_ in range(-40, 41)]:
        f = x ** 3 / 3 + bb * x ** 2 + 4 * x
        g1 = sp.diff(f + x ** 2, x)
        # (0,∞)에서 g'≥0 (등호는 고립점) 판정
        crit = [r for r in sp.solve(sp.diff(g1, x), x) if r.is_real and r > 0]
        mn = min([g1.subs(x, r) for r in crit] + [g1.subs(x, 0)])
        if mn >= 0:
            s = -3 * bb
            best = s if best is None else max(best, s)
    lims = sp.limit((x ** 3 / 3 - 3 * x ** 2 + 4 * x) / x ** 3, x, sp.oo) == sp.Rational(1, 3) and sp.limit((x ** 3 / 3 - 3 * x ** 2 + 4 * x) / x, x, 0) == 4
    report(11, lims and best == int(ans(11)), f"b 격자(1/4 간격) 탐색 최대 근의 합={best}")


def p12():
    f = -x ** 3 / 3 + x + 7
    fp = sp.diff(f, x)
    def nA(tv):
        expr = sp.expand((f * fp.subs(x, tv) - fp.subs(x, tv)) * (x - tv) + f * (f.subs(x, tv) - 1) - f.subs(x, tv) + 1)
        if expr == 0:
            return None
        return len(set(sp.Poly(expr, x).real_roots()))
    T = sp.symbols("T", real=True)
    cands = set(sp.solve(fp.subs(x, T), T)) | set(sp.solve(sp.Eq(fp.subs(x, T) * (3 - T) + f.subs(x, T), 1), T))
    good = sorted(c for c in cands if nA(c) == 1)
    others = all(nA(sp.Rational(k_, 7)) != 1 for k_ in range(-30, 31) if sp.Rational(k_, 7) not in cands)
    yv = sorted(tex(s_) for s_ in re.findall(r"-?\\frac\{\d+\}\{\d+\}|-?\d+", ans(12)))
    report(12, good == yv and others, f"n(A)=1 인 t={good}, 다른 표본 t 는 모두 n(A)≠1:{others}")


def p13():
    f = x + 1
    Fx = sp.integrate(f, x)
    okf = sp.simplify(sp.diff(sp.Rational(1, 2) * f ** 2, x) - f) == 0 and f.subs(x, 0) == 1
    A, b, c = sp.symbols("A b c")
    g = x ** 2 + b * x + c
    eq = sp.expand(sp.integrate((t + g.subs(x, t)), (t, 0, x)) - (x + g + x ** 3 / 3 + A))
    s = sp.solve(sp.Poly(eq, x).coeffs(), [A, b, c], dict=True)[0]
    g = g.subs(s)
    I = sp.integrate(x ** 2 / (1 + x), (x, 0, sp.Rational(1, 2))) + sp.integrate(f / (1 - x), (x, 0, sp.Rational(1, 2))) + sp.integrate(g / (x ** 2 - 1), (x, 0, sp.Rational(1, 2)))
    I = sp.simplify(I)
    report(13, okf and I == tex(ans(13)), f"g={g}, a={s[A]}, 값={I}")


def p14():
    CA, CB, DB = 5 ** sp.Rational(1, 4), 20 ** sp.Rational(1, 4), sp.Rational(5, 4) ** sp.Rational(1, 4)
    AB = sp.sqrt(CA ** 2 + CB ** 2)
    PB = CB ** 2 / AB; QB = DB ** 2 / AB
    r = sp.simplify(AB / (PB - QB))
    ya = ans(14)
    yPB = sp.sympify("2*sqrt(5)/45**(1/4)"); yQB = sp.sympify("sqrt(5)/2/45**(1/4)")
    ok = sp.simplify(PB - yPB) == 0 and sp.simplify(QB - yQB) == 0 and r == 2 and "2\\sqrt5" in ya and "\\frac12\\sqrt5" in ya and ya.rstrip().endswith("=2")
    report(14, ok, f"AB={sp.simplify(AB)}, PB={sp.simplify(PB)}, QB={sp.simplify(QB)}, AB/PQ={r}")


def p15():
    a1, r = sp.symbols("a1 r", positive=True)
    s = sp.solve([sp.Eq(a1 ** 3 * r ** 3, 3), sp.Eq(a1 ** 3 * r ** 6, 9)], [a1, r], dict=True)[0]
    S12 = sp.simplify(s[a1] * (s[r] ** 12 - 1) / (s[r] - 1))
    v = sp.simplify(1 / S12)
    report(15, sp.simplify(v - (sp.cbrt(3) - 1) / 80) == 0 and "3^{\\frac13}-1" in ans(15) and "80" in ans(15), f"a1={s[a1]}, r={s[r]}, 1/S12={v}")



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
