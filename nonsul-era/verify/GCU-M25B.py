#!/usr/bin/env python3
"""가천대 2025학년도 논술 모의고사 B형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-M25B"
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
    s = s.replace("\\dfrac", "\\frac").replace("\\ ", "")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"\\log_(\d+)\s*(\d+)", r"log(\2,\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)\s*(log|sqrt)", r"\1*\2", s)
    return sp.nsimplify(sp.sympify(s, locals={"log": lambda v, base: sp.log(v, base)}), rational=False)


def pick(s: str, name: str):
    m = re.search(rf"{name}\s*=\s*([^,]+?)(?:,|$)", s.replace("\\,", ","))
    return tex(m.group(1))


def p05():
    X = sp.symbols("X", real=True)
    # log_a b = X, log_b a = 1/X, log_a(a^3/b) = 3 - X ;  X:(1/X) = (3-X):2
    sols = [v for v in sp.solve(sp.Eq(2 * X, (3 - X) / X), X) if v != 1]  # a≠b ⇒ X≠1
    vals = {sp.simplify(v - 1 / v) for v in sols}
    report(5, vals == {tex(ans(5))}, f"log_a b 후보={sols}, 값={vals}")


def p06():
    f = sp.together(sum(1 / ((x + k_) * (x + k_ + 1)) for k_ in range(1, 6)))
    A, B, C = sp.symbols("A B C")
    g = A * x ** 2 + B * x + C
    sA = sp.solve(sp.Eq(sp.limit(f * g, x, sp.oo), 10), A)[0]
    sol = sp.solve([sp.Eq(-B / sA, -2), sp.Eq(C / sA, -1)], [B, C])
    g = g.subs(A, sA).subs(sol)
    xm = sp.solve(sp.diff(g, x), x)[0]
    report(6, sp.simplify(g.subs(x, xm) - tex(ans(6))) == 0, f"f={sp.factor(f)}, g={g}, 최솟값={g.subs(x, xm)}")


def p07():
    A, B = sp.symbols("A B")
    c2 = sp.symbols("c2")
    f = c2 * x ** 2 + A * x + B
    c2v = sp.solve(sp.Eq(sp.limit(f / (2 * x ** 2 + 3), x, sp.oo), 2), c2)[0]
    f = f.subs(c2, c2v)
    gl = sp.expand(f.subs(x, x - 1) - f)
    gr = sp.expand(f.subs(x, x - 1))
    sol = sp.solve([sp.Eq(gl.subs(x, 1), gr.subs(x, 1)), sp.Eq(sp.diff(gl, x).subs(x, 1), sp.diff(gr, x).subs(x, 1))], [A, B])
    v = f.subs(sol).subs(x, 3)
    report(7, v == int(ans(7)), f"f={f.subs(sol)}, f(3)={v}")


def p08():
    T = sp.symbols("T", real=True)
    def S(tv):
        # 삼각형 PQR: x∈[0,1], x+t ≤ y ≤ 1+t ; 정사각형과의 공통부분: y∈[max(x+t,0), min(1+t,1)]
        lo = sp.Max(x + tv, 0); hi = sp.Min(1 + tv, 1)
        return sp.integrate(sp.Max(hi - lo, 0), (x, 0, 1))
    g = S(sp.Integer(0)) == sp.Rational(1, 2)
    samples = [sp.Rational(-9, 10), sp.Rational(-1, 2), sp.Rational(-1, 7)]
    sums = {sp.nsimplify(S(al) + S(1 + al)) for al in samples}
    n_ = sums == {sp.Rational(2, 3)}
    picked = [lab for lab, ok in zip("ㄱㄴ", (g, n_)) if ok]
    report(8, picked == re.findall(r"[ㄱㄴ]", ans(8)), f"S(0)={S(sp.Integer(0))}, S(α)+S(1+α)={sums}, 참={picked}")



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
