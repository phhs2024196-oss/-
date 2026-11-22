#!/usr/bin/env python3
"""가천대 2025학년도 논술 모의고사 A형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-M25A"
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
    A = sp.symbols("A", positive=True)
    y = A ** (x - 1) + 2
    # a>1: 증가 → x=2 최소
    s1 = [v for v in sp.solve(sp.Eq(y.subs(x, 2), sp.Rational(19, 9)), A)]
    ok1 = all(v <= 1 for v in s1) and "존재하지 않" in ans(5, "1")
    s2 = [v for v in sp.solve(sp.Eq(y.subs(x, 3), sp.Rational(19, 9)), A) if 0 < v < 1]
    # 감소 확인 및 최솟값 재확인
    ok2 = len(s2) == 1 and min(y.subs(A, s2[0]).subs(x, v) for v in (2, sp.Rational(5, 2), 3)) == sp.Rational(19, 9) and "존재한다" in ans(5, "2")
    report(5, ok1 and ok2, f"① a={s1}(a>1 모순) ② a={s2}")


def p06():
    th = sp.symbols("th", real=True)
    s_c = sp.symbols("s c", real=True)
    S, C = s_c
    sol = sp.solve([sp.Eq(C + S, sp.Rational(1, 5)), sp.Eq(S ** 2 + C ** 2, 1)], [S, C], dict=True)
    vals = {sp.simplify(d[S] ** 3 + d[C] ** 3) for d in sol}
    chk = sp.simplify(sp.sin(sp.pi / 2 + th) - sp.cos(sp.pi / 2 + th) - (sp.cos(th) + sp.sin(th))) == 0
    report(6, chk and vals == {tex(ans(6))}, f"변환 확인={chk}, 값={vals}")


def p07():
    for n in range(0, 5):
        cs = sp.symbols(f"c0:{n+1}")
        f = sum(cs[i] * x ** i for i in range(n + 1))
        eq = sp.expand(sp.integrate(f.subs(x, t), (t, 0, x)) - (f + a * x ** 2 + x))
        sol = [s for s in sp.solve(sp.Poly(eq, x).coeffs(), list(cs) + [a], dict=True) if n == 0 or s.get(cs[n], cs[n]) != 0]
        if sol:
            break
    s = sol[0]
    fa = f.subs(s).subs(x, s[a])
    report(7, n == 1 and sp.simplify(fa - tex(ans(7))) == 0, f"최저 차수 해 n={n}, f={f.subs(s)}, a={s[a]}, f(a)={fa}")


def p08():
    an = [1 + 3 * (k_ - 1) for k_ in range(1, 200)]
    rhs = sum(v ** 2 for v in an[:4])
    n = next(n for n in range(1, 200) if sum(an[:n]) > rhs)
    report(8, n == int(ans(8)), f"Σa_k²={rhs}, 최소 n={n}")



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
