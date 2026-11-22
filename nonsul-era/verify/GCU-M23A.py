#!/usr/bin/env python3
"""가천대 2023학년도 논술 모의고사 A형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산하고 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-M23A"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "GCU" / "PAST"
x, t, k, a, b, c, n, th = sp.symbols("x t k a b c n theta", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def yaml_answer(seq: int) -> str:
    it = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def tex(s: str):
    """간단한 LaTeX 수(분수·근호) → sympy."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\ ", "")
    for _ in range(5):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"(\d)sqrt", r"\1*sqrt", s)
    s = re.sub(r"\)\(", ")*(", s)
    s = re.sub(r"\)sqrt", ")*sqrt", s)
    return sp.nsimplify(sp.sympify(s))


def same(v, s) -> bool:
    return sp.simplify(sp.nsimplify(v) - tex(s)) == 0


def nums(s: str):
    """YAML 집합형 답에서 수(분수·근호 포함)를 순서대로 추출."""
    parts = re.split(r"[①②③④]", s.replace("\\ ", " "))
    return [p.strip().strip(",").strip() for p in parts if p.strip()]


def cubic_level_k(f, lo, hi, minval):
    """닫힌구간 [lo,hi] 최솟값 = minval 로 상수 a 결정 후, y=k 와 교점 2개인 k 전체."""
    crit = [r for r in sp.solve(sp.diff(f, x), x) if r.is_real]
    cands = [f.subs(x, p) for p in crit if lo <= p <= hi] + [f.subs(x, lo), f.subs(x, hi)]
    at0 = [sp.simplify(e.subs(a, 0)) for e in cands]
    amin = cands[at0.index(min(at0))]
    aval = sp.solve(sp.Eq(amin, minval), a)[0]
    g = f.subs(a, aval)
    ks = []
    for kk in sorted({sp.nsimplify(g.subs(x, p)) for p in crit} | {sp.Integer(0)}):
        roots = sp.Poly(g - kk, x).real_roots()
        if len(set(roots)) == 2:
            ks.append(kk)
    return aval, ks


def p05():
    cs = sp.solve(sp.Eq(6*x - 1/x, -1), x)
    vals = set()
    for cv in cs:
        tv = 2*sp.pi - sp.acos(cv)                       # cos 값이 cv 인 (π, 2π) 의 각
        if sp.Rational(3, 2)*sp.pi < tv < 2*sp.pi:
            vals.add(sp.simplify(sp.sin(tv)*sp.cos(tv)))
    v = list(vals)
    report(5, len(v) == 1 and same(v[0], yaml_answer(5)), f"cos 후보={cs}, sinθcosθ={v}")


def p06():
    aval, ks = cubic_level_k(-2*x**3 + 3*x**2 + 12*x + a, -1, 5, -105)
    report(6, same(sum(ks), yaml_answer(6)), f"a={aval}, k={ks}, 합={sum(ks)}")


def p07():
    rhs = x**3 + x**2 - 6*x
    av = [r for r in sp.solve(rhs, x) if r > 0]
    f = sp.diff(rhs, x)
    v = [f.subs(x, r) for r in av]
    report(7, len(v) == 1 and same(v[0], yaml_answer(7)), f"a={av}, f(x)={f}, f(a)={v}")


def p08():
    S = [sp.Integer(-3)]
    for _ in range(12):
        Sn = sp.symbols("Sn")
        r = [s for s in sp.solve(sp.Eq(3*(Sn + S[-1]), -(Sn - S[-1])**2), Sn) if s - S[-1] < 0]
        assert len(r) == 1
        S.append(r[0])
    an = [S[0]] + [S[i] - S[i-1] for i in range(1, len(S))]
    w = nums(yaml_answer(8))
    nn = sp.Symbol("n")
    ok = (same(1, w[0]) and same(an[1], w[1]) and all(an[i+1] - an[i] == tex(w[2]) for i in range(1, 10))
          and all(an[i-1] == sp.sympify(w[3].replace("n", "*n").lstrip("*").replace("-*", "-")).subs(nn, i) for i in range(1, 12)))
    report(8, ok, f"a_1..a_6={an[:6]}")


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
