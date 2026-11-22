#!/usr/bin/env python3
"""가천대 2023학년도 논술 모의고사 B형 수학(문제 5~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산하고 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-M23B"
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
    f = sp.Piecewise((x**3 + 2*x**2 + 3*x, x < 1), (2*x - 1, True))
    g = 2*x**2 + a*x
    h = sp.Symbol("h", positive=True)
    L = sp.limit((f*g).subs(x, 1 - h), h, 0); R = sp.limit((f*g).subs(x, 1 + h), h, 0); V = (f*g).subs(x, 1)
    sol = set(sp.solve(L - V, a)) & set(sp.solve(R - V, a)) if sp.solve(R - V, a) else set(sp.solve(L - V, a))
    v = list(sol)
    report(5, len(v) == 1 and same(v[0], yaml_answer(5)), f"좌극한={L}, 우극한={R}, 함숫값={V}, a={v}")


def p06():
    cs = sp.symbols("c0:4")
    f = sum(cs[i]*x**i for i in range(4))                  # 차수 3 이하 가정 후 계수 결정(차수 비교로 충분)
    eqs = sp.Poly(f - (sp.diff(f, x) + 3*x**2 - 4*x), x).coeffs()   # F'=f 를 F=f+x^3-2x^2 미분식에 대입
    sol = sp.solve(eqs, cs, dict=True)[0]
    fx = f.subs(sol)
    roots = [r for r in sp.solve(sp.Eq(fx, 5*x**2), x) if r.is_real]
    s = sp.simplify(sum(roots))
    report(6, same(s, yaml_answer(6)), f"f={fx}, 실근={roots}, 합={s}")


def p07():
    d = sp.symbols("d")
    an = lambda i: a + (i - 1)*d
    sol = sp.solve([an(8) + an(9) + an(10) - 30, an(21) + an(23) - 72], [a, d])
    v = an(30).subs(sol)
    report(7, same(v, yaml_answer(7)), f"a1={sol[a]}, d={sol[d]}, a30={v}")


def p08():
    import mpmath as mp
    lo, hi = [sp.nsimplify(v) for v in re.findall(r"-?\d+(?:/\d+)?", yaml_answer(8))]

    def meets(av):
        h = lambda z: mp.log(z + av) / mp.log(0.5) - (-(3 ** (-z + 2)) + 1)
        zs = [mp.mpf(i) / 2000 for i in range(1, 4000) if i / 2000 + av > 0]
        vals = [h(z) for z in zs]
        return any(vals[i] == 0 or vals[i] * vals[i + 1] < 0 for i in range(len(vals) - 1))
    tests = [-1.5, -1.01, -0.99, -0.5, 0, 0.5, 10, 200, 255, 257, 300, 1000]
    ok = all(meets(mp.mpf(tv)) == (float(lo) < tv < float(hi)) for tv in tests)
    report(8, ok, f"수치 판정(x∈(0,2) 교점 유무)이 {lo}<a<{hi} 와 일치")


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
