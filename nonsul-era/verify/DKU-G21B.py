"""DKU-G21B 검산: 문제 조건에서 독립 재계산한 값을 items/DKU/PAST 의 answer.value 와 비교."""
PREFIX = "DKU-G21B"
WIDTH = 2
import itertools
from fractions import Fraction
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def sym(s: str):
    return sp.sympify(s, locals={"E": sp.E, "pi": sp.pi})


def sym_set(s: str):
    return sorted(float(sp.N(v)) for v in sp.sympify(s.replace("{", "[").replace("}", "]"), locals={"E": sp.E}))


def assign(s: str):
    return {k.strip(): sym(v.strip()) for k, v in (kv.split("=") for kv in s.split(","))}


def close(a, b, tol=1e-9) -> bool:
    return abs(float(sp.N(a, 30)) - float(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


PEND = []


def sub(ok, msg):
    PEND.append((bool(ok), msg))


def flush(seq):
    report(seq, all(o for o, _ in PEND), " | ".join(m for _, m in PEND))
    PEND.clear()


def main(g):
    fns = sorted((nm for nm in g if re.fullmatch(r"p\d+", nm)), key=lambda s: int(s[1:]))
    for nm in fns:
        seq = int(nm[1:])
        try:
            g[nm]()
        except Exception as e:  # noqa: BLE001
            PEND.clear()
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)


def p01():
    P = sp.symbols("P", positive=True)
    pr = {"A": sp.Rational(2, 5), "B": sp.Rational(2, 5), "C": sp.Rational(3, 5), "D": sp.Rational(3, 5), "E": P}
    edges = {"A": ("L", "T"), "B": ("T", "R"), "C": ("L", "M"), "D": ("M", "R"), "E": ("T", "M")}
    names = list(edges)

    def lit(state):
        par = {v: v for v in "LTRM"}

        def find(u):
            while par[u] != u:
                u = par[u]
            return u
        for k, on in zip(names, state):
            if on:
                par[find(edges[k][0])] = find(edges[k][1])
        return find("L") == find("R")
    tot = onE = 0
    for st in itertools.product([0, 1], repeat=5):
        if lit(st):
            w = sp.Integer(1)
            for k, s in zip(names, st):
                w *= pr[k] if s else 1 - pr[k]
            tot += w
            if st[4]:
                onE += w
    tot = sp.expand(tot)
    onE = sp.expand(onE)
    v1 = tot.subs(P, 1)
    sub(sp.Rational(md_answer(1, "1")) == v1, f"논제1 P(p=1)={v1}")
    poly = sp.Poly(tot, P)
    a_, b_ = poly.coeff_monomial(P) * 625, poly.coeff_monomial(1) * 625
    got = {"a": a_, "b": b_}
    exp = assign(md_answer(1, "2"))
    sub(poly.degree() == 1 and all(got[k] == exp[k] for k in "ab"), f"논제2 a={a_}, b={b_}")
    # 논제3: P(E|on)=onE/tot ∈ [1/2, 2/3]
    ratio = sp.simplify(onE / tot)
    sols = sp.solve_univariate_inequality(sp.And(ratio >= sp.Rational(1, 2), ratio <= sp.Rational(2, 3)), P, relational=False) \
        if False else None
    pts = sorted(set(sp.solve(sp.Eq(ratio, sp.Rational(1, 2)), P) + sp.solve(sp.Eq(ratio, sp.Rational(2, 3)), P)))
    pts = [v for v in pts if 0 <= v <= 1]
    # 구간 [min,max] 안이 전부 조건을 만족하는지 격자로 확인
    lo, hi = min(pts), max(pts)
    inside = all(sp.Rational(1, 2) <= ratio.subs(P, lo + (hi - lo) * sp.Rational(i, 20)) <= sp.Rational(2, 3) for i in range(21))
    out = [ratio.subs(P, v) for v in (0, 1)]
    exp3 = assign(md_answer(1, "3"))
    sub(inside and lo == exp3["min"] and hi == exp3["max"], f"논제3 min={lo}, max={hi}, ratio(0,1)={out}")
    flush(1)


def p02():
    # 논제1: f(1)=2, ∫0^1 f =1 을 만족하는 여러 f 에서 A(1) 를 직접 수치 계산
    def area(fn, tv, f_tv):
        # P=(t,f(t)), Q=(f(t),t). 영역 = ∫0^t f + 사다리꼴(PQ 아래, x∈[t,f(t)]) - ∫0^{f(t)} g
        def ginv(y):
            lo, hi = mp.mpf(0), mp.mpf(max(f_tv, 1) * 2)
            for _ in range(70):
                mid = (lo + hi) / 2
                if fn(mid) < y:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        int_f = mp.quad(fn, [0, tv])
        trap = (f_tv + tv) / 2 * (f_tv - tv)
        int_g = mp.quad(ginv, [0, f_tv / 2, f_tv])
        return int_f + trap - int_g, int_g

    vals = []
    for s in (0, 0.5, 1.5, 3):
        fn = lambda z, s=s: 2 * z + s * z * (z - 1) * (z - mp.mpf(1) / 2)
        assert abs(mp.quad(fn, [0, 1]) - 1) < 1e-12 and abs(fn(1) - 2) < 1e-12
        A, ig = area(fn, mp.mpf(1), fn(1))
        # 조건 ∫0^2 x g'(x) dx = 2 g(2) - ∫0^2 g = 1 확인
        cond = 2 * 1 - ig
        vals.append((A, cond))
    ok1 = all(abs(A - sp.Rational(md_answer(2, "1")).evalf(20)) < 1e-9 and abs(c - 1) < 1e-9 for A, c in vals)
    sub(ok1, "논제1 A(1)=" + ", ".join(mp.nstr(A, 10) + "/" + repr(float(c - 1)) for A, c in vals))
    # 논제2: 삼차함수 h
    a_, c_ = sp.symbols("a c")
    hh = a_ * (x**3 - sp.Rational(9, 2) * x**2 + 6 * x) + c_
    cands = []
    for eq in (hh.subs(x, 1), hh.subs(x, 2)):
        s = sp.solve([eq, sp.integrate(hh, (x, 0, 3)) - sp.Rational(3, 2)], [a_, c_], dict=True)[0]
        if s[a_] > 0:
            cands.append(hh.subs(s))
    assert len(cands) == 1
    h = sp.expand(cands[0])
    assert sp.diff(h, x).subs(x, 1) == 0 and sp.diff(h, x).subs(x, 2) == 0
    # A(t) 공식을 구체적 f 로 직접 확인 (f=x+x^2/2+... 임의의 f'>1)
    fn = lambda z: z + z**2 / 2 + mp.mpf(1) / 3 * z**3 + z
    A2, _ = area(fn, mp.mpf(2), fn(2))
    form = 2 * mp.quad(fn, [0, 2]) - 2 * fn(2) + (fn(2) ** 2 - 4) / 2
    fok = abs(A2 - form) < 1e-8
    # A(2)=2∫f-7/2 => f(2)^2-4f(2)+3=0, f(2)>2
    F2 = sp.symbols("F2", real=True)
    sol = [v for v in sp.solve(-2 * F2 + (F2**2 - 4) / 2 + sp.Rational(7, 2), F2) if v > 2]
    assert len(sol) == 1
    g2 = sol[0]  # f(2)
    # g: [0,3]->[0,2] 증가 전사(g(3)=2), 따라서 h(g(x))=1/2 의 해 개수 = [0,2] 에서 h(s)=1/2 의 해 개수
    rts = [r for r in sp.real_roots(sp.Poly(h - sp.Rational(1, 2), x)) if 0 <= r <= 2]
    cnt = len(set(rts))
    sub(fok and int(md_answer(2, "2")) == cnt and g2 == 3, f"논제2 h={h}, f(2)={g2}, count={cnt}, 공식확인={fok}")
    flush(2)


if __name__ == "__main__":
    main(globals())
