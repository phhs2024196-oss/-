#!/usr/bin/env python3
"""SKKU-G21A (성균관대 2021학년도 수시 논술우수전형 자연계 1교시 수학 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SKKU-G21A"

WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str, loc=None) -> dict:
    """'key = expr; key2 = expr2' 형식의 정답 문자열을 dict 로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = seg.split("=", 1)
        out[k.strip()] = sp.sympify(v.strip(), locals=loc or {})
    return out


def eq(a, b) -> bool:
    return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 if not isinstance(a, tuple) else all(eq(u, w) for u, w in zip(a, b))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


x = sp.symbols("x", real=True)


def normal_line(a, p):
    """x^2=4py 위 C(a, a^2/4p) 에서 접선에 수직인 직선 y = m x + c 의 (m, c)."""
    slope_t = sp.diff(x**2 / (4 * p), x).subs(x, a)
    m = -1 / slope_t
    c = a**2 / (4 * p) - m * a
    return sp.simplify(m), sp.simplify(c)


def p01():
    a = sp.symbols("a", real=True, nonzero=True)
    p = sp.symbols("p", real=True, nonzero=True)
    # (i) DE 를 일반 a, p 로
    m, c = normal_line(a, p)
    DE = sp.Abs(sp.simplify(c - a**2 / (4 * p)))
    md1 = sp.sympify(md_answer(1, "1").split("=", 1)[1], locals={"p": p})
    focus_dir = sp.Abs(p - (-p))
    ok1 = all(abs(float((DE - md1).subs({a: av, p: pv}))) < 1e-12 and abs(float((md1 - focus_dir).subs(p, pv))) < 1e-12
              for av in (-3, -0.5, 0.7, 5) for pv in (-2, -0.3, 0.25, 4))
    # (ii) y = n x^2  <=>  p = 1/(4n): a_n 을 법선으로부터 직접
    n, k = sp.symbols("n k", positive=True, integer=True)
    an = sp.simplify(DE.subs(p, 1 / (4 * n)).subs(a, 1))
    an = sp.simplify(sp.Abs(an))
    S = sp.summation((an * an.subs(n, n + 1)).subs(n, k), (k, 1, sp.oo))
    ok2 = sp.simplify(S - sp.sympify(md_answer(1, "2"))) == 0
    # (iii) 법선과 포물선의 교점, x 절편
    ap = sp.symbols("a", positive=True)
    vals = []
    for pv in (sp.Rational(1, 3), 2, -1, sp.Symbol("q", positive=True)):
        mm, cc = normal_line(ap, pv)
        xs = sp.solve(sp.Eq(x**2 / (4 * pv), mm * x + cc), x)
        ys = [sp.simplify(xx**2 / (4 * pv)) for xx in xs]
        hx = sp.solve(sp.Eq(0, mm * x + cc), x)[0]
        vals.append(sp.limit(sp.simplify(ys[0] * ys[1] / (ap * hx)), ap, sp.oo))
    ok3 = all(sp.simplify(v - sp.sympify(md_answer(1, "3"))) == 0 for v in vals)
    report(1, ok1 and ok2 and ok3, f"DE=2|p| ok={ok1} a_n={an} sum={S} lim={vals}")


def p02():
    al = sp.symbols("alpha", real=True)
    beta = (2 * al + 1) / (2 * al + 2)
    F = al - beta
    crit = sp.solve(sp.diff(F, al), al)
    gt = [c for c in crit if c > -1]; lt = [c for c in crit if c < -1]
    mn = sp.simplify(F.subs(al, gt[0])); mx = sp.simplify(F.subs(al, lt[0]))
    # 격자로 최소/최대 확인
    import numpy as np
    Ff = sp.lambdify(al, F)
    g1 = np.linspace(-0.999, 20, 200001); g2 = np.linspace(-40, -1.001, 200001)
    okg = abs(Ff(g1).min() - float(mn)) < 1e-6 and abs(Ff(g2).max() - float(mx)) < 1e-6
    P1 = md_parts(2, "1")
    ok1 = okg and eq(P1["min_gt"], mn) and eq(P1["max_lt"], mx)
    r = sp.symbols("r", positive=True)
    e2 = sp.sympify(md_answer(2, "2"), locals={"r": r})
    e3 = sp.sympify(md_answer(2, "3"), locals={"r": r})
    P4 = md_parts(2, "4", {"r": r})
    ok2 = ok3 = ok4 = True
    for av in (-1.0001, -1.3, -1 - 2**-0.5, -2, -5, -37.5, -1e3):
        bv = float(beta.subs(al, av)); rv = (av**2 + bv**2) ** 0.5
        ok2 &= abs(av - bv - float(e2.subs(r, rv))) < 1e-9 * max(1, rv)
        pq = ((av + 2) ** 2 + (bv - 2) ** 2) ** 0.5
        ok3 &= abs(pq - float(e3.subs(r, rv))) < 1e-9 * max(1, rv)
        ok4 &= abs(2 * av**2 + float(P4["b1"].subs(r, rv)) * av + float(P4["b0"].subs(r, rv))) < 1e-7 * max(1, rv**2)
    report(2, ok1 and ok2 and ok3 and ok4, f"min(a>-1)={mn} max(a<-1)={mx} ii={ok2} iii={ok3} iv={ok4}")


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
