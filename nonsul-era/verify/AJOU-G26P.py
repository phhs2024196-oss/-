#!/usr/bin/env python3
"""AJOU-G26P (아주대 2026학년도 논술고사 자연계열(약학과)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "AJOU-G26P"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def md_pairs(no: int, sub: str) -> dict:
    """'k=3; limit=1/2' 또는 'radius: ..; circle: ..' 꼴을 사전으로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = re.split(r"[:=]", seg, maxsplit=1)
        out[k.strip()] = v.strip()
    return out


def md_set(no: int, sub: str) -> list:
    return [sp.sympify(s) for s in md_answer(no, sub).strip().strip("{}").split(",")]


def same(a, b, tol=1e-10) -> bool:
    return abs(complex(sp.N(a, 30)) - complex(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


X, Y = sp.symbols("X Y", real=True)


def _normal(av):
    """곡선 y=x²/2 위 (av, av²/2)에서의 법선(접선 기울기 av 에 수직)."""
    return -(x - av) / av + av**2 / 2


def _P(av, bv):
    px = sp.solve(sp.Eq(_normal(av), _normal(bv)), x)[0]
    return sp.simplify(px), sp.simplify(_normal(av).subs(x, px))


def p01():
    a, b, q = sp.symbols("a b q", positive=True)
    # (1) a=3, b>3 에서 P_y 의 하한
    _, py = _P(3, b)
    inc = sp.simplify(sp.diff(py, b))
    ok1 = all(float(inc.subs(b, v)) > 0 for v in (3.001, 4, 10, 100)) and \
        same(sp.limit(py, b, 3, "+"), sp.sympify(md_answer(1, "1-1")))
    # (2) H, 거리 최대
    L1 = _normal(a)
    H = [r for r in sp.solve(sp.Eq(x**2 / 2, L1), x) if sp.simplify(r - a) != 0][0]
    d = (L1.subs(x, q) - q**2 / 2) / sp.sqrt(1 + 1 / a**2)      # H<q<a 에서 곡선이 직선 아래
    qs = [r for r in sp.solve(sp.diff(d, q), q)]
    qq = sp.symbols("qq", real=True)
    d2 = d.subs(q, qq)
    crit = sp.solve(sp.diff(d2, qq), qq)
    dmax = sp.simplify(d2.subs(qq, crit[0]))
    md2 = sp.sympify(md_answer(1, "1-2"), locals={"a": a})
    ok2 = len(crit) == 1 and all(abs(float((dmax - md2).subs(a, v))) < 1e-12 for v in (0.3, 1, 2.7)) \
        and all(float(H.subs(a, v)) < float(crit[0].subs(a, v)) < v for v in (0.3, 1, 2.7))
    # (3) P 가 포물선 위 ⇔ 인수
    px, py = _P(a, b)
    fac = sp.factor(sp.simplify(py - px**2 / 2))
    d3 = md_pairs(1, "1-3")
    rel = d3["relation"].split("=")
    relx = sp.sympify(rel[0], locals={"a": a, "b": b}) - sp.sympify(rel[1], locals={"a": a, "b": b})
    rest = sp.simplify(fac / relx)
    ok_rel = sp.simplify(rest.subs({a: 1, b: 3})) != 0 and all(
        float(rest.subs({a: u, b: v})) != 0 and (float(rest.subs({a: u, b: v})) > 0) == (float(rest.subs({a: 0.5, b: 1})) > 0)
        for u, v in ((0.1, 0.2), (1, 5), (2, 3)))                   # 남은 인수는 a,b>0 에서 부호 일정(0 아님)
    up = sp.sympify(re.search(r"<a<(.+)$", d3["range"].replace(" ", "")).group(1))
    acrit = sp.solve(sp.Eq(a, sp.solve(relx, b)[0]), a)            # a<b=2/a 의 경계
    ok3 = ok_rel and len(acrit) == 1 and same(acrit[0], up)
    # (2-1) a=1 → b=2, 외접원
    bv = sp.solve(relx.subs(a, 1), b)[0]
    P1 = sp.Matrix(_P(1, bv)); A1 = sp.Matrix([1, sp.Rational(1, 2)]); B1 = sp.Matrix([bv, bv**2 / 2])
    cx, cy = sp.symbols("cx cy")
    C = sp.Matrix([cx, cy])
    sc = sp.solve([(A1 - C).dot(A1 - C) - (B1 - C).dot(B1 - C), (A1 - C).dot(A1 - C) - (P1 - C).dot(P1 - C)], [cx, cy], dict=True)[0]
    r2 = ((B1 - C).dot(B1 - C)).subs(sc)
    d4 = md_pairs(1, "2-1")
    circ = sp.sympify(d4["circle"], locals={"x": X, "y": Y})
    ok4 = same(sp.sqrt(r2), sp.sympify(d4["radius"])) and \
        sp.expand(circ - ((X - sc[cx])**2 + (Y - sc[cy])**2 - r2)) == 0
    # (2-2) S1, S2 (b=2/a), a→0+
    bb = 2 / a
    px, py = _P(a, bb)
    Rx = px / 2                                                  # P 에서의 접선의 x절편
    tanP = px * (x - px) + px**2 / 2
    S1 = sp.integrate(x**2 / 2 - tanP, (x, px, Rx)) + sp.integrate(x**2 / 2, (x, Rx, 0))
    S2 = sp.integrate(_normal(bb) - _normal(a), (x, px, a)) + sp.integrate(_normal(bb) - x**2 / 2, (x, a, bb))
    lim = sp.limit(sp.simplify(S2 / S1), a, 0, "+")
    ok5 = same(lim, sp.sympify(md_answer(1, "2-2")))
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"γ={sp.limit(_P(3, b)[1], b, 3, '+')} dmax={dmax} factor={fac} "
           f"a<{acrit} center=({sc[cx]},{sc[cy]}) r²={r2} lim={lim} [{ok1},{ok2},{ok3},{ok4},{ok5}]")


def _g(Xv, m):
    Xv = sp.nsimplify(Xv)
    k = sp.floor((m * Xv + 1) / 4)
    u = m * Xv - 4 * k
    return u if u < 1 else -u + 2


def _roots(a, b, m, c, d, K=8):
    """a x² + b g(x) = c x + d 의 서로 다른 실근(조각별 이차방정식, 정확 계산)."""
    a, b, m, c, d = map(sp.nsimplify, (a, b, m, c, d))
    out = set()
    for k in range(-K, K + 1):
        for lo, hi, gx in ((sp.Rational(4 * k - 1) / m, sp.Rational(4 * k + 1) / m, m * x - 4 * k),
                           (sp.Rational(4 * k + 1) / m, sp.Rational(4 * k + 3) / m, -(m * x - 4 * k) + 2)):
            for r in sp.solve(sp.expand(a * x**2 + b * gx - c * x - d), x):
                if r.is_real and bool(r >= lo) and bool(r < hi):
                    out.add(sp.nsimplify(r))
    return out


def p02():
    # (1-1) a=2,b=16,m=1: 정수 q 별 교점 수
    F = lambda Xv: 2 * sp.nsimplify(Xv)**2 + 16 * _g(Xv, 1)
    cnts = {q: len(_roots(2, 16, 1, 0, q)) for q in range(-30, 120)}
    mx = max(cnts.values())
    ok1 = sum(q for q, v in cnts.items() if v == mx) == int(sp.sympify(md_answer(2, "1-1")))
    # (1-2)(1-3) a_n
    an = [len(_roots(2, 16, 1, 0, F(nv - 6))) for nv in range(1, 31)]
    ok2 = sum(an[:5]) == int(sp.sympify(md_answer(2, "1-2")))
    d3 = md_pairs(2, "1-3")
    r = int(d3["r"])
    tail_const = all(v == r for v in an[10:])                  # n≥11 에서 a_n=r (n≤30 확인; n≥9 이후 f(n-6) 단조 증가 구간)
    ok3 = tail_const and sum(v - r for v in an) == int(d3["sum"])
    # (2) b=1,m=3: 모든 p>0 에서 교점 2개 → 최소 a. 후보 a 와 그보다 조금 작은 a 비교
    amd = sp.sympify(md_answer(2, "2"))
    def ok_a(av):
        ps = [sp.Rational(i, 6) for i in range(1, 120)]
        for Xv in [sp.Rational(j, 3) for j in range(-12, 13)] + [-sp.Rational(3, 2) / av, sp.Rational(3, 2) / av]:
            v = av * Xv**2 + _g(Xv, 3)
            if v > 0:
                ps.append(v)
        return all(len(_roots(av, 1, 3, 0, p_, K=5)) == 2 for p_ in ps)
    ok4 = ok_a(amd) and ok_a(amd + sp.Rational(1, 2)) and not ok_a(amd - sp.Rational(1, 50))
    # (3) a=b=1,m=2: 교점 개수 최소인 c 의 집합
    md_set_ = sp.sympify(md_answer(2, "3"))
    import numpy as np
    cs = [sp.nsimplify(v) for v in np.round(np.linspace(-4, 4, 161), 3)] + \
        [sp.Rational(-13, 6), -2, 2 * sp.sqrt(2) - 2, sp.Rational(5, 2)]
    counts = {c: len(_roots(1, 1, 2, c, 0, K=6)) for c in cs}
    mn = min(counts.values())
    ok5 = all((counts[c] == mn) == bool(md_set_.contains(c)) for c in cs)
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"max교점={mx} a1..5={an[:5]} a6..12={an[5:12]} r={r} min교점={mn} [{ok1},{ok2},{ok3},{ok4},{ok5}]")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
