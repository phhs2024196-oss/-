#!/usr/bin/env python3
"""SJU-M20 (세종대 2020학년도 논술 모의평가 자연계열) — sympy 검산. 정답 하드코딩 금지, 설정으로부터 독립 재계산."""
import re
import sys
from fractions import Fraction as Fr
from itertools import product, combinations

import sympy as sp

PREFIX = "SJU-M20"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(no: int, sub: str) -> dict:
    """'k1 = v1; k2 = v2' 형식의 정답을 dict 로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def S(s: str):
    return sp.sympify(s, locals={"E": sp.E, "e": sp.E, "x": x, "t": t, "M": sp.Symbol("M", positive=True)})


def same(a, b, tol=1e-12) -> bool:
    try:
        return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 or abs(complex(sp.N(a - b, 30))) < tol
    except Exception:  # noqa: BLE001
        return abs(complex(sp.N(a - b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a = sp.Symbol("a", positive=True)
    dist = {1: a / 2, 2: 1 - a, 3: a / 2}
    p = sum(dist[i] * dist[j] for i in dist for j in dist if 2 < sp.Rational(i + j, 2) < 4)
    ok1 = sp.simplify(p - sp.sympify(md_answer(1, "1"), locals={"a": a})) == 0
    EX = sp.simplify(sum(k * v for k, v in dist.items()))
    ET = sp.simplify(sum(sp.Rational(k, 100) * EX for k in range(1, 100)) + EX)
    ok2 = same(ET, S(md_answer(1, "2")))
    ah = sp.Rational(20, 100)
    half = sp.Rational(196, 100) * sp.sqrt(ah * (1 - ah) / 100)
    lo, hi = ah - half, ah + half
    import re as _re
    nums = [sp.nsimplify(v) for v in _re.findall(r"[-\d./]+", md_answer(1, "3"))]
    ok3 = len(nums) == 2 and same(lo, nums[0]) and same(hi, nums[1])
    report(1, ok1 and ok2 and ok3, f"P={sp.factor(p)} E(X)={EX} E(T)={ET} CI=[{lo},{hi}]")


def p02():
    # 좌표로 삼각형을 직접 구성: A 원점, AB 는 x축과 -π/3? → 일반적으로 AB, AC 방향을 두고 내심·접점 계산
    A = sp.Matrix([0, 0])
    AB = sp.Rational(3, 2) * sp.sqrt(3); AC = 4 * sp.sqrt(3)
    B = A + AB * sp.Matrix([sp.cos(-sp.pi / 3), sp.sin(-sp.pi / 3)])
    C = A + AC * sp.Matrix([1, 0])
    a_ = sp.sqrt((B - C).dot(B - C)); b_ = AC; c_ = AB
    I = (a_ * A + b_ * B + c_ * C) / (a_ + b_ + c_)                      # 내심
    def foot(Pt, Q1, Q2):
        d = Q2 - Q1
        return Q1 + d * ((Pt - Q1).dot(d) / d.dot(d))
    P = foot(I, A, B); Q = foot(I, B, C); R = foot(I, A, C)
    r = sp.sqrt(sp.simplify((P - I).dot(P - I)))
    OP, OR = sp.simplify(P - I), sp.simplify(R - I)
    def coeffs(vec):
        s, u = sp.symbols("s u")
        sol = sp.solve(list(s * OP + u * OR - vec), [s, u], dict=True)[0]
        return sp.nsimplify(sp.simplify(sol[s])), sp.nsimplify(sp.simplify(sol[u]))
    def mdco(no, sub):
        rhs = md_answer(no, sub).split("=", 1)[1]
        e = sp.sympify(rhs, locals={"OP": sp.Symbol("OP"), "OR": sp.Symbol("OR")})
        return e.coeff(sp.Symbol("OP")), e.coeff(sp.Symbol("OR"))
    c1 = coeffs(B - A); c2 = coeffs(C - B); c3 = coeffs(Q - I)
    ok = sp.simplify(r - 1) == 0 and c1 == mdco(2, "1") and c2 == mdco(2, "2") and c3 == mdco(2, "3")
    report(2, ok, f"r={sp.simplify(r)} AB={c1} BC={c2} OQ={c3} BC_len={sp.simplify(a_)}")


def p03():
    import mpmath as mp
    c, X, Y = sp.symbols("c X Y", positive=True)
    rel = X**2 * Y**2 + Y**4 - c**2 * X**2                               # (d m)^2 = c^2
    xofy = [e for e in sp.solve(rel, X) if e.subs({Y: sp.Rational(1, 2), c: 2}).is_positive][0]
    dxdy = sp.diff(xofy, Y).subs(Y, 1)
    g = sp.lambdify(c, dxdy - 3, "mpmath")
    roots = set()
    for k in range(1, 400):                                 # c>1 (y=1 이 곡선 위) 구간 스캔
        lo_, hi_ = 1 + k / 100, 1 + (k + 1) / 100
        if mp.sign(g(lo_)) != mp.sign(g(hi_)):
            roots.add(round(float(mp.findroot(g, (lo_, hi_), solver="bisect")), 12))
    cs = [sp.nsimplify(r_, [sp.sqrt(2)]) for r_ in roots]
    ok1 = len(cs) == 1 and same(cs[0], S(md_answer(3, "1")))
    cc = cs[0]
    xy = xofy.subs(c, cc)
    a = xy.subs(Y, 1)
    f_at = lambda xv: mp.findroot(lambda yy: float(1) * (xv**2 * yy**2 + yy**4 - float(cc**2) * xv**2), 0.6)
    y6 = sp.nsimplify(f_at(1 / mp.sqrt(6)))
    ans = parts(3, "2")
    line = sp.sympify(md_answer(3, "2").split(";")[0].split(":")[1].split("=")[0], locals={"x": X, "y": Y})
    # 접선: 점 (a,1), 기울기 1/3
    ok_line = sp.simplify(line.subs({X: a, Y: 1})) == 0 and sp.simplify(-sp.diff(line, X) / sp.diff(line, Y) - sp.Rational(1, 3)) == 0
    px, py = 1 / sp.sqrt(6), sp.sqrt(sp.Rational(1, 2))
    ok_y = abs(float(py) - float(f_at(1 / mp.sqrt(6)))) < 1e-12
    dist = sp.Abs(line.subs({X: px, Y: py})) / sp.sqrt(sp.diff(line, X)**2 + sp.diff(line, Y)**2)
    ok2 = ok_line and ok_y and same(dist, S(ans["distance"]))
    # (3) 수치 적분: f 를 수치로, f' 는 역함수 미분 1/(dx/dy)
    dxdy_f = sp.lambdify(Y, sp.diff(xy, Y), "mpmath")
    def integrand(xv):
        yv = f_at(xv)
        return xv**2 / yv**3 / dxdy_f(yv)
    val = mp.quad(integrand, [1 / mp.sqrt(6), 1])
    ok3 = abs(val - float(S(md_answer(3, "3")))) < 1e-9
    report(3, ok1 and ok2 and ok3, f"c={cc} a={a} dist={sp.N(dist, 12)} int={mp.nstr(val, 14)}")


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
