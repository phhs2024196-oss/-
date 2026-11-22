#!/usr/bin/env python3
"""가천대 2025학년도 논술고사 자연계열 자연F(보고서 문항카드 82~90, 문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G25F"
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
    th = sp.symbols("th", real=True)
    lhs = sp.expand_trig(sp.sin(sp.Rational(3, 2) * sp.pi + th) - sp.cos(sp.Rational(3, 2) * sp.pi + th))
    sols = sp.solve(sp.Eq(lhs, sp.Rational(-3, 4)), th)
    vals = {sp.nsimplify(sp.N(sp.tan(v) + 1 / sp.tan(v), 40), rational=True) for v in sols}
    report(7, vals == {tex(ans(7))}, f"lhs={lhs}, 값={vals}")


def p08():
    n = sp.symbols("n", integer=True, positive=True)
    v = sum(sp.Rational(3 * i * i + 5 * i + 2, 18 * i * i + 30 * i + 8) for i in range(1, 13))
    report(8, v == tex(ans(8)), f"합={v}")


def p09():
    # a_1 자유 → a_{n+1}=2^{e_n}/a_n 로 생성, 합은 a_1 과 무관해야 함
    def tot(expo, x1):
        s, cur = 0, x1
        for nn in range(1, 101):
            s += sp.log(cur, 2)
            cur = 2 ** expo(nn) / cur
        return sp.nsimplify(sp.simplify(s))
    ea = lambda nn: sp.sin(sp.Rational(nn, 6) * sp.pi)
    eb = lambda nn: sp.cos(sp.Rational(nn, 3) * sp.pi)
    ec = lambda nn: sp.tan(sp.Rational(2 * nn - 1, 4) * sp.pi)
    r1 = [tot(ea, sp.Integer(3)) + tot(eb, sp.Integer(5)) + tot(ec, sp.Integer(7)), tot(ea, sp.Integer(1)) + tot(eb, sp.Integer(1)) + tot(ec, sp.Integer(1))]
    report(9, r1[0] == r1[1] == tex(ans(9)), f"합={r1}")


def p10():
    tot, ks = 0, []
    mm = sp.symbols("mm", positive=True)
    for kk in range(2, 16):
        # m>0 에서 교점: x=1, (1,k) 구간 -(x-1)(x-k)=m(x-1) → x=k-m, x>k 구간 (x-1)(x-k)=m(x-1) → x=k+m
        for mv in sp.solve(sp.Eq((kk - mm) ** 2, 1 * (kk + mm)), mm):
            if mv.is_real and 0 < mv < kk - 1:
                xs = sorted([sp.Integer(1), kk - mv, kk + mv])
                r = sp.simplify(xs[1] / xs[0])
                assert sp.simplify(xs[2] - xs[1] * r) == 0
                if sp.nsimplify(r).is_rational:
                    ks.append(kk); tot += kk
    report(10, tot == tex(ans(10)), f"k={ks}, 합={tot}")


def p11():
    X = sp.symbols("X", real=True)
    eq = sp.Eq(sp.log(5 ** X + 10, 5), X / 2 + sp.log(sp.Rational(10, 25), 5) + 2)
    u = sp.symbols("u", positive=True)    # u=5^(x/2)
    sols_u = [s0 for s0 in sp.solve(sp.Eq((u ** 2 + 10), 10 * u), u)]
    rts = [sp.log(s0 ** 2, 5) for s0 in sols_u]
    assert all(abs(sp.N(eq.lhs.subs(X, r0) - eq.rhs.subs(X, r0))) < 1e-20 for r0 in rts) and len(rts) == 2
    m = sp.nsimplify(sp.simplify(sum(5 ** (-r0) for r0 in rts)))
    A = sp.symbols("A")
    y = lambda xv: -sp.log(-xv, 5) + A
    ends = [-sp.root(5, 5), -sp.Rational(1, 125)]
    xx = sp.symbols("xx")
    assert sp.diff(-sp.log(-xx, 5), xx).subs(xx, -1) > 0   # 증가함수
    av = sp.solve(sp.Eq(sp.simplify(y(ends[0])), m), A)[0]
    M = sp.simplify(y(ends[1]).subs(A, av))
    a_ = ans(11)
    report(11, av == pick(a_, "a") and M == pick(a_, "M") and m == pick(a_, "m"), f"m={m}, a={av}, M={M}")


def p12():
    kk = sp.symbols("kk", positive=True)
    f = (kk - x) ** 3 + 6 * (kk - x) ** 2 + 3 * (kk - x) - 18
    ok = True
    # 수치: 여러 k에 대해 f가 (0,k)에서 부호 변화(=g 극값) 여부와 답 조건 비교
    thr = sp.N(tex(re.sub(r"\\sqrt\{(\d+)\}", r"\\sqrt\1", ans(12).split(">")[1])))
    for kv in [sp.Rational(i, 20) for i in range(1, 120)]:
        fk = sp.expand(f.subs(kk, kv))
        roots = [r0 for r0 in sp.Poly(fk, x).real_roots() if 0 < r0 < kv]
        has = any(fk.subs(x, r0 - sp.Rational(1, 10**6)) * fk.subs(x, r0 + sp.Rational(1, 10**6)) < 0 for r0 in roots)
        ok &= (has == (kv > thr))
    sol = sp.solve_univariate_inequality(f.subs(x, 0) > 0, kk, relational=False)
    lo = (sol & sp.Interval.open(0, sp.oo)).inf
    report(12, ok and abs(sp.N(lo - thr, 30)) < 1e-12, f"ok={ok}, f(0)>0 해={sol}, 경계≈{thr}")


def p13():
    B, C, A = sp.symbols("B C A", real=True)
    f = x ** 2 + B * x + C
    rts = sp.solve(sp.Eq(f, f.subs(x, 2)), x)            # x=2, x=-B-2
    bv = sp.solve(sp.Eq(sum(rts), 6), B)[0]
    out = []
    for cv in sp.solve(sp.Eq(sp.Abs(f.subs({B: bv, x: 5})), 6), C):
        fv = f.subs({B: bv, C: cv})
        vert = sp.solve(sp.diff(fv, x), x)[0]
        if fv.subs(x, vert) >= 0:
            continue
        F = lambda lo, hi: sp.integrate(sp.Abs(fv).rewrite(sp.Piecewise), (t, lo, hi)) if False else mp.quad(lambda s: abs(float(fv.subs(x, s))), [lo, hi])
        fl = sp.lambdify(x, fv)
        I = lambda lo, hi: mp.quad(lambda s: abs(fl(s)), [lo, 3 - mp.sqrt(-float(fv.subs(x, vert))), 3, 3 + mp.sqrt(-float(fv.subs(x, vert))), hi]) if lo <= hi else -mp.quad(lambda s: abs(fl(s)), [hi, lo])
        G = lambda av: I(av, 5) - 2 * I(av, 3)
        avs = sorted({round(float(mp.findroot(G, s0)), 10) for s0 in (-3, 0, 1.5, 2.5)})
        for av in avs:
            out.append(sp.nsimplify(av) * fv.subs(x, 3))
    report(13, len(set(out)) == 1 and out[0] == tex(ans(13)), f"b={bv}, af(3)={out}")


def p14():
    m = sp.symbols("m", positive=True)
    tt = sp.symbols("tt")
    sol = sp.solve(sp.Eq((sp.Matrix([4 - tt, 4 - m * tt])).dot(sp.Matrix([1, m])), 0), tt)[0]   # AB ⊥ 방향(1,m)
    Bp = sp.Matrix([sol, m * sol]); Cp = sp.Matrix([sol, sol]); Ap = sp.Matrix([4, 4]); O = sp.Matrix([0, 0])
    area = lambda P, Q, R: sp.Abs((Q - P)[0] * (R - P)[1] - (Q - P)[1] * (R - P)[0]) / 2
    S = area(Ap, Bp, Cp); T = area(O, Bp, Cp)
    L = sp.limit(sp.simplify(S / ((m - 1) * T)), m, 1, "+")
    report(14, L == tex(ans(14)), f"t={sp.simplify(sol)}, 극한={L}")


def p15():
    f = x ** 4 / 4 - sp.Rational(5, 3) * x ** 3 + x ** 2 + 8 * x + 9
    crit = sp.solve(sp.diff(f, x), x)
    maxs = [c0 for c0 in crit if sp.diff(f, x, 2).subs(x, c0) < 0]
    B, C = sp.symbols("B C")
    out = []
    for av in maxs:
        g = x ** 3 + B * x ** 2 + C * x - sp.Rational(1, 3)
        for s in sp.solve([sp.diff(g, x).subs(x, av), g.subs(x, av) - f.subs(x, av)], [B, C], dict=True):
            gv = g.subs(s)
            if sp.diff(gv, x, 2).subs(x, av) >= 0:
                continue                                   # x=a 에서 g 도 극대여야 함
            mins = [c0 for c0 in sp.solve(sp.diff(gv, x), x) if sp.diff(gv, x, 2).subs(x, c0) > 0]
            out += [gv.subs(x, c0) for c0 in mins]
    report(15, out == [tex(ans(15))], f"극대점={maxs}, 극솟값={out}")

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
