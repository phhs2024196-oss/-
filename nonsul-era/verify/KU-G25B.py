#!/usr/bin/env python3
"""KU-G25B (고려대 2025학년도 수시 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KU-G25B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def golden(fn, lo, hi, sign=1, it=160):
    """구간 [lo,hi]에서 sign*fn 의 최소점(황금분할 탐색)."""
    r = (mp.sqrt(5) - 1) / 2
    for _ in range(it):
        m1, m2 = hi - r * (hi - lo), lo + r * (hi - lo)
        if sign * fn(m1) < sign * fn(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def p01():
    mp.mp.dps = 30
    X = lambda s: mp.cos(s)**3
    Y = lambda s: 2 * mp.sin(s)**3
    slope = lambda s: (6 * mp.sin(s)**2 * mp.cos(s)) / (-3 * mp.cos(s)**2 * mp.sin(s))   # dy/dt ÷ dx/dt
    xint = lambda s: X(s) - Y(s) / slope(s)                   # 접선의 x절편
    def t_of(a):                                              # 접점 매개변수(유일성: 격자에서 부호변화 1회)
        g = [mp.mpf(i) / 400 * mp.pi / 2 for i in range(1, 400)]
        ch = [i for i in range(len(g) - 1) if (xint(g[i]) - a) * (xint(g[i + 1]) - a) < 0]
        assert len(ch) == 1, (a, ch)
        return mp.findroot(lambda s: xint(s) - a, (g[ch[0]] + g[ch[0] + 1]) / 2)
    # (1)
    s = t_of(mp.mpf(1) / 2)
    m, c = slope(s), Y(s) - slope(s) * X(s)
    rhs = sp.sympify(md_answer(1, "1").split("=")[1], locals={"x": x})
    ok1 = abs(m - float(rhs.coeff(x))) < 1e-12 and abs(c - float(rhs.subs(x, 0))) < 1e-12
    # (2) a^3 tan(g(a)) 최대 — 격자 + 정밀화
    H = lambda a: a**3 * abs(slope(t_of(a)))
    grid = [mp.mpf(i) / 200 for i in range(2, 199)]
    a0 = max(grid, key=H)
    amax = golden(H, a0 - mp.mpf(1) / 200, a0 + mp.mpf(1) / 200, sign=-1)
    ok2 = abs(H(amax) - float(sp.sympify(md_answer(1, "2")))) < 1e-10
    # (3) S1+S2 = 곡선과 '접선·좌표축' 사이 넓이: ∫ y dx - ∫_0^1 max(L,0) dx
    under = mp.quad(lambda u: Y(u) * 3 * mp.cos(u)**2 * mp.sin(u), [0, mp.pi / 2])
    def S(a):
        u = t_of(a); k = slope(u); b = Y(u) - k * X(u)
        return under - mp.quad(lambda xx: max(k * xx + b, 0), [0, a, 1])
    grid = [mp.mpf(i) / 100 for i in range(2, 99)]
    a1 = min(grid, key=S)
    amin = golden(S, a1 - mp.mpf(1) / 100, a1 + mp.mpf(1) / 100, it=80)
    ok3 = abs(amin - float(sp.sympify(md_answer(1, "3")))) < 1e-7
    report(1, ok1 and ok2 and ok3, f"l: y={float(m):.6f}x+{float(c):.6f}  max={float(H(amax)):.10f}@a={float(amax):.6f}  argmin S={float(amin):.10f}")


def p02():
    a, b, c, d, k, h = sp.symbols("a b c d k h", real=True)
    f = a * x**3 + b * x**2 + c * x + d
    eqs = [sp.diff(f, x, 2).subs(x, sp.Rational(2, 3))]                        # (가)
    gl = (1 - x) * sp.exp(-x) + sp.cos(x)
    gr = x * sp.diff(f, x) + f
    eqs.append(sp.limit(gl, x, 0, "-") - gr.subs(x, 0))                          # 연속
    dl = sp.limit((gl.subs(x, h) - gl.subs(x, 0)) / h, h, 0, "-")
    dr = sp.limit((gr.subs(x, h) - gr.subs(x, 0)) / h, h, 0, "+")
    eqs.append(dl - dr)                                                         # 미분가능
    sol = sp.solve(eqs, [b, c, d], dict=True)[0]
    fa = f.subs(sol)
    # (다): k = f(t) - t f'(t) 의 실근 3개인 k 범위 = (극소값, 극대값)
    kt = sp.expand(fa.subs(x, t) - t * sp.diff(fa, x).subs(x, t))
    crit = sp.solve(sp.diff(kt, t), t)
    vals = sorted([sp.simplify(kt.subs(t, r).subs(a, 1)) for r in crit])  # a>0 에서 순서 동일
    vals_a = sorted([sp.simplify(kt.subs(t, r)) for r in crit], key=lambda e: float(e.subs(a, 1)))
    asol = [s_ for s_ in sp.solve([vals_a[0] - 2, vals_a[1] - sp.Rational(62, 27)], [a], dict=True) if s_[a] > 0]
    fsol = sp.expand(fa.subs(asol[0]))
    ok1 = sp.simplify(fsol - sp.sympify(md_answer(2, "1"), locals={"x": x})) == 0 and len(asol) == 1
    # 근 개수 수치 확인
    for kv in (2.01, 2.2, 2.29):
        roots = [r for r in sp.Poly(kt.subs(a, asol[0][a]) - kv, t).nroots() if abs(sp.im(r)) < 1e-12]
        ok1 = ok1 and len(roots) == 3
    # (2) 수치 적분
    ff = sp.lambdify(x, fsol, "mpmath")
    F = lambda xx: mp.quad(lambda u: abs(ff(u)), [0, min(xx, 1), xx] if xx > 1 else [0, xx])
    val = mp.quad(lambda xx: ff(xx) * mp.e**F(xx), [0, 1, 2])
    ok2 = abs(val - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2")), 30)))) < 1e-9
    report(2, ok1 and ok2, f"f={fsol}  integral={mp.nstr(val, 15)}")


def p03():
    B = sp.Matrix([5, 3]); OA = sp.Matrix([3, 3])
    th = sp.symbols("theta", real=True)
    P = B + sp.Matrix([sp.cos(th), sp.sin(th)])
    def argmin(expr):
        fn = sp.lambdify(th, expr, "mpmath")
        g = [2 * mp.pi * i / 20000 for i in range(20000)]
        t0 = min(g, key=fn)
        tm = mp.findroot(lambda s: mp.diff(fn, s), t0)
        return tm, fn(tm)
    e1 = (OA + P).dot(OA + P)
    tm, v = argmin(e1)
    ans = md_answer(3, "1").replace(" ", "")
    mval = sp.sympify(re.search(r"min=([^;]+)", ans).group(1))
    Pmd = sp.sympify(re.search(r"P=(\(.*\))", ans).group(1))
    Pn = [float(P[0].subs(th, tm)), float(P[1].subs(th, tm))]
    ok1 = abs(v - float(mval)) < 1e-10 and all(abs(Pn[i] - float(Pmd[i])) < 1e-10 for i in range(2))
    A = [OA * sp.Rational(i, 3) for i in range(4)]
    w = (P - A[1]) + 4 * (P - A[2]) + (P - A[3])
    tm2, v2 = argmin(w.dot(w))
    ab = sp.sympify(md_answer(3, "2"))
    ok2 = abs(v2 - float(ab[0] + ab[1] * sp.sqrt(10))) < 1e-9 and all(sp.Integer(q) == q for q in ab)
    report(3, ok1 and ok2, f"min1={mp.nstr(v, 12)} P=({Pn[0]:.10f},{Pn[1]:.10f}) min2={mp.nstr(v2, 15)}")


def p04():
    def brute(m):
        return sum(1 for bl in range(m + 1) for ye in range(2 * m)
                   if 0 <= 2 * m - bl - ye <= 2 * m + 2)
    an = sp.sympify(md_answer(4, "2"), locals={"n": n})
    ok2 = all(brute(m) == an.subs(n, m) for m in range(1, 31))
    ok1 = brute(2) == sp.sympify(md_answer(4, "1"))
    # 열거로 확인한 a_n 다항식(보간)으로 급수 계산 — md 식 대신 독립 보간식 사용
    k = sp.symbols("k", integer=True, positive=True)
    poly = sp.interpolate([(m, brute(m)) for m in range(1, 5)], k)
    S = sp.summation(1 / (poly + 8 * k + 15), (k, 1, sp.oo))
    ok3 = sp.simplify(S - sp.sympify(md_answer(4, "3"))) == 0
    # 대표 오답(첫 항만 남김) 재현
    wrong = sp.Rational(2, 9) * sp.Rational(1, 3)
    report(4, ok1 and ok2 and ok3, f"a_2={brute(2)} a_n(interp)={sp.expand(poly)} sum={S} (오답 재현 {wrong})")


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
