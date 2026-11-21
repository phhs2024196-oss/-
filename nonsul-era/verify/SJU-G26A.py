#!/usr/bin/env python3
"""SJU-G26A (세종대 2026학년도 논술고사 자연계열A) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SJU-G26A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
mp.mp.dps = 25


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no, sub, loc=None):
    """'k = v; k2 = v2' 꼴 정답을 dict 로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = seg.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def golden(fun, lo, hi, maximize=False, it=120):
    """황금분할 탐색(단봉 구간)."""
    sgn = -1 if maximize else 1
    gr = (mp.sqrt(5) - 1) / 2
    a_, b_ = mp.mpf(lo), mp.mpf(hi)
    c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    fc, fd = sgn * fun(c_), sgn * fun(d_)
    for _ in range(it):
        if fc < fd:
            b_, d_, fd = d_, c_, fc
            c_ = b_ - gr * (b_ - a_); fc = sgn * fun(c_)
        else:
            a_, c_, fc = c_, d_, fd
            d_ = a_ + gr * (b_ - a_); fd = sgn * fun(d_)
    return (a_ + b_) / 2


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1, abs(float(b)))


def p01():
    # (나) 미분: e^{-x} finv(x) = d/dx x^2
    finv = sp.simplify(sp.diff(x**2, x) * sp.exp(x))
    chk = sp.simplify(sp.integrate(sp.exp(-t) * finv.subs(x, t), (t, 0, x)) - x**2) == 0
    v1 = finv.subs(x, sp.log(2)); v2 = finv.subs(x, sp.Rational(1, 2))
    v3 = 1 / sp.diff(finv, x).subs(x, sp.Rational(1, 2))            # f'(finv(1/2))
    md = [sp.sympify(s) for s in md_answer(1, "1").split(";")]
    ok1 = chk and all(sp.simplify(a - b) == 0 for a, b in zip([v1, v2, v3], md))
    # (1-2) f 를 수치 역함수로 직접 적분
    fl = sp.lambdify(x, finv, "mpmath")
    f = lambda y: mp.findroot(lambda u: fl(u) - y, (mp.mpf(0), mp.mpf(3)), solver="bisect", verify=False) if y > 0 else mp.mpf(0)
    I = mp.quad(f, [0, mp.sqrt(mp.e)])
    ok2 = close(I, sp.N(sp.sympify(md_answer(1, "2")), 30))
    # (1-3) g=f-x/4 를 [0,4ln2] 에서 직접 최대화
    g = lambda y: f(y) - y / 4
    L = 4 * mp.log(2)
    xs = [L * i / 400 for i in range(1, 400)]
    a0 = max(xs, key=g)
    al = golden(g, a0 - L / 400, a0 + L / 400, maximize=True)
    al_s = sp.symbols("alpha")
    mdg = sp.sympify(md_answer(1, "3"), locals={"alpha": al_s})
    ok3 = close(g(al), mdg.subs(al_s, sp.Float(str(al), 30))) and g(al) > g(al + 0.1) and g(al) > g(al - 0.1)
    report(1, ok1 and ok2 and ok3, f"finv={finv} int={float(I):.12f} alpha={float(al):.10f} g(alpha)={float(g(al)):.10f}")


def p02():
    a, b, C = sp.symbols("a b C", real=True)
    # (2-1) f=e^x+C 꼴 후보를 대입해 항등식으로 a,b 결정 (f'=e^x 는 2번 미분으로 독립 확인)
    F = sp.Function("F")
    lhs_f = lambda fx: sp.integrate((2 * t - x) * fx.subs(x, t), (t, 0, x))
    rhs = (x - 2) * sp.exp(x) + a * x + b
    d2 = sp.diff(sp.diff(rhs, x), x)                                # = x f'(x) (식 정리) → f' = d2/x
    fprime = sp.simplify(d2 / x)
    fx = sp.integrate(fprime, x) + C
    eq = sp.simplify(sp.expand(lhs_f(fx) - rhs))
    sol = sp.solve([eq.subs(x, 0), sp.diff(eq, x).subs(x, 0)], [a, b], dict=True)[0]
    ident = sp.simplify(eq.subs(sol)) == 0
    dif = sp.simplify(fx.subs(x, 2) - fx.subs(x, 1))
    m1 = md_parts(2, "1")
    ok1 = ident and sol[a] == sp.sympify(m1["a"]) and sol[b] == sp.sympify(m1["b"]) and sp.simplify(dif - sp.sympify(m1["d"])) == 0
    # (2-2) r(0) 존재 조건 → C, r(x) 를 극한으로 직접 계산
    h = sp.symbols("h", positive=True)
    gneg = x**2 + 2 * x + 3
    Cv = sp.solve(sp.Eq(fx.subs(x, 0), gneg.subs(x, 0)), C)[0]
    fpos = fx.subs(C, Cv)
    def gpw(z):
        return sp.Piecewise((gneg.subs(x, z), z < 0), (fpos.subs(x, z), True))
    def rval(x0):
        x0 = sp.nsimplify(x0)
        if x0 == 0:
            expr = (fpos.subs(x, h) - gneg.subs(x, -2 * h)) / h
        elif x0 > 0:
            expr = (fpos.subs(x, x0 + h) - fpos.subs(x, x0 - 2 * h)) / h
        else:
            expr = (gneg.subs(x, x0 + h) - gneg.subs(x, x0 - 2 * h)) / h
        return sp.limit(expr, h, 0, "+")
    m2 = md_parts(2, "2")
    ok2 = sp.simplify(fpos - sp.sympify(m2["f"], locals={"x": x})) == 0
    for x0 in (-2, sp.Rational(-1, 3), 0, sp.Rational(1, 2), 3):
        key = "r0" if x0 == 0 else ("rneg" if x0 < 0 else "rpos")
        ok2 = ok2 and sp.simplify(rval(x0) - sp.sympify(m2[key], locals={"x": x}).subs(x, x0)) == 0
    # (2-3) k 를 촘촘히 훑어 교점 수를 수치로 세고, md 집합과 대조
    rneg = sp.lambdify(x, sp.sympify(m2["rneg"], locals={"x": x}))
    rpos = sp.lambdify(x, sp.sympify(m2["rpos"], locals={"x": x}))
    r0 = float(sp.sympify(m2["r0"]))
    import numpy as np
    xn = -np.logspace(-9, 2, 20001)[::-1]
    xp = np.logspace(-9, 1.2, 20001)
    def count(k):
        c = 0
        dn = rneg(xn) - (np.exp(2 * xn) + k)
        c += int(np.sum(np.sign(dn[:-1]) * np.sign(dn[1:]) < 0))
        c += 1 if abs(r0 - (1 + k)) < 1e-12 else 0
        dp = rpos(xp) - (np.exp(2 * xp) + k)
        c += int(np.sum(np.sign(dp[:-1]) * np.sign(dp[1:]) < 0))
        # 접점(중근): x>0 에서 최소 |차| 가 0 근처이고 부호가 안 바뀌면 1개
        tt = sp.symbols("tt")
        roots = [complex(z) for z in sp.Poly(tt**2 - 3 * tt + sp.nsimplify(k), tt).nroots()]
        if any(abs(z.imag) < 1e-12 for z in roots) and abs(roots[0] - roots[1]) < 1e-9 and roots[0].real > 1:
            c += 1
        return c
    def in_md(k):
        s = md_answer(2, "3").replace(" ", "")
        iv, pts = s.split("U")
        hi = float(sp.sympify(iv.split(",")[1].rstrip("]")))
        P = [float(sp.sympify(v)) for v in pts.strip("{}").split(",")]
        return k <= hi + 1e-12 or any(abs(k - p) < 1e-12 for p in P)
    ks = [-3 + 0.0137 * i for i in range(700)] + [2.0, 2.25, 4.0, 5.0]
    bad = [k for k in ks if (count(k) == 2) != in_md(k)]
    ok3 = not bad
    report(2, ok1 and ok2 and ok3, f"a,b={sol[a]},{sol[b]} f={fpos} r(0)={rval(0)} k-mismatch={bad[:5]}")


def p03():
    # (3-1) 좌표로 내접원 반지름 = 넓이/반둘레
    s3 = math.sqrt(3)
    A, B, Cp = (0.0, 0.0), (2.0, 0.0), (1.0, s3)
    def inr(P, Q, R):
        a_, b_, c_ = math.dist(Q, R), math.dist(P, R), math.dist(P, Q)
        area = abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2
        return area / ((a_ + b_ + c_) / 2)
    fnum = lambda xx: inr(A, (xx, 0.0), Cp) + inr((xx, 0.0), B, Cp)
    m1 = md_parts(3, "1")
    fmd = sp.lambdify(x, sp.sympify(m1["f"], locals={"x": x}))
    ok1 = all(abs(fnum(v) - fmd(v)) < 1e-12 for v in (0.1, 0.5, 1.0, 1.37, 1.9))
    grid = [i / 20000 for i in range(1, 40000)]
    vals = [fnum(v) for v in grid]
    lo, hi = min(vals), max(vals)
    rng = [float(sp.sympify(v)) for v in m1["range"].strip("[]").split(",")]
    ok1 = ok1 and abs(lo - rng[0]) < 1e-3 and abs(hi - rng[1]) < 1e-9
    # (3-2) Q(1/4) 의 접선과 x=1 의 교점 P, P 에서 다른 접선의 접점 R 를 수치로 찾고 각 계산
    mp.mp.dps = 30
    fm = lambda v: fmd(v)
    fq = lambda u: mp.sqrt(3) * (3 - mp.sqrt(u**2 - 2 * u + 4)) / 3
    assert abs(fq(0.7) - fmd(0.7)) < 1e-12
    q = mp.mpf(1) / 4
    Py = fq(q) + mp.diff(fq, q) * (1 - q)
    Rx = mp.findroot(lambda s: fq(s) + mp.diff(fq, s) * (1 - s) - Py, 1.6)
    v1 = (q - 1, fq(q) - Py); v2 = (Rx - 1, fq(Rx) - Py)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    sin_t = abs(cross) / (mp.sqrt(v1[0]**2 + v1[1]**2) * mp.sqrt(v2[0]**2 + v2[1]**2))
    ok2 = close(sin_t, sp.N(sp.sympify(md_answer(3, "2")), 30))
    # (3-3) g(t)=∫|f-t| 를 수치 최소화
    def G(tv):
        # |f-t| 의 꺾이는 점을 분할점으로
        pts = [0, 2]
        if fq(0) < tv < fq(1):
            u = mp.findroot(lambda s: fq(s) - tv, (mp.mpf(0), mp.mpf(1)), solver="bisect", verify=False)
            pts = [0, u, 2 - u, 2]
        return mp.quad(lambda s: abs(fq(s) - tv), pts)
    ts = [fq(0) + (fq(1) - fq(0)) * i / 200 for i in range(1, 200)]
    t0 = min(ts, key=G)
    dt = (fq(1) - fq(0)) / 200
    tm = golden(G, t0 - dt, t0 + dt)
    ok3 = close(tm, sp.N(sp.sympify(md_answer(3, "3")), 30), 1e-8)
    report(3, ok1 and ok2 and ok3, f"range=[{lo:.6f},{hi:.10f}] R_x={float(Rx):.10f} sin={float(sin_t):.12f} argmin={float(tm):.10f}")


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
