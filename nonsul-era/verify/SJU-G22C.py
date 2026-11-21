#!/usr/bin/env python3
"""SJU-G22C (세종대 2022학년도 수시 논술고사 [C]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G22C"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.strip().split("=", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import mpmath as mp
    mp.mp.dps = 30
    f = 2 * sp.exp(x) - sp.exp(-x)
    fp = sp.diff(f, x)
    # (1) [증명-수치확인] f' > 0 (격자)
    ok1 = all(fp.subs(x, v) > 0 for v in [sp.Rational(k, 10) for k in range(-100, 101)])
    finv = lambda y: mp.findroot(lambda u: 2 * mp.e**u - mp.e**(-u) - y, 0)
    # (2) [증명-수치확인] F'(x)=x f^{-1}(x): F'(1)=0, 부호 음→양, F''(1)>0
    Fp = lambda v: v * finv(v)
    ok2 = abs(Fp(1)) < 1e-20 and Fp(0.9) < 0 and Fp(1.1) > 0 and mp.diff(Fp, 1) > 0
    # (3) F(1) 수치 적분 (역함수 직접 계산)
    F1 = mp.quad(lambda v: v * finv(v), [0, 1])
    ok3 = abs(F1 - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "3")), 40)))) < 1e-20
    report(1, ok1 and ok2 and ok3, f"[증명-수치확인] f'>0, F'(1)=0, F''(1)={float(mp.diff(Fp,1)):.6f}, F(1)={float(F1):.12f}")


def p02():
    T = sp.symbols("T", positive=True)
    f = T * (x - T) * (x - T - 1)
    L = sp.diff(f, x).subs(x, T) * (x - T)
    area = sp.integrate(f - L, (x, 0, T))
    ok1 = sp.simplify(area - sp.sympify(md_answer(2, "1"), locals={"t": T})) == 0 and sp.factor(f - L) == T * (x - T)**2
    a, b = sp.symbols("a b", real=True)
    sols = sp.solve([(T - a)**2 + b**2 - a**2, b - (a - T) / T], [a, b], dict=True)   # 원 지남 + 법선 위
    sol = [s for s in sols if s[a].subs(T, 2) > 0 and s[b].subs(T, 2) < 0][0]
    A_, B_ = sol[a], sol[b]
    lim = sp.limit(T * B_**2 / A_, T, sp.oo)
    ok2 = sp.simplify(lim - sp.sympify(md_answer(2, "2"))) == 0
    # (3) θ = ∠PZQ 를 좌표로 직접: P=(2a-t,0)
    def sin_theta(tv):
        av, bv = float(A_.subs(T, tv)), float(B_.subs(T, tv))
        import math
        P = (2 * av - tv - av, -bv); Q = (tv - av, -bv)
        cr = P[0] * Q[1] - P[1] * Q[0]
        return abs(cr) / (math.hypot(*P) * math.hypot(*Q))
    t1 = float(sp.sympify(md_answer(2, "3")))
    t2 = 14 - t1
    ok3 = 0 < t1 < t2 and abs(sin_theta(t1) - sin_theta(t2)) < 1e-12
    # 유일성: s(t)=2t/(t^2+1) 일 때 sin 같고 합 14 → 대수적으로
    s1, s2 = sp.symbols("s1 s2", positive=True)
    cand = sp.solve([2 * s1 / (s1**2 + 1) - 2 * s2 / (s2**2 + 1), s1 + s2 - 14], [s1, s2], dict=True)
    cand = [c for c in cand if c[s1] < c[s2]]
    ok3 = ok3 and len(cand) == 1 and abs(float(cand[0][s1]) - t1) < 1e-12 and abs(sin_theta(3.0) - 6 / 10) < 1e-12
    report(2, ok1 and ok2 and ok3, f"area={area} a={sp.simplify(A_)} lim={lim} t1={t1:.10f} sin(t1)={sin_theta(t1):.10f}")


def p03():
    import numpy as np
    import mpmath as mp
    a = sp.symbols("a", positive=True)
    ell = x**3 / 3 + x**2 + sp.Rational(5, 3)
    # (3-2) ℓ(a) = a ℓ'(a)
    roots = [r for r in sp.solve(sp.Eq(ell.subs(x, a), a * sp.diff(ell, x).subs(x, a)), a) if r.is_real and r > 0]
    av = roots[0]
    fa = [v for v in sp.solve(sp.Eq(ell.subs(x, av)**2, av**2 + (sp.Symbol("y") + sp.sqrt(2))**2), sp.Symbol("y")) if v > 0][0]
    ok2 = len(roots) == 1 and sp.simplify(sp.Matrix([av, fa]) - sp.Matrix(list(sp.sympify(md_answer(3, "2"))))) == sp.zeros(2, 1)
    # 별해 다항식 확인
    hpoly = a**2 + (a * (a + 1) * sp.sqrt(a**2 + 2 * a - 1))**2 - ell.subs(x, a)**2
    okalt = sp.expand(hpoly - sp.Rational(1, 9) * (a - 1) * (8*a**5 + 38*a**4 + 65*a**3 + 55*a**2 + 25*a + 25)) == 0
    # (3-3) md 의 f 로 조건과 경로 길이를 직접 재구성
    P = parts(md_answer(3, "3"))
    f1 = sp.sympify(P["f1"], locals={"x": x}); f2 = sp.sympify(P["f2"], locals={"x": x})
    F = lambda v: float(f1.subs(x, v)) if v <= 1 else float(f2.subs(x, v))
    xs = np.linspace(0.01, 2, 200)
    fd = sp.lambdify(x, sp.Piecewise((sp.diff(f1, x), x <= 1), (sp.diff(f2, x), True)))
    fdd = sp.lambdify(x, sp.Piecewise((sp.diff(f1, x, 2), x <= 1), (sp.diff(f2, x, 2), True)))
    okcond = all(F(v) > 0 and fd(v) > 0 and fdd(v) > 0 for v in xs)
    okjoin = abs(F(1) - float(f2.subs(x, 1))) < 1e-12 and abs(float(sp.diff(f1, x).subs(x, 1) - sp.diff(f2, x).subs(x, 1))) < 1e-12
    # 접선 at a=1 이 A(0,-√2) 지남
    oktan = abs(float(-1 * sp.diff(f2, x).subs(x, 1) + f2.subs(x, 1) + sp.sqrt(2))) < 1e-12
    # 최단 경로 길이 재구성: x<=1 선분, x>1 선분+호
    def L(v):
        if v <= 1:
            return mp.sqrt(v**2 + (F(v) + mp.sqrt(2))**2)
        return mp.sqrt(1 + (F(1) + mp.sqrt(2))**2) + mp.quad(lambda s: mp.sqrt(1 + fd(float(s))**2), [1, v])
    ok3 = okcond and okjoin and oktan and all(abs(L(v) - float(ell.subs(x, v))) < 1e-9 for v in [0, 0.3, 0.7, 1, 1.4, 2])
    # (3-1) [증명-수치확인] x=1 좌우 미분계수
    h = mp.mpf("1e-6")
    dl = (L(1) - L(1 - h)) / h; dr = (L(1 + h) - L(1)) / h
    ok1 = abs(dl - dr) < 1e-4
    report(3, ok1 and ok2 and ok3 and okalt, f"a={av} f(a)={fa} 좌우미분={float(dl):.6f},{float(dr):.6f} 조건(가)={okcond} 경로길이 일치={ok3}")


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
