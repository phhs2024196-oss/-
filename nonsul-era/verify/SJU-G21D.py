#!/usr/bin/env python3
"""SJU-G21D (세종대 2021학년도 수시 논술고사[D] 자연계열) — sympy 검산. 정답 하드코딩 금지, 설정으로부터 독립 재계산."""
import re
import sys
from fractions import Fraction as Fr
from itertools import product, combinations

import sympy as sp

PREFIX = "SJU-G21D"
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
    a, K = sp.symbols("a K", real=True)
    # 우변 정적분을 상수 K 로 두고 x=0 대입 → aK=1
    sol_aK = sp.solve(sp.Eq(0, 1 - a * K), a)[0]
    F = sp.exp(x) - sol_aK * K * sp.exp(3 * x)
    f = sp.diff(F, x)
    Kval = sp.integrate(sp.exp(-t) * f.subs(x, t), (t, 0, sp.log(3)))
    aval = sp.simplify(sol_aK.subs(K, Kval))
    ans = parts(1, "1")
    ok1 = same(f, S(ans["f(x)"])) and same(aval, S(ans["a"]))
    # 원식 확인
    lhs = sp.integrate(f.subs(x, t), (t, 0, x))
    rhs = sp.exp(x) - aval * sp.exp(3 * x) * sp.integrate(sp.exp(-t) * f.subs(x, t), (t, 0, sp.log(3)))
    ok1 = ok1 and sp.simplify(lhs - rhs) == 0
    # (2) A(x)=(-x)(e^x - f) 최대 (x<0) — 수치 격자 + 정확 임계점
    A = -x * (sp.exp(x) - f)
    crit = [c for c in sp.solve(sp.diff(A, x), x) if c.is_real and c < 0]
    Amax = max(A.subs(x, c) for c in crit)
    grid = max(float(A.subs(x, -i / 1000)) for i in range(1, 5000))
    ok2 = same(Amax, S(md_answer(1, "2"))) and grid <= float(Amax) + 1e-12
    # (3) 넓이: l2 = 극댓값 수평선, l1 = x=0 접선
    xm = [c for c in sp.solve(sp.diff(f, x), x) if c.is_real][0]
    l2 = f.subs(x, xm)
    l1 = sp.diff(f, x).subs(x, 0) * x + f.subs(x, 0)
    xc = sp.solve(sp.Eq(l1, l2), x)[0]
    area = sp.integrate(l2 - f, (x, xm, xc)) + sp.integrate(l1 - f, (x, xc, 0))
    # 수치 교차: 영역 = {xm<=x<=0, f<=y<=min(l1,l2)}
    import mpmath as mp
    fn = sp.lambdify(x, f, "mpmath"); l1n = sp.lambdify(x, l1, "mpmath")
    num = mp.quad(lambda u: min(l1n(u), float(l2)) - fn(u), [float(xm), float(xc), 0])
    ok3 = same(area, S(md_answer(1, "3"))) and abs(num - float(area)) < 1e-10
    report(1, ok1 and ok2 and ok3, f"f={f} a={aval} maxA={Amax} area={sp.simplify(area)} (xm={xm}, xc={xc})")


def p02():
    from math import comb
    # (1) n=3: 7개 중 4개 동시 추출, 상금 직접 열거
    balls = [0] * 3 + [1] * 4
    prizes = [sum(5000 if b == 0 else 3000 for b in c) for c in combinations(range(7), 4) for b in [None] for c in [[balls[i] for i in c]]]
    N = len(prizes)
    mu = Fr(sum(prizes), N)
    var = Fr(sum(p * p for p in prizes), N) - mu * mu
    sd = sp.sqrt(sp.Rational(var.numerator, var.denominator))
    ok1 = same(sd, S(md_answer(2, "1")))
    # (2) n=2: 상태 전이 직접 계산
    from functools import lru_cache
    @lru_cache(None)
    def third_one(z, o, step):
        tot = z + o
        if step == 3:
            return Fr(o, tot)
        res = Fr(0)
        if o:
            res += Fr(o, tot) * third_one(z, o + 1, step + 1)
        if z:
            res += Fr(z, tot) * third_one(z - 1, o, step + 1)
        return res
    pr = third_one(2, 4, 1)
    ok2 = same(sp.Rational(pr.numerator, pr.denominator), S(md_answer(2, "2")))
    # (3) 정규근사: P(|Z|<=0.1k/sqrt(k/4)) >= 0.95  <=> 0.2 sqrt(k) >= 1.96
    kmin = next(k for k in range(1, 10000) if sp.Rational(1, 5) * sp.sqrt(k) >= sp.Rational(196, 100))
    ok3 = kmin == int(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3, f"sd={sd} P(3rd=1)={pr} kmin={kmin}")


def p03():
    import mpmath as mp
    # (3-3) 후보 f=(x+1)e^x 가 조건·거리식을 만족하는지 확인하고, (3-1) f'(1) 을 그로부터가 아니라 속력 등식에서 계산
    fsym = sp.Function("f")
    th = t**2 - 2 * t
    X = fsym(t) * sp.cos(th); Y = fsym(t) * sp.sin(th)
    sp2 = sp.simplify(sp.diff(X, t)**2 + sp.diff(Y, t)**2)
    target = sp.exp(2 * t) * (4 * (t**2 - 1)**2 + (t + 2)**2)
    # t=1 에서 f 에 무관한 항만 남음 → f'(1)^2
    d1 = sp.Symbol("d1", positive=True)
    eq = sp.expand(sp2.subs(sp.Derivative(fsym(t), t), d1).subs(t, 1)) - target.subs(t, 1)
    fp1 = sp.solve(eq, d1)
    ok1 = len(fp1) == 1 and same(fp1[0], S(md_answer(3, "1")))
    fc = S(md_answer(3, "3"))
    speed2 = sp2.subs(sp.Derivative(fsym(t), t), sp.diff(fc.subs(x, t), t)).subs(fsym(t), fc.subs(x, t))
    ok3 = sp.simplify(speed2 - target) == 0 and fc.subs(x, 0) == 1
    ok3 = ok3 and all(float(sp.diff(fc, x).subs(x, v)) > 0 for v in [0.01, 0.5, 1, 3])
    # (3-2) [증명-수치확인] 등식을 만족하는 f 에 대해 곱 <= 0 (여기선 =0) 및 대수 항등식 점검
    F, Fp = sp.symbols("F Fp", real=True)
    ident = sp.expand((Fp**2 - (x + 2)**2 * sp.exp(2 * x)) * (F**2 - (x + 1)**2 * sp.exp(2 * x))
                      + (2 * x - 2)**2 * (F**2 - (x + 1)**2 * sp.exp(2 * x))**2
                      - (F**2 - (x + 1)**2 * sp.exp(2 * x)) * (Fp**2 + F**2 * (2 * x - 2)**2 - sp.exp(2 * x) * (4 * (x**2 - 1)**2 + (x + 2)**2)))
    ok2 = sp.simplify(ident) == 0
    report(3, ok1 and ok2 and ok3, f"[증명-수치확인] f'(1)={fp1} 항등식 확인 f=(x+1)e^x 가 거리식 만족")


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
