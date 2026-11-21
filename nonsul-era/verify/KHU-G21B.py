#!/usr/bin/env python3
"""KHU-G21B (경희대 2021학년도 논술 자연계 12.06 오전, 수학 논제 I-1·I-2·I-3) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-G21B"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def p01():
    """[논제 I-1] 이항분포, mσ=80, P(|X-m|≥2m/5)=0.0456 (정규근사, 표 값)."""
    n, p = sp.symbols("n p", positive=True)
    m = n * p; sg = sp.sqrt(n * p * (1 - p))
    table = {1: 0.3413, 2: 0.4772, 3: 0.4987}
    # 2m/(5σ)=z 에서 1-2P(0≤Z≤z)=0.0456 이 되는 z 를 표에서 찾는다
    z = [k for k, v in table.items() if abs((1 - 2 * v) - 0.0456) < 1e-12][0]
    sols = sp.solve([sp.Eq(2 * m / (5 * sg), z), sp.Eq(m * sg, 80)], [n, p], dict=True)
    good = [s for s in sols if s[n] >= 60 and sp.Rational(1, 10) <= s[p] <= sp.Rational(1, 2)]
    ok = len(good) == 1 and good[0][n] == 100 and good[0][p] == sp.Rational(1, 5)
    nn, pp = int(good[0][n]), float(good[0][p])
    approx = 2 * (1 - Phi(2.0))
    report(1, ok, f"z={z}, (n,p)={good[0][n]},{good[0][p]}; np={nn*pp}, npq={nn*pp*(1-pp)}≥5; 정규 2(1-Φ(2))={approx:.4f}")


def p02():
    """[논제 I-2] OP=1, 원 C1이 l(P), m(Q)에 접함, O1Q 연장선과 n의 교점 R, S=넓이(O1PR)."""
    th = sp.symbols("theta", positive=True)
    # 좌표로 독립 계산: l: x축, m: 각 2θ, n: 각 3θ
    P = sp.Matrix([1, 0]); r = sp.tan(th); O1 = sp.Matrix([1, r])
    u = sp.Matrix([sp.cos(2 * th), sp.sin(2 * th)])
    Q = u * (O1.dot(u))                      # O1에서 m에 내린 수선의 발
    d = Q - O1
    s = sp.symbols("s")
    v = sp.Matrix([sp.cos(3 * th), sp.sin(3 * th)])
    w = sp.symbols("w")
    sol = sp.solve(list(O1 + s * d - w * v), [s, w], dict=True)[0]
    R = O1 + sol[s] * d
    area = sp.Rational(1, 2) * sp.Abs((P - O1)[0] * (R - O1)[1] - (P - O1)[1] * (R - O1)[0])
    # 수치로 S(θ)=tan^2θ sin2θ 확인
    Sform = sp.tan(th) ** 2 * sp.sin(2 * th)
    ok1 = all(abs(float(area.subs(th, t)) - float(Sform.subs(th, t))) < 1e-10 for t in [0.2, 0.5, 0.7, 0.9, 1.0])
    I = sp.integrate(sp.expand_trig(sp.simplify(Sform.rewrite(sp.cos))), (th, sp.pi / 6, sp.pi / 4))
    Inum = sp.Integral(area, (th, sp.pi / 6, sp.pi / 4)).evalf(30)
    target = sp.log(sp.Rational(3, 2)) - sp.Rational(1, 4)
    ok2 = abs(sp.N(I - target, 30)) < 1e-20 and abs(Inum - sp.N(target, 30)) < 1e-15
    report(2, ok1 and ok2, f"S(θ)=tan²θ·sin2θ(좌표 5점 일치), ∫={sp.nsimplify(sp.N(I,40), [sp.log(2), sp.log(3)])} ≈ {float(Inum):.10f}")


def p03():
    """[논제 I-3] f(x+y)=f(x)f(y)e^{2xy}, f'(0)=0, f>0."""
    x, y, t = sp.symbols("x y t", real=True)
    # (1) P=ln f, P(x+y)=P(x)+P(y)+2xy, P'(0)=0 → P'(x)=2x+P'(0) → P=x^2 (P(0)=0)
    P = sp.Function("P")
    ode = sp.dsolve(sp.Eq(P(x).diff(x), 2 * x + 0), P(x), ics={P(0): 0})
    f = sp.exp(ode.rhs)
    ok1 = sp.simplify(f.subs(x, x + y) - f * f.subs(x, y) * sp.exp(2 * x * y)) == 0 and sp.diff(f, x).subs(x, 0) == 0
    # (2) g'(f(x))f'(x)=2x(1+2x^2)f(√2x), g(1)=0 → g(f(x)) = ∫_0^x ...
    rhs = 2 * x * (1 + 2 * x**2) * f.subs(x, sp.sqrt(2) * x)
    gfe = sp.integrate(rhs.subs(x, t), (t, 0, 1))       # f(0)=1, f(1)=e
    ok2 = sp.simplify(gfe - sp.exp(2)) == 0 and sp.simplify(f.subs(x, 1) - sp.E) == 0
    # (3) ∫_0^x t f(x-t) h(x-t) dt = f(x)-1 → h 를 다항식 가정 없이: 두 번 미분
    h = sp.Function("h")
    s = sp.symbols("s", real=True)
    # s=x-t: ∫_0^x (x-s) f(s) h(s) ds = f(x)-1 → 두 번 미분: f(x)h(x) = f''(x)
    H = sp.simplify(sp.diff(f, x, 2) / f)
    lhs = sp.integrate(((x - s) * f.subs(x, s) * H.subs(x, s)), (s, 0, x))
    ok3a = sp.simplify(lhs - (f - 1)) == 0
    a = sp.symbols("a", real=True)
    # 법선: (a,h(a))를 지나고 기울기 -1/h'(a), x축과 (49,0)
    eq = sp.Eq(0 - H.subs(x, a), -1 / sp.diff(H, x).subs(x, a) * (49 - a))
    sols = [r for r in sp.solve(eq, a) if r.is_real]
    ok3 = ok3a and sols == [1]
    report(3, ok1 and ok2 and ok3, f"f={f}, g(e)={sp.simplify(gfe)}, h={sp.expand(H)}, a={sols}")


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
