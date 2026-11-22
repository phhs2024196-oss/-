"""DKU-G22A 검산: 문제 조건에서 독립 재계산한 값을 items/DKU/PAST 의 answer.value 와 비교."""
PREFIX = "DKU-G22A"
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
    # 논제1: g'(x)=f'(f(x)) f'(x) 의 서로 다른 실근 개수(부호변화 + 중근 여부 판별)
    def nreal(n, E):
        def fp(z):
            return n * (2 * z - z * z) * np.exp(-z)  # f'(z) = n(2z - z^2)e^{-z} (e 는 지수함수 그대로)
        def f(z):
            return n * z * z * np.exp(-z)
        xs = np.linspace(-8.000013, 8.000071, 800001)
        gp = fp(f(xs)) * fp(xs)
        return int(np.sum(gp[:-1] * gp[1:] < 0))
    # e=2.7 로 계산하라는 조건: f 의 극댓값 f(2)=4n/e^2 을 e=2.7 로 비교
    sol = []
    for nn in range(1, 11):
        # 극대 f(2)=4n/e^2 (e=2.7 및 실제 e 모두) 와 2 비교, 수치 근 개수도 병행
        c27 = 4 * nn / 2.7**2
        cE = 4 * nn / float(mp.e) ** 2
        cnt = nreal(nn, mp.e)
        if cnt == 3:
            sol.append(nn)
        assert (c27 <= 2) == (cE <= 2) or nn not in (3, 4)
    got = sorted(sol)
    exp = [int(float(v)) for v in sym_set(md_answer(1, "1"))]
    sub(got == exp, f"논제1 n={got}")
    # 논제2: C^1 인 구체적 h 를 만들어 직접 수치적분
    u = sp.symbols("u")
    c = sp.symbols("c0:4")
    pol = sum(c[i] * (x - 1) ** i for i in range(4))
    eqs = [pol.subs(x, 3) - pol.subs(x, 1) - 1,
           sp.diff(pol, x).subs(x, 3) - sp.diff(pol, x).subs(x, 1) / 3,
           sp.integrate(pol, (x, 1, 3)) - 9]
    sl = sp.solve(eqs + [c[3] - 0], c, dict=True)[0]
    pf = sp.lambdify(x, pol.subs(sl), "mpmath")
    dpf = sp.lambdify(x, sp.diff(pol.subs(sl), x), "mpmath")

    def hv(z):
        z = mp.mpf(z)
        k = 0
        while z >= 3:
            z /= 3
            k += 1
        return pf(z) + k

    def hd(z):
        z = mp.mpf(z)
        k = 0
        while z >= 3:
            z /= 3
            k += 1
        return dpf(z) / mp.mpf(3) ** k
    # 연속성 및 도함수 연속성 점검(x=3,9,27,81)
    chk = all(abs(hv(b + mp.mpf(10) ** -15) - hv(b - mp.mpf(10) ** -15)) < 1e-9 and
              abs(hd(b + mp.mpf(10) ** -15) - hd(b - mp.mpf(10) ** -15)) < 1e-9 for b in (3, 9, 27, 81))
    mp.mp.dps = 30
    chk = chk and abs(mp.quad(hv, [1, 3]) - 9) < 1e-12
    I = mp.quad(lambda z: hv(z) + hd(z), [1, 3, 9, 27, 81])
    sub(chk and abs(I - sp.Rational(md_answer(1, "2"))) < 1e-8, f"논제2 numeric={mp.nstr(I, 12)}")
    # 논제3: s=S(a)>0, S(b)=4s+3 일 때 조건(2)의 등식을 s 에 대한 방정식으로 수치 풀이
    def eq(s):
        w = 4 * s + 3
        lhs = mp.quad(lambda S: S * mp.e**S / (S + 1) ** 2, [s, w])
        rhs = mp.e**s / (w + 1) * (mp.e ** (6 * (s + 1)) - 67 * mp.e ** (3 * (s + 1)) + 252)
        return lhs - rhs
    grid = [mp.mpf(i) / 50 for i in range(1, 400)]
    vals = [eq(s) for s in grid]
    roots = [mp.findroot(eq, (grid[i], grid[i + 1]), solver="anderson") for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0]
    ok3 = len(roots) == 1 and abs(4 * roots[0] + 3 - sym(md_answer(1, "3")).evalf(30)) < 1e-9
    sub(ok3, f"논제3 s={mp.nstr(roots[0], 10) if roots else None}, S(b)={mp.nstr(4 * roots[0] + 3, 10) if roots else None}")
    flush(1)


def p02():
    # 논제1
    xs = np.linspace(-60.000013, 40.000071, 2000001)
    phi = np.exp(xs) - np.abs(xs**2 - np.e * xs)
    sc = int(np.sum(phi[:-1] * phi[1:] < 0))
    near = [i for i in range(1, len(xs) - 1) if abs(phi[i]) < 1e-6 and abs(phi[i]) <= abs(phi[i - 1]) and abs(phi[i]) <= abs(phi[i + 1]) and phi[i - 1] * phi[i + 1] >= 0]
    sub(sc + len(near) == int(md_answer(2, "1")) and not near, f"논제1 근의 개수={sc}, 접근={len(near)}")
    # 논제2: (1) 접선 -> t=u-1, 적분 조건으로 u 결정
    uu = sp.symbols("u", real=True)
    usol = sp.solve(sp.Eq(sp.exp(uu) - sp.exp(uu - 1), 1 - sp.exp(-1)), uu)[0]
    t0 = usol - 1
    # 접선 y=e^u (x-t), 점 (0,f(0)) 에서 접함: f(0)=e^u(-t), f'(0)=e^u
    f0 = sp.exp(usol) * (0 - t0)
    f1 = sp.exp(usol)
    a_, b_ = sp.symbols("a b")
    fc = a_ * x**3 + b_ * x**2 + f1 * x + f0
    bs = sp.solve(fc.subs(x, sp.Rational(3, 2)) - sp.Rational(5, 2), b_)[0]
    fc = fc.subs(b_, bs)
    # (3): f(w)=w (w>0) 의 서로 다른 양의 실근이 정확히 1개인 a>0
    good = []
    for k in range(1, 81):
        av = sp.Rational(k, 8)
        rr = {r for r in sp.real_roots(sp.Poly(sp.expand(fc.subs(a_, av) - x), x)) if r > 0}
        if len(rr) == 1:
            good.append(av)
    aval = good[0] if len(good) == 1 else None
    F = sp.expand(fc.subs(a_, aval)) if aval is not None else None
    # g=|f(x)-x| 의 미분불가능점: 부호가 바뀌는(홀수 중복도) 근
    pts = []
    if F is not None:
        d = sp.Poly(F - x, x)
        for r, m in sp.roots(d, x).items():
            if m % 2 == 1:
                pts.append(r)
    exp = sp.Rational(md_answer(2, "2"))
    sub(aval == 2 and pts == [exp], f"논제2 a={aval}, f={F}, p={pts}")
    flush(2)


if __name__ == "__main__":
    main(globals())
