#!/usr/bin/env python3
"""SGU-G22B (서강대 2022학년도 수시 논술 자연계열 2차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SGU-G22B"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (3-1) 1패 후: 3전2선승 → A 2승 필요/B 1승, 5전3선승 → A 3승/B 2승 (재귀로 경로 전수)
    p = sp.symbols("p")
    def wp(na, nb):
        memo = {}
        def go(a, b):
            if a == 0:
                return sp.Integer(1)
            if b == 0:
                return sp.Integer(0)
            if (a, b) not in memo:
                memo[(a, b)] = p * go(a - 1, b) + (1 - p) * go(a, b - 1)
            return memo[(a, b)]
        return sp.expand(go(na, nb))
    sols = [s for s in sp.solve(sp.Eq(wp(2, 1), wp(3, 2)), p) if 0 < s < 1]
    ok1 = sols == [sp.sympify(md_answer(1, "1"))]
    # (3-2) 무작위 a,b (0<2a<b): 연속 조건을 수치로 풀어 p>0 인 해가 유일하고 md 식과 일치
    a, b = sp.symbols("a b", positive=True)
    md2 = sp.sympify(md_answer(1, "2"), locals={"a": a, "b": b})
    random.seed(1)
    ok2 = True
    for _ in range(20):
        av = random.uniform(0.1, 3)
        bv = random.uniform(2 * av + 0.01, 2 * av + 10)
        h = lambda t: bv - av * mp.e ** (-2 * t) - av * mp.e ** (2 * t)
        roots = set()
        for t0 in [i / 10 for i in range(-50, 51)]:
            try:
                r = mp.findroot(h, t0)
                roots.add(round(float(r), 10))
            except Exception:  # noqa: BLE001
                pass
        pos = [r for r in roots if r > 0]
        ok2 &= len(pos) == 1 and abs(pos[0] - float(md2.subs({a: av, b: bv}))) < 1e-8
    # (3-3) f(x)=x e^{f(x)} 를 [0,1] 에서 풀어 수치 적분, g=xe^{-x} 증가 확인
    def fval(t):
        if t == 0:
            return mp.mpf(0)
        return mp.findroot(lambda y: y - t * mp.e ** y, (mp.mpf(0), mp.mpf(1)), solver="bisect")
    I3 = mp.quad(fval, [0, 1 / mp.e])
    X = sp.symbols("X")
    gp = sp.diff(X * sp.exp(-X), X)
    mono = all(gp.subs(X, sp.Rational(i, 100)) > 0 for i in range(0, 100))
    ok3 = mono and abs(I3 - mp.mpf(sp.N(sp.sympify(md_answer(1, "3")), 40))) < 1e-15
    # (3-4) 해 f = 24/(7x^4) + h, h(x)=2h(2x) 인 두 가지 h 로 조건을 맞춘 뒤 g 를 직접 극한
    xs, N = sp.symbols("x N", positive=True)
    vals = []
    for h0 in (1 / xs, (1 + sp.sin(2 * sp.pi * sp.log(xs, 2)) / 5) / xs):
        c = sp.symbols("c")
        f = sp.Rational(24, 7) / xs**4 + c * h0
        assert sp.simplify(f - 2 * f.subs(xs, 2 * xs) - 3 / xs**4) == 0
        cv = sp.solve(sp.Eq(sp.integrate(f, (xs, 1, 2)), 3), c)[0]
        f = f.subs(c, cv)
        gx = sp.limit(2**N * f.subs(xs, 2**N * xs), N, sp.oo) if h0 == 1 / xs else \
            sp.simplify(2**N * f.subs(xs, 2**N * xs)).subs(N, 60)   # 주기항은 N 정수에서 불변
        vals.append(sp.N(sp.Integral(gx, (xs, 1, 2)), 30))
    ok4 = all(abs(v - sp.sympify(md_answer(1, "4"))) < 1e-12 for v in vals)
    report(1, ok1 and ok2 and ok3 and ok4, f"p={sols} 3-2 random20 ok={ok2} int_f={mp.nstr(I3, 15)} int_g={[sp.N(v, 12) for v in vals]}")


def p02():
    # (4-1) [증명-수치확인] f = 이차 보간 + c(x-1)(x-2)(x-3)·(무작위 다항식) 들에서 a, b 존재
    xs = sp.symbols("x")
    base = sp.interpolate([(1, 3), (2, 3), (3, 5)], xs)
    random.seed(2)
    ok1 = True
    for _ in range(30):
        pert = sum(random.uniform(-3, 3) * xs**k for k in range(3))
        f = base + (xs - 1) * (xs - 2) * (xs - 3) * pert
        f1 = sp.lambdify(xs, sp.diff(f, xs))
        f2 = sp.lambdify(xs, sp.diff(f, xs, 2))
        grid = [1 + 2 * i / 4000 for i in range(1, 4000)]
        d1 = [f1(t) - 1.5 for t in grid]
        has_a = any(d1[i] * d1[i + 1] <= 0 for i in range(len(d1) - 1))
        has_b = max(f2(t) for t in grid) > 1
        ok1 &= has_a and has_b
    # (4-2) 주기 2, ∫_1^3 f = 1 인 연속함수 여러 개로 (1/x)∫_{-x}^x f 를 큰 x 에서 계산
    t = sp.symbols("t")
    md2 = float(sp.sympify(md_answer(2, "2")))
    ok2 = True
    for f in (sp.Rational(1, 2) + 3 * sp.sin(sp.pi * t), sp.Rational(1, 2) + sp.cos(sp.pi * t) ** 2 - sp.Rational(1, 2) + 2 * sp.sin(3 * sp.pi * t)):
        assert sp.simplify(f.subs(t, t + 2) - f) == 0 and sp.integrate(f, (t, 1, 3)) == 1
        F = sp.integrate(f, t)
        for X in (10**4 + sp.Rational(7, 10), 10**6 + sp.Rational(13, 10)):
            v = float((F.subs(t, X) - F.subs(t, -X)) / X)
            ok2 &= abs(v - md2) < 1e-3
    # (4-3) [증명-수치확인] f'<0 on (0, pi/2)
    fs = sp.sin(xs) / xs
    fp = sp.lambdify(xs, sp.diff(fs, xs), "mpmath")
    ok3 = all(fp(mp.pi / 2 * i / 2000) < 0 for i in range(1, 2000))
    # (4-4) k=1..6, x→0+ 수치 극한
    md4 = mp.mpf(sp.N(sp.sympify(md_answer(2, "4")), 40))
    ok4 = True
    for k in range(1, 7):
        for x0 in (mp.mpf("1e-4"), mp.mpf("1e-6")):
            v = mp.quad(lambda u: (mp.sin(u) / u) ** k / u, [x0, 3 * x0])
            ok4 &= abs(v - md4) < 1e-6
    report(2, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] 4-1 random30, 4-2 two periodic f, 4-3 grid, 4-4 k=1..6 -> {mp.nstr(md4, 12)}")


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
