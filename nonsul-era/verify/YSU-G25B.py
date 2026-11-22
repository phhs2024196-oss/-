#!/usr/bin/env python3
"""YSU-G25B (연세대 2025학년도 수시 논술전형(추가시험) 자연계열 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import random
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "YSU-G25B"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    x = sp.symbols("x", positive=True)
    m, n = sp.symbols("m n")
    pairs = [(a, b) for a in range(1, 31) for b in range(1, 31) if a != b]
    # (1-1) X 를 적분으로 직접 계산 후 870 개 순서쌍 평균
    Xexpr = sp.integrate(m - n * sp.log(x)**2, (x, 1, sp.E))
    EX = sp.nsimplify(sp.simplify(sum(Xexpr.subs({m: a, n: b}) for a, b in pairs) / len(pairs)))
    ok1 = sp.simplify(EX - sp.sympify(md_answer(1, "1"))) == 0
    # (1-2) N=30!/(mn) 마다 조건 (Ⅰ)(Ⅱ)(Ⅲ) 를 만족하는 (x,y,z) 를 직접 센다(소인수 거듭제곱 배정 전수)
    cache = {}
    total = Fraction(0)
    for a, b in pairs:
        N = math.factorial(30) // (a * b)
        fac = tuple(sorted(sp.factorint(N).items()))
        if fac not in cache:
            pw = [p**e for p, e in fac]
            cnt = 0
            seen = set()
            for asg in itertools.product(range(3), repeat=len(pw)):
                v = [1, 1, 1]
                for q, s in zip(pw, asg):
                    v[s] *= q
                xx, yy, zz = v
                if xx > yy > zz and math.gcd(xx, yy) == 1 and math.gcd(yy, zz) == 1 and math.gcd(zz, xx) == 1 \
                        and xx * yy * zz == N:
                    seen.add((xx, yy, zz))
            cnt = len(seen)
            cache[fac] = cnt
        Y = cache[fac]
        total += Fraction(2 * Y + 1, 3**7)
    E2 = total / len(pairs)
    ok2 = sp.Rational(E2.numerator, E2.denominator) == sp.sympify(md_answer(1, "2"))
    report(1, ok1 and ok2, f"E(X)={EX} E((2Y+1)/3^7)={E2} (distinct factorizations {len(cache)})")


def tangent_XY(f, al):
    """곡선 y=f(x) 위 P(al,f(al)) 의 접선에서 A,B 를 구하고 OQ=OA+OB-OP 로 X,Y 를 계산."""
    xx = sp.symbols("xx")
    fp = sp.diff(f, xx).subs(xx, al)
    fa = f.subs(xx, al)
    line = fp * (xx - al) + fa
    Ax = sp.solve(sp.Eq(line, 0), xx)[0]
    By = line.subs(xx, 0)
    X = Ax + 0 - al
    Y = 0 + By - fa
    return sp.simplify(X), sp.simplify(Y), xx


def p02():
    a, b, al = sp.symbols("a b alpha", positive=True)
    xx = sp.symbols("xx")
    f = sp.sqrt(b**2 * (1 - xx**2 / a**2))
    X, Y, _ = tangent_XY(f, al)
    XY = sp.simplify(X * Y)
    crit = [r for r in sp.solve(sp.diff(XY, al), al) if r.is_positive]
    mx = sp.simplify(XY.subs(al, crit[0]))
    # 수치로 구간 (0,a) 최대 확인
    okn = True
    for av, bv in [(1, 1), (2, 3), (5, 0.7)]:
        g = sp.lambdify(al, XY.subs({a: av, b: bv}))
        grid = max(g(av * i / 2000) for i in range(1, 2000))
        okn &= abs(grid - float(mx.subs({a: av, b: bv}))) < 1e-5
    ok1 = sp.simplify(mx - sp.sympify(md_answer(2, "1"), locals={"a": a, "b": b})) == 0 and okn
    # (2-2) 일반 f 에 대해 XY 를 구하고 XY=c 를 미분방정식으로 풂
    F = sp.Function("F")
    t = sp.symbols("t", positive=True)
    X2, Y2, _ = tangent_XY(F(xx), t)
    XYg = sp.simplify(X2 * Y2)               # = t F(t)
    c = sp.symbols("c")
    sol = sp.solve(sp.Eq(XYg.subs(t, 1).subs(F(1), 3), c), c)[0]
    fsol = sp.solve(sp.Eq(XYg, sol), F(t))[0]  # F(t) = c/t
    h = fsol**2 + 1 / fsol
    cands = [r for r in sp.solve(sp.diff(h, t), t) if r.is_positive]
    k = cands[0]
    okmin = all(h.subs(t, k) <= h.subs(t, v) for v in (0.5, 2, 3, 3.7, 3.8, 5, 10))
    ok2 = sp.simplify(k - sp.sympify(md_answer(2, "2"))) == 0 and okmin
    report(2, ok1 and ok2, f"XY={XY} max={mx} | XY_general={XYg} f={fsol} k={sp.nsimplify(k)}")


def build(fv, gv, fpv, gpv, a, b):
    """조건 (Ⅰ)(Ⅱ)(Ⅲ) 를 수치로 직접 만족시키는 Q,R,S 를 구성(후보 중 조건 확인)."""
    import numpy as np
    P = np.array([fv, gv]); Pp = np.array([fpv, gpv])
    # (Ⅰ) Q = kP, OP·PQ<0, |PQ|=a|OP|
    Qc = [P * (1 + s * a) for s in (1, -1)]
    Q = [q for q in Qc if P @ (q - P) < 0 and abs(np.linalg.norm(q - P) - a * np.linalg.norm(P)) <= 1e-9 * a * np.linalg.norm(P)][0]
    PQ = Q - P
    perp = np.array([Pp[1], -Pp[0]]); perp = perp / np.linalg.norm(perp)
    Rc = [P + s * np.linalg.norm(PQ) * perp for s in (1, -1)]
    R = [r for r in Rc if (r - P) @ PQ > 0][0]
    PR = R - P
    Sc = [P + s * b * PR for s in (1, -1)]
    S = [s_ for s_ in Sc if PR @ (s_ - P) < 0][0]
    return P, Q, R, S


def p03():
    import numpy as np
    import mpmath as mp
    mp.mp.dps = 30
    # (3-1)
    def integrand(tv):
        P, Q, R, S = build(tv, 4 - 2 * tv, 1, -2, 2, 5)
        return float(P @ (S - P))**2
    I = float(mp.quad(lambda v: integrand(float(v)), [0, 1, 2]))
    ok1 = abs(I - float(sp.sympify(md_answer(3, "1")))) < 1e-6
    # (3-2)
    T = np.array([0.0, 1.0])
    def val(tv):
        P, Q, R, S = build(2 * np.cos(tv), np.sin(tv), -2 * np.sin(tv), np.cos(tv), 5, 1)
        return float(T @ (R - P))**2
    grid = np.linspace(0, np.pi / 2, 20001)
    i0 = int(np.argmax([val(v) for v in grid]))
    lo, hi = grid[max(i0 - 1, 0)], grid[min(i0 + 1, 20000)]
    for _ in range(200):                      # 황금분할 탐색(단봉 구간)
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if val(m1) < val(m2):
            lo = m1
        else:
            hi = m2
    mx = val((lo + hi) / 2)
    ok2 = abs(mx - float(sp.sympify(md_answer(3, "2")))) < 1e-8
    # (3-3) [증명-수치확인] f>a, f'>0, g=1/f 인 여러 함수와 a 에서 OP·OR>0 반례 탐색
    random.seed(1)
    worst = float("inf")
    fams = [
        lambda tv, a: (a + np.exp(tv), np.exp(tv)),
        lambda tv, a: (a + 1e-6 + 1 / (1 + np.exp(-tv)), np.exp(-tv) / (1 + np.exp(-tv))**2),
        lambda tv, a: (a + 3 + np.arctan(tv), 1 / (1 + tv**2)),
    ]
    ok3 = md_answer(3, "3") is None
    for _ in range(3000):
        a = 2 + random.random() * 6
        tv = random.uniform(-30, 30)
        fam = random.choice(fams)
        fv, fpv = fam(tv, a)
        if fpv <= 1e-12:
            continue
        gv, gpv = 1 / fv, -fpv / fv**2
        P, Q, R, S = build(fv, gv, fpv, gpv, a, 1)
        d = float(P @ R)
        closed = fv**2 + 1 / fv**2 - 2 * a
        worst = min(worst, d / (a * a - 2 * a))
        ok3 &= d > 0 and abs(d - closed) < 1e-6 * max(1, abs(closed))
    report(3, ok1 and ok2 and ok3,
           f"int={I:.9f} max={mx:.12f} [증명-수치확인] OP·OR>0 및 =f^2+1/f^2-2a, min ratio to a^2-2a={worst:.4f}")


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
