#!/usr/bin/env python3
"""EWHA-G22 (이화여대 2022학년도 수시 논술 자연계열) — sympy 검산.
규칙: 정답 하드코딩 비교 금지. 문항 YAML 의 answer 를 읽어 설정으로부터 독립 재계산한 값과 비교.
증명 소문항은 수치 반례 탐색 [증명-수치확인]."""
import pathlib
import random
import re
import sys

import sympy as sp
import yaml

PREFIX = "EWHA-G22"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    x, p = sp.symbols("x p", real=True)
    f = x**3 - p * x**2 + p * x
    roots = sp.solve(sp.Eq(f, x), x)
    beta_md = sp.sympify(md_answer(1, "1"), locals={"p": p})
    # (1) beta 는 (0,1) 안의 근이고, 부호 조건을 여러 p 에서 수치 확인
    ok1 = any(sp.simplify(r - beta_md) == 0 for r in roots)
    rng = random.Random(1)
    for _ in range(50):
        pv = rng.uniform(1.001, 1.999)
        b = float(beta_md.subs(p, pv))
        g = sp.lambdify(x, (f - x).subs(p, pv))
        xs1 = [b * i / 200 for i in range(1, 200)]
        xs2 = [b + (1 - b) * i / 200 for i in range(1, 200)]
        ok1 &= all(g(t) > 0 for t in xs1) and all(g(t) < 0 for t in xs2)
    # (2)(3)(4) [증명-수치확인]: f' 판별식 < 0, 수열 200항 추적
    disc = sp.discriminant(sp.diff(f, x), x)
    ok2 = all(disc.subs(p, pv) < 0 for pv in [sp.Rational(k, 100) for k in range(101, 200)])
    ok3 = ok4 = True
    for _ in range(300):
        pv = rng.uniform(1.001, 1.999)
        b = pv - 1
        a = rng.uniform(1e-6, 1 - 1e-6)
        if abs(a - b) < 1e-9:
            continue
        side = a < b
        for _n in range(200):
            an = a ** 3 - pv * a ** 2 + pv * a
            ok2 &= 0 < an < 1
            ok3 &= (0 < an < b) if side else (b < an < 1)
            ok4 &= (an > a) if side else (an < a)
            if abs(an - a) < 1e-15:   # 고정점 근처 수렴 — 부동소수 한계
                break
            a = an
    report(1, ok1 and ok2 and ok3 and ok4, f"beta={beta_md} [증명-수치확인] (2)(3)(4) 300개 수열×200항")


def p02():
    x, k = sp.symbols("x k", real=True)
    quad = sp.expand(x**2 + 3 * (x + k)**2 - 3)
    D = sp.discriminant(quad, x)
    rng_k = sp.solve_univariate_inequality(D > 0, k, relational=False)
    md1 = md_answer(2, "1").replace(" ", "")
    ok1 = rng_k == sp.Interval.open(-2, 2) and md1 == "-2<k<2"
    # P: 기울기 1 접선의 접점, x>0
    X, Y = sp.symbols("X Y", real=True)
    sols = sp.solve([X**2 + 3 * Y**2 - 3, sp.Eq(-X / (3 * Y), 1)], [X, Y], dict=True)
    P = next((s[X], s[Y]) for s in sols if s[X] > 0)
    r1, r2 = sp.solve(quad, x)
    Q = (r1, r1 + k); R = (r2, r2 + k)
    area = sp.Rational(1, 2) * sp.Abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1]))
    fmd = sp.sympify(md_answer(2, "2"), locals={"k": k})
    ok2 = all(abs(float(area.subs(k, v) - fmd.subs(k, v))) < 1e-12 for v in (-1.9, -1, -0.3, 0.5, 1.2, 1.99))
    g = sp.expand(fmd**2)
    crit = [c for c in sp.solve(sp.diff(g, k), k) if -2 < c < 2]
    cand = [g.subs(k, c) for c in crit]
    mx = max(cand)
    # 경계 극한보다 큰지 확인
    ok3 = mx > sp.limit(g, k, 2, "-") and mx > sp.limit(g, k, -2, "+") and \
        sp.simplify(mx - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3, f"D>0:{rng_k} P={P} max={mx}")


def _dist_point_ray(px, py, sign):
    """점 (px,py)에서 반직선 y=(4/3)*sign*x+8 (sign*x>=0) 까지 거리."""
    import math
    dx, dy = sign * 3 / 5, 4 / 5            # 방향 단위벡터 (꼭짓점 (0,8)에서)
    t = max(0.0, (px - 0) * dx + (py - 8) * dy)
    qx, qy = t * dx, 8 + t * dy
    return math.hypot(px - qx, py - qy)


def _radius(s):
    """중심 (s,r) 반지름 r 인 원이 꺾인 그래프와 한 점에서 만나는 r (거리 = 반지름) 이분법."""
    lo, hi = 1e-9, 100.0
    for _ in range(200):
        r = (lo + hi) / 2
        d = min(_dist_point_ray(s, r, 1), _dist_point_ray(s, r, -1))
        if d > r:
            lo = r
        else:
            hi = r
    return (lo + hi) / 2


def p03():
    x, y = sp.symbols("x y", real=True)
    foot = sp.solve([sp.Eq(y, sp.Rational(4, 3) * x + 8), sp.Eq(y - 5, -sp.Rational(3, 4) * (x - 4))], [x, y], dict=True)[0]
    ok1 = sp.sympify(md_answer(3, "1")) == sp.Tuple(foot[x], foot[y])
    s = sp.symbols("s", real=True)
    f2 = sp.sympify(md_answer(3, "2"), locals={"s": s})
    f3 = sp.sympify(md_answer(3, "3"), locals={"s": s})
    ok2 = all(abs(_radius(v) - float(f2.subs(s, v))) < 1e-9 for v in (4, 4.5, 6, 10, 25.3))
    ok3 = all(abs(_radius(v) - float(f3.subs(s, v))) < 1e-9 for v in (-3.9, -2, -0.7, 0, 1.3, 3.99))
    report(3, ok1 and ok2 and ok3, f"foot=({foot[x]},{foot[y]}) f(6)~{_radius(6):.9f} f(1.3)~{_radius(1.3):.9f}")


if __name__ == "__main__":
    fns = sorted((n for n in dir() if re.fullmatch(r"p\d+", n)), key=lambda t: int(t[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
