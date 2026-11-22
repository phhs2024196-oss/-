#!/usr/bin/env python3
"""YSU-G25 (연세대 2025학년도 수시 논술 자연계열 수학, 본시험 2024.10) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "YSU-G25"
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
    # 주머니: 1,3,5 각 1개 / 2,4,6 각 2개 (복원). 격자 가로 4, 세로 2. 경계 도달 시 종료.
    balls = {1: 1, 3: 1, 5: 1, 2: 2, 4: 2, 6: 2}
    tot = sum(balls.values())
    north_total = Fraction(0)
    north_by_step = {}

    def walk(x, y, step, pr):
        nonlocal north_total
        for b, c in balls.items():
            q = pr * Fraction(c, tot)
            nx, ny = (x, y + b) if b % 2 else (x + b, y)
            if ny >= 2:                       # 북쪽 경계
                north_total += q
                north_by_step[step] = north_by_step.get(step, 0) + q
            elif nx >= 4:                     # 동쪽 경계
                pass
            else:
                walk(nx, ny, step + 1, q)

    walk(0, 0, 1, Fraction(1))
    ans = north_by_step[2] / north_total
    ok = ans == Fraction(md_answer(1, "1"))
    report(1, ok, f"P(north)={north_total} P(north,2nd)={north_by_step[2]} cond={ans}")


def p02():
    t, x, k = sp.symbols("t x k", positive=True)
    g = (t**2025 + t**2024 + 1) / (t**2025 + t**1885 + 1)
    # f'(x) 의 무한대 극한 (분자·분모를 t^2025 로 나눔)
    lim_fp = sp.limit((1 + 1 / t + t**-2025) / (1 + t**-140 + t**-2025), t, sp.oo)
    exact = sum(2 * k_ * lim_fp / ((k_ - 1) * k_ * (k_ + 1)) for k_ in range(2, 11))
    # 수치: 큰 n 에서 실제 적분 차를 고정밀 계산 (피적분함수 = 1 + (1/t)(1 - t^-139... ) 형태로 안정화)
    import mpmath as mp
    mp.mp.dps = 30
    h = lambda s: (1 + 1 / s + s**-2025) / (1 + s**-140 + s**-2025)
    def S(n):
        return sum((mp.quad(h, [n, n + 2 * kk])) / ((kk - 1) * kk * (kk + 1)) for kk in range(2, 11))
    s1, s2 = S(10**6), S(10**8)
    md = sp.Rational(md_answer(2, "1"))
    ok = sp.simplify(exact - md) == 0 and abs(s2 - float(md)) < abs(s1 - float(md)) + 1e-15 and abs(s2 - float(md)) < 1e-6
    report(2, ok, f"lim f'={lim_fp} sum={exact} S(1e6)={mp.nstr(s1, 12)} S(1e8)={mp.nstr(s2, 12)}")


def p03():
    def f(n):
        return 2025 // math.gcd(n, 2025)            # nk 가 2025 의 배수가 되는 최소 k
    # 정의 확인: 직접 최소 k 탐색과 일치
    assert all(f(n) == next(k for k in range(1, 2026) if n * k % 2025 == 0) for n in range(1, 300))
    c1 = sum(1 for n in range(1, 2026) if f(n) == f(n * n))
    c2 = sum(1 for n in range(1, 2026) if f(f(n)) < n)
    ok = c1 == int(md_answer(3, "1")) and c2 == int(md_answer(3, "2"))
    report(3, ok, f"3-1 count={c1} 3-2 count={c2}")


def p04():
    n = sp.symbols("n", positive=True)
    al = sp.symbols("alpha", positive=True)
    a_n = sp.log((n + 1) / n) - 1 / (n + 1)
    l1 = sp.limit(n * (3 * n + 1)**3 * a_n**2, n, sp.oo)
    ok1 = sp.simplify(l1 - sp.Rational(md_answer(4, "1"))) == 0
    md2 = sp.sympify(md_answer(4, "2"), locals={"alpha": al})
    ok2 = True
    msgs = []
    for A in range(2, 8):
        b_n = sp.Rational(1, 1 - A) * (1 / (n + 1)**(A - 1) - 1 / n**(A - 1)) - 1 / (n + 1)**A
        # n^p b_n 이 0 이 아닌 유한 극한을 갖는 p 탐색
        found = None
        for p in range(0, 3 * A + 3):
            L = sp.limit(n**p * b_n, n, sp.oo)
            if L.is_finite and L != 0:
                found = (p, L)
                break
        if found is None or sp.simplify(found[0] + found[1] - md2.subs(al, A)) != 0:
            ok2 = False
        msgs.append(f"a={A}:p={found[0] if found else None},L={found[1] if found else None}")
    report(4, ok1 and ok2, f"4-1 lim={l1}; 4-2 " + " ".join(msgs))


def p05():
    X, k, m = sp.symbols("X k m", positive=True)
    xn = sp.symbols("x_n", positive=True)
    # 접선의 x절편
    tan_root = sp.solve(sp.Eq(0, xn**2 + 2 * xn * (X - xn)), X)[0]
    r = sp.simplify(tan_root / xn)                      # 공비
    a_k = sp.summation(k * r**(sp.Symbol("j", integer=True, nonnegative=True)), (sp.Symbol("j", integer=True, nonnegative=True), 0, sp.oo))
    a_k = sp.simplify(k / (1 - r))
    ok1 = sp.simplify(a_k.subs(k, 2025) - sp.sympify(md_answer(5, "1"))) == 0
    # 법선의 x절편
    a = a_k
    b_k = sp.solve(sp.Eq(0, -(X - a) / (2 * a) + a**2), X)[0]
    D = sp.expand(b_k.subs(k, m) - a_k.subs(k, m))
    # 곱의 로그를 수치로 큰 m 에서 계산 + 구분구적 정적분 기호 계산
    import mpmath as mp
    mp.mp.dps = 30
    def logc(M):
        Dm = D.subs(m, M)
        return mp.fsum(mp.mpf(int((a.subs(k, kk))**2)) / M**3 * mp.log(mp.mpf(int(a.subs(k, kk)**3)) / int(Dm) + 1) for kk in range(1, M + 1))
    xs = sp.symbols("xs", positive=True)
    I = sp.integrate(4 * xs**2 * sp.log(xs**3 / 2 + 1), (xs, 0, 1))
    md = sp.sympify(md_answer(5, "2"))
    v1, v2 = logc(2000), logc(20000)
    ok2 = abs(sp.N(sp.exp(I) - md, 30)) < 1e-20 and abs(mp.e**v2 - float(md)) < 1e-3 and abs(mp.e**v2 - float(md)) < abs(mp.e**v1 - float(md))
    report(5, ok1 and ok2, f"r={r} a_k={a_k} b_k={sp.expand(b_k)} b_m-a_m={D} I={sp.simplify(I)} c_20000~{mp.nstr(mp.e**v2, 10)}")


def p06():
    import numpy as np
    A = np.array([-2.0, 0.0])
    def Q(P):
        v = P - A
        return A + 9 * v / np.dot(v, v)                 # 같은 방향, AP·AQ=9
    # 6-1: C1 위 점들의 상이 md 의 도형식을 만족하는지
    x, y = sp.symbols("x y", real=True)
    lhs, rhs = md_answer(6, "1").split("=")
    F = sp.lambdify((x, y), sp.sympify(lhs) - sp.sympify(rhs))
    th = np.linspace(0, 2 * np.pi, 2001)
    qs = np.array([Q(np.array([np.cos(t), np.sin(t)])) for t in th])
    ok1 = max(abs(F(*q)) for q in qs) < 1e-9
    # 상이 원 전체를 덮는지(각도 범위)
    ang = np.sort(np.arctan2(qs[:, 1], qs[:, 0] - 4))
    ok1 = ok1 and np.max(np.diff(ang)) < 0.05
    # 6-2: 도형 R 의 상의 길이(폴리라인)
    N = 200001
    t1 = np.linspace(2 * np.pi / 3, 4 * np.pi / 3, N)        # C1 의 호 R1 O' R2
    t2 = np.linspace(-np.pi / 3, np.pi / 3, N)               # C2 의 호 R1 O R2
    L = 0.0
    for P in (np.stack([np.cos(t1), np.sin(t1)], 1), np.stack([np.cos(t2) - 1, np.sin(t2)], 1)):
        qq = np.array([Q(p) for p in P])
        L += np.sum(np.linalg.norm(np.diff(qq, axis=0), axis=1))
    md = float(sp.sympify(md_answer(6, "2")))
    ok2 = abs(L - md) < 1e-6
    report(6, ok1 and ok2, f"6-1 max|F|<1e-9 coverage ok={ok1}; 6-2 length~{L:.9f}")


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
