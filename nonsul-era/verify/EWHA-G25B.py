#!/usr/bin/env python3
"""EWHA-G25B (이화여대 2025학년도 수시 논술 자연계열Ⅱ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p01 은 공식 예시 답안의 1<a<3 교점이 무연근임을 원래 거리 조건으로 직접 확인한다."""
PREFIX = "EWHA-G25B"
import re
import sys

import mpmath as mp
import sympy as sp

WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")
x, y = sp.symbols("x y", real=True)


def p01():
    # (1) md 의 타원 위 점들이 b+c=4 를 만족하는지 + 역으로 b+c=4 인 점이 타원 위인지
    lhs, rhs = md_answer(1, "1").split("=")
    E1 = sp.sympify(lhs, locals={"x": x, "y": y}) - sp.sympify(rhs, locals={"x": x, "y": y})
    th = [2 * mp.pi * i / 400 for i in range(400)]
    pts = [(mp.sqrt(3) * mp.cos(u), 4 + 2 * mp.sin(u)) for u in th]
    ok1 = all(abs(float(E1.subs({x: float(px), y: float(py)}))) < 1e-12 and
              abs(mp.sqrt(px**2 + (py - 3)**2) + mp.sqrt(px**2 + (py - 5)**2) - 4) < 1e-12 for px, py in pts)
    # (2) 원래 조건 sqrt(x^2+(y-3)^2)+y=a, y>=0 를 직접 풀어 md 기술과 비교
    def T_points(a):
        out = []
        for yv in [i * 0.01 for i in range(0, 1301)]:
            r = a - yv
            if r < 0:
                continue
            x2 = r**2 - (yv - 3)**2
            if x2 >= -1e-12:
                out.append((mp.sqrt(max(x2, 0)), yv))
        return out
    ans2 = md_answer(1, "2")
    par = sp.sympify(ans2.split("a>3:")[1].split(" (y>=0)")[0].replace("y=", ""), locals={"x": x, "a": sp.Symbol("a")})
    ok2 = len(T_points(2.0)) == 0 and len(T_points(2.9)) == 0 and "0<a<3: empty" in ans2
    ok2 &= all(abs(px) < 1e-9 and 0 <= py <= 3 for px, py in T_points(3.0))
    for av in (4.0, 6.5, 10.0):
        ok2 &= all(abs(float(par.subs({x: px, sp.Symbol("a"): av})) - py) < 1e-9 for px, py in T_points(av))
    # (3) 원래 거리 조건으로 타원 S 위 교점 개수를 a 격자에서 직접 셈
    def count(a):
        g = lambda u: mp.sqrt(3 * mp.cos(u)**2 + (1 + 2 * mp.sin(u))**2) + 4 + 2 * mp.sin(u) - a
        N = 4000
        vals = [g(2 * mp.pi * i / N) for i in range(N + 1)]
        c = sum(1 for i in range(N) if vals[i] == 0 or vals[i] * vals[i + 1] < 0)
        return c
    two = [av for av in [0.25 * k for k in range(1, 48)] if count(av) == 2]
    lo, hi = min(two), max(two)
    rng, ptxt = md_answer(1, "3").split(";")
    ok3 = rng.strip() == "3<a<9" and 3 < lo and hi < 9 and abs(lo - 3.25) < 1e-9 and abs(hi - 8.75) < 1e-9
    # 공식 답의 1<a<3 교점이 무연근인지: a=2 에서 (±1,4) 대입
    bad = mp.sqrt(1 + 1) + 4
    # md 교점 공식: a=6 에서 실제로 S, T 위인지
    a6 = 6.0
    X = mp.sqrt((a6 - 3) * (9 - a6) / 3); Y = 2 * a6 / 3
    ok3 &= abs(X**2 / 3 + (Y - 4)**2 / 4 - 1) < 1e-12 and abs(mp.sqrt(X**2 + (Y - 3)**2) + Y - a6) < 1e-12
    ok3 &= "sqrt((a-3)*(9-a)/3)" in ptxt.replace(" ", "") and "2*a/3" in ptxt
    report(1, ok1 and ok2 and ok3, f"S 400점·T(a=2,2.9 공집합)·교점 2개인 a 격자 [{lo},{hi}] (공식 1<a<3 무연근: a=2,(1,4)에서 d+e={float(bad):.3f}≠2)")


def p02():
    def f(xv):
        n = 1
        while not (2**(n - 1) - 1 <= xv < 2**n - 1) and n < 60:
            n += 1
        return mp.sqrt(2**(n - 1) * (xv + 1) - 2**(2 * n - 2)) + 2**(n - 1) - 1
    # (1) [증명-수치확인] 격자 증가 + 이음점 연속
    grid = [i * 0.013 for i in range(0, 20000)]
    vals = [f(v) for v in grid]
    ok1 = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    ok1 &= all(abs(f(2**n - 1 - 1e-12) - (2**n - 1)) < 1e-4 for n in range(1, 9))
    nn = sp.symbols("n", integer=True, positive=True)
    e2 = sp.sympify(md_answer(2, "2"), locals={"n": nn})
    e3 = sp.sympify(md_answer(2, "3"), locals={"n": nn})
    ok2 = ok3 = True
    for n in range(1, 8):
        an, an1 = 2**(n - 1) - 1, 2**n - 1
        A = mp.quad(f, [an, an1 - 1e-15]) if n > 1 else mp.quad(f, [an, an1 - 1e-15])
        ok2 &= abs(A - float(e2.subs(nn, n))) < 1e-8 * max(1, abs(A))
        # 역함수 g: 이분법으로 직접 구해 수치적분
        def g(yv, lo=an, hi=an1):             # 이분법으로 역함수 값을 직접 구함
            lo, hi = mp.mpf(lo), mp.mpf(hi)
            for _ in range(80):
                mid = (lo + hi) / 2
                if f(mid) < yv:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        B = mp.quad(g, [an, an1])
        ok3 &= abs(B - float(e3.subs(nn, n))) < 1e-7 * max(1, abs(B))
    report(2, ok1 and ok2 and ok3, "증가 20000점 [증명-수치확인], n=1..7 에서 ∫f 와 역함수 이분법 ∫g 일치")


def p03():
    X = sp.symbols("X")
    a = [None] + [sp.integrate(sp.cos(X)**(k - 1), (X, 0, sp.pi / 2)) for k in range(1, 16)]
    ok1 = all(sp.simplify(a[k + 2] - sp.Rational(k, k + 1) * a[k]) == 0 for k in range(1, 14))  # [증명-수치확인]
    # (2) 점화식으로 큰 n 까지 a_n 을 만들어 b_n, c_n 의 추이 확인
    mp.mp.dps = 30
    A = {1: mp.pi / 2, 2: mp.mpf(1)}
    for k in range(1, 400001):
        A[k + 2] = mp.mpf(k) / (k + 1) * A[k]
    n = 200000
    b = A[2 * n + 1] / A[2 * n]; c = A[2 * n - 1] / A[2 * n]
    lb, lc = [sp.sympify(s.split("=")[1]) for s in md_answer(3, "2").split(";")]
    ok2 = abs(b - float(lb)) < 1e-5 and abs(c - float(lc)) < 1e-5 and b <= 1 <= c
    # (3) 곱을 직접 계산
    m = 200000
    lp = mp.fsum(2 * (mp.log(2 * k - 1) - mp.log(2 * k)) for k in range(1, m + 1))
    val = mp.e**lp * m
    ok3 = abs(val - float(sp.sympify(md_answer(3, "3")))) < 1e-5
    report(3, ok1 and ok2 and ok3, f"점화식 k=1..13 [증명-수치확인], b_n={float(b):.8f}, c_n={float(c):.8f}, 곱×n={float(val):.8f}")


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
