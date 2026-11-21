#!/usr/bin/env python3
"""SKKU-G23A (성균관대 2023학년도 수시 논술우수전형 자연계 1교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import random
import re
import sys

import sympy as sp

PREFIX = "SKKU-G23A"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    p, X = sp.symbols("p X", real=True)
    # (i) OP 기울기 -1/2 → P, 접선 L, 원과의 교점 E,F 직접 계산
    ps = [r for r in sp.solve(sp.Eq((p + 1)**2 / p, sp.Rational(-1, 2)), p) if -1 < r < 0]
    p0 = ps[0]
    m = 2 * (p0 + 1)
    Ly = m * (X - p0) + (p0 + 1)**2
    xs = sp.solve(X**2 + Ly**2 - 1, X)
    E = (xs[0], Ly.subs(X, xs[0])); F = (xs[1], Ly.subs(X, xs[1]))
    ef2 = sp.nsimplify(sp.simplify((E[0] - F[0])**2 + (E[1] - F[1])**2))
    ok1 = len(ps) == 1 and sp.simplify(ef2 - sp.Rational(md_answer(1, "1"))) == 0
    # (ii) OP 최소점(수치 탐색)에서 기울기곱 = -1 [증명-수치확인]
    grid = [-1 + i / 100000 for i in range(1, 100000)]
    D = lambda q: q * q + (q + 1)**4
    pm = min(grid, key=D)
    pm = float(sp.nsolve(2 * p + 4 * (p + 1)**3, p, pm))
    prod = ((pm + 1)**2 / pm) * 2 * (pm + 1)
    ok2 = abs(prod + 1) < 1e-9 and D(pm) <= min(D(q) for q in grid) + 1e-12
    # (iii) AP*PB 를 좌표로 직접 계산해 최대점이 (ii)의 최소점과 같은지 [증명-수치확인]
    def ab(q):
        y0 = (q + 1)**2
        k = math.sqrt(1 - y0 * y0)
        return (q + k) * (k - q)
    qa = max(grid, key=ab)
    ok3 = abs(qa - pm) < 2e-5 and all(abs(ab(q) - (1 - D(q))) < 1e-12 for q in random.Random(1).sample(grid, 50))
    ok3 = ok3 and "1 - OP^2" in md_answer(1, "3")
    report(1, ok1 and ok2 and ok3, f"EF^2={ef2} p*={pm:.8f} slope_prod={prod:.12f} argmax(AP*PB)={qa:.5f}")


def p02():
    # (i) 사인법칙: sinA = sqrt3 a / (2a), A<90°
    A = sp.symbols("A", real=True)
    sols = [s for s in sp.solveset(sp.sin(A) - sp.sqrt(3) / 2, A, sp.Interval.open(0, sp.pi / 2))]
    ok1 = len(sols) == 1 and sp.simplify(sols[0] - sp.sympify(md_answer(2, "1"))) == 0
    Aval = sols[0]
    cosA = sp.cos(Aval)
    # (ii) a=10: 정수 b<=c 전수 (BC^2 = b^2+c^2-2bc cosA = 300)
    found = {(c, b) for b in range(1, 101) for c in range(b, 101) if sp.Integer(b * b + c * c) - 2 * b * c * cosA == 300}
    import ast
    ok2 = found == ast.literal_eval(md_answer(2, "2"))
    # (iii) a=sqrt10: b+c=M 인 0<b<=c 실수해 존재 여부를 판별식으로 M=1..2023 전수
    Ms = []
    for M in range(1, 2024):
        prod = sp.Rational(M * M - 30, 3)          # (b+c)^2 - 3bc = 30
        disc = M * M - 4 * prod
        if prod > 0 and disc >= 0:
            Ms.append(M)
    ok3 = sum(Ms) == int(md_answer(2, "3"))
    # (iv) a=sqrt2: b^2, c^2 = X 의 근 (X^2-(6+N)X+N^2), 실수·비정수 조건, N=1..2023
    Ns = []
    for N in range(1, 2024):
        Xs = sp.Symbol("X")
        roots = sp.solve(Xs**2 - (6 + N) * Xs + N * N, Xs)
        if all(r.is_real and r > 0 for r in roots) and not any(r.is_integer for r in roots):
            # 직접 확인: b=sqrt(r1), c=sqrt(r2) 가 b^2-bc+c^2=6 을 만족
            b, c = sorted(sp.sqrt(r) for r in roots)
            if abs(float(b * b - b * c + c * c - 6)) < 1e-9 and abs(float(b * c - N)) < 1e-9:
                Ns.append(N)
    ok4 = sum(Ns) == int(md_answer(2, "4"))
    report(2, ok1 and ok2 and ok3 and ok4, f"A={Aval} (AB,AC)={sorted(found)} M={Ms} N={Ns}")


def p03():
    t, n = sp.symbols("t n", positive=True)
    # (i) 정사각형을 기하로 직접 쌓아 변의 길이 계산: B 원점, BC 를 x축, 빗변 y = t x
    def sides(tv, k):
        xR = 1 / tv + 1                      # R1 의 오른쪽 변 = C 의 x좌표 (R1 왼쪽 위 꼭짓점 (1/t,1))
        right = xR
        out = []
        for _ in range(k):
            # 한 변 s, 왼쪽 위 꼭짓점 (right - s, s) 가 빗변 위: s = t (right - s)
            s = tv * right / (1 + tv)
            out.append(s)
            right -= s
        return out
    expr = sp.sympify(md_answer(3, "1"), locals={"t": t, "n": n})
    ok1 = all(abs(sides(tv, 8)[k - 1]**2 - float(expr.subs({t: tv, n: k}))) < 1e-12 for tv in (0.3, 1.0, math.sqrt(3), 2.7) for k in range(1, 9))
    # (ii) theta = pi/3 에서 정확한 등비 합으로 직접 계산
    tv = sp.sqrt(3)
    q = 1 / (1 + tv)**2
    s = lambda k: q**(k - 1)
    def PQ(N):
        P = sum(q**(2 * j) for j in range((N + 1) // 2))          # 홀수 번째 (1,3,...)
        Q = sum(q**(2 * j + 1) for j in range(N // 2))            # 짝수 번째
        return P, Q
    # 2023 항 합을 닫힌 식(등비합)으로
    def PQc(N):
        no, ne = (N + 1) // 2, N // 2
        P = (1 - q**(2 * no)) / (1 - q**2)
        Q = q * (1 - q**(2 * ne)) / (1 - q**2)
        return P, Q
    P23, Q23 = PQc(2023); P21, Q21 = PQc(2021)
    val = P23 / Q23 + Q21 / P21
    ok2 = abs(sp.N(val - sp.sympify(md_answer(3, "2")), 50)) < sp.Float("1e-40")
    # (iii) 평균 비교: 여러 θ 에서 고정밀 수치 확인 + 귀납 단계 차이식 확인 [증명-수치확인]
    import mpmath as mp
    mp.mp.dps = 60
    ok3 = ">" in md_answer(3, "3")
    for tvv in (0.05, 0.5, 1, mp.sqrt(3), 5, 40):
        qq = 1 / (1 + mp.mpf(tvv))**2
        P = (1 - qq**2024) / (1 - qq**2); Q = qq * (1 - qq**2022) / (1 - qq**2)
        ok3 = ok3 and P / 1012 > Q / 1011
        for nn in range(1, 60):
            An = lambda m_: m_ * sum(qq**(2 * j) for j in range(m_ + 1)) - (m_ + 1) * sum(qq**(2 * j - 1) for j in range(1, m_ + 1))
            ok3 = ok3 and An(nn) > 0
    report(3, ok1 and ok2 and ok3, f"s_n check ok={ok1} value={sp.N(val, 15)} mean_odd>mean_even={ok3}")


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
