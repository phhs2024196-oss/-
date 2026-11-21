#!/usr/bin/env python3
"""KHU-G21E (경희대 2021학년도 논술 의학계 12.07 오후, 수학 논제 I-1·I-2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-G21E"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[논제 I-1] 마름모 |x|+|y|<1 내부에서 OP = √2·(가장 가까운 변까지 거리)인 곡선이 둘러싼 R."""
    # 몬테카를로 대신 격자 수치로 R 의 넓이를 직접 근사(조건 OP ≤ √2·d 인 영역 = 곡선 안쪽)
    N = 1200
    cnt = 0
    vol = 0.0
    h = 2.0 / N
    for i in range(N):
        xv = -1 + (i + 0.5) * h
        # 이 x 에서 단면 길이(수치): y 를 이분 탐색 없이 격자로
        ys = 0
        for j in range(N):
            yv = -1 + (j + 0.5) * h
            if abs(xv) + abs(yv) >= 1: continue
            d = min(abs(sx * xv + sy * yv - 1) / math.sqrt(2) for sx in (1, -1) for sy in (1, -1))
            if math.hypot(xv, yv) <= math.sqrt(2) * d: ys += 1
        cnt += ys
        L = ys * h
        vol += math.pi / 2 * (L / 2) ** 2 * h       # 반원 단면: 지름 L
    area_num = cnt * h * h
    x = sp.symbols("x", real=True)
    # 1사분면 곡선: √(x²+y²)=|x+y-1| ⇒ 정리해 y 를 x 로 (sympy)
    yy = sp.symbols("y", real=True)
    ycurve = sp.solve(sp.Eq(x**2 + yy**2, (x + yy - 1) ** 2), yy)[0]
    xint = [s for s in sp.solve(ycurve, x) if s >= 0][0]
    u = sp.symbols("u", positive=True)          # u = 1-x > 0 (로그의 실수 가지 유지)
    yu = sp.simplify(ycurve.subs(x, 1 - u))
    area = 4 * sp.integrate(yu, (u, 1 - xint, 1))
    V = 2 * sp.integrate(sp.pi / 2 * yu**2, (u, 1 - xint, 1))
    ok = sp.simplify(area - 2 * (1 - sp.log(2))) == 0 and sp.simplify(V - sp.pi * (sp.Rational(3, 4) - sp.log(2))) == 0 \
        and abs(area_num - float(area)) < 5e-3 and abs(vol - float(V)) < 5e-3
    report(1, ok, f"y={sp.simplify(ycurve)}, 넓이={sp.simplify(area)}(격자 {area_num:.4f}), 부피={sp.simplify(V)}(격자 {vol:.4f})")


def p02():
    """[논제 I-2] a_n = ∫_0^k x^n (x-k)^n / n! e^x dx, 0≤k≤2."""
    x, k = sp.symbols("x k", real=True)
    a = {n: sp.expand(sp.integrate(x**n * (x - k) ** n / sp.factorial(n) * sp.exp(x), (x, 0, k))) for n in range(1, 5)}
    ex = {
        1: (2 - k) * sp.exp(k) - (k + 2),
        2: (k**2 - 6 * k + 12) * sp.exp(k) - (k**2 + 6 * k + 12),
        3: (-k**3 + 12 * k**2 - 60 * k + 120) * sp.exp(k) - (k**3 + 12 * k**2 + 60 * k + 120),
        4: (k**4 - 20 * k**3 + 180 * k**2 - 840 * k + 1680) * sp.exp(k) - (k**4 + 20 * k**3 + 180 * k**2 + 840 * k + 1680),
    }
    ok2 = all(sp.simplify(a[n] - ex[n]) == 0 for n in range(1, 5))
    # 점화식 a_n=(4n-2)a_{n-1}+k^2 a_{n-2} 확인(n=3,4)
    okrec = all(sp.simplify(a[n] - ((4 * n - 2) * a[n - 1] + k**2 * a[n - 2])) == 0 for n in (3, 4))
    # (1) |a_n| ≤ (e^k-1)/n! → 0 : k=2 에서 수치 확인 + 상한 성립
    ok1 = True
    for kv in [0.5, 1.0, 2.0]:
        for n in (5, 10, 20, 40):
            val = abs(float(sp.Integral(x**n * (x - kv) ** n / sp.factorial(n) * sp.exp(x), (x, 0, kv)).evalf(30)))
            ok1 = ok1 and val <= (math.exp(kv) - 1) / math.factorial(n) + 1e-300
    # 대표 오답: 공식 해설의 a_2 = 6a_1 + k^2(e-1) (오기) 재현
    wrong = sp.simplify(6 * a[1] + k**2 * (sp.E - 1) - a[2])
    report(2, ok1 and ok2 and okrec, f"(1) 상한 (e^k-1)/n! 성립={ok1}, 극한 0; (2) a1..a4 일치={ok2}, 점화식={okrec}; 공식 오기 k^2(e-1) 차이={wrong}")


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
