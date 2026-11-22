#!/usr/bin/env python3
"""HYU-M23 (한양대 2023학년도 수시 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산 후 YAML answer.value 와 비교."""
import re
import sys

import sympy as sp

PREFIX = "HYU-M23"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import math
    import numpy as np
    from scipy.integrate import quad
    # 1. 좌표: B=(0,0), C=(s,0), A 는 AB=AC=2s
    s = sp.symbols("s", positive=True)
    B = sp.Matrix([0, 0]); C = sp.Matrix([s, 0]); A = sp.Matrix([s / 2, sp.sqrt(4 * s**2 - s**2 / 4)])
    v = (B - A) + sp.Rational(3, 2) * (C - B)
    sv = sp.solve(sp.Eq(v.dot(v), 4), s)[0]
    AB = sp.simplify(2 * sv)
    ok1 = sp.simplify(AB - sp.sympify(md_answer(1, "1"))) == 0
    # 2. BC=4 로 두고 PB=AB+BC/2, PC=AC-BC/2 인 P 를 원의 교점으로 구함
    bc = 4.0
    Bn = np.array([0.0, 0.0]); Cn = np.array([bc, 0.0]); An = np.array([bc / 2, math.sqrt(4 * bc**2 - bc**2 / 4)])
    PB, PC = 2 * bc + bc / 2, 2 * bc - bc / 2
    px = (PB**2 - PC**2 + bc**2) / (2 * bc); py2 = PB**2 - px**2
    ratios = set()
    for sgn in (1, -1):
        Pn = np.array([px, sgn * math.sqrt(max(py2, 0))])
        PA = np.linalg.norm(An - Pn)
        d = bc / 2                                         # AQ - PQ = d
        # Q 가 P 중심 반지름 r 원 위에 있고 AQ=r+d 가 가능한 최소 r: |PA-r| <= r+d
        rs = np.linspace(0, PA, 2000001)
        feas = rs[np.abs(PA - rs) <= rs + d + 1e-12]
        ratios.add(round((feas.min() + d) / bc, 6))
    ok2 = len(ratios) == 1 and abs(ratios.pop() - float(sp.sympify(md_answer(1, "2")))) < 1e-5
    # 3. θ = 각 A (코사인법칙을 좌표로 대신: 벡터 사잇각)
    u = (B - A).subs(s, 1); w = (C - A).subs(s, 1)
    th = float(sp.acos(u.dot(w) / sp.sqrt(u.dot(u) * w.dot(w))))
    f = lambda z: (z - 7 / 8)**3 + 1 / th
    fp = lambda z: 3 * (z - 7 / 8)**2
    I1 = quad(lambda z: z * math.sin(z) * fp(math.cos(z)), 0, th, epsabs=1e-14)[0]
    I2 = quad(lambda z: f(math.sin(z)), math.pi / 2 - th, math.pi / 2, epsabs=1e-14)[0]
    ok3 = abs((I1 - I2) - float(sp.sympify(md_answer(1, "3")))) < 1e-9
    report(1, ok1 and ok2 and ok3, f"AB={AB} AQ/BC~{(feas.min() + d) / bc:.6f} I={I1 - I2:.12f}")


def p02():
    import math
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import sici
    # 1. 큰 n 에서 직접 합(중점 리만합, 오차 O(1/n^2))
    N = 200000
    k = np.arange(1, 2 * N + 1)
    xn = np.sum(2 * N / (2 * N + 2 * k - 1)) / N
    ok1 = abs(xn - float(sp.sympify(md_answer(2, "1")))) < 1e-9
    # 2. (i),(ii) 를 만족하는 f'(t)=1/√3+Si(t)-Si(π/√3) 로 부분적분 공식 [t f'] + [cos t] 를 수치 확인한 뒤,
    #    극한에는 조건 (ii),(iii) 의 끝값을 대입
    a0 = math.pi / math.sqrt(3)
    fp = lambda z: 1 / math.sqrt(3) + sici(z)[0] - sici(a0)[0]
    xx = a0 * 0.9
    lhs = quad(fp, xx, math.sqrt(3) * xx, epsabs=1e-13)[0]
    rhs = math.sqrt(3) * xx * fp(math.sqrt(3) * xx) - xx * fp(xx) + math.cos(math.sqrt(3) * xx) - math.cos(xx)
    okp = abs(lhs - rhs) < 1e-10
    val = math.pi * 1 - a0 * (1 / math.sqrt(3)) + math.cos(math.pi) - math.cos(a0)
    ok2 = okp and abs(val - float(sp.sympify(md_answer(2, "2")))) < 1e-12
    # 3. 좌표로 S_k 계산: A(-1,0), B(1,0), P_k=(cos(π - kπ/(n+1)), sin(...))
    def ratio(n):
        tot = 0.0
        for kk in range(1, n + 1):
            ang = math.pi - kk * math.pi / (n + 1)
            P = (math.cos(ang), math.sin(ang))
            Sk = abs((P[0] + 1) * 0 - P[1] * 2) / 2
            tot += Sk**2
        return tot / (n + 1)
    target = float(sp.sympify(md_answer(2, "3")))
    ok3 = all(abs(ratio(n) - target) < 1e-9 for n in (50, 1000, 20000))
    report(2, ok1 and ok2 and ok3, f"xn={xn:.12f} lim2={val:.10f} (f'(π) from (i)(ii)={fp(math.pi):.4f})")


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
