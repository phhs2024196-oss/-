#!/usr/bin/env python3
"""SGU-G23A (서강대 2023학년도 수시 논술 자연계열 1차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "SGU-G23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


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


def H(nn, r):
    return sp.binomial(nn + r - 1, r)


def p01():
    # (1) 확률의 합 = 1 을 만족하는 자연수 d 를 전수 탐색
    num = sum(H(11 - i, i) * H(11 - i, i) for i in range(11))
    ds = [d for d in range(1, 200) if H(d, 10) == num]
    ok1 = ds == [int(md_answer(1, "1"))]
    # (2) b = 확률의 합이 1이 되게 하는 정규화 상수
    s = sp.symbols("s", positive=True)
    b = sum(sp.binomial(21, 2 * i + 1) * s**(20 - 2 * i) * (1 - s)**(2 * i + 1) for i in range(11))
    ok2 = sp.expand(b - sp.sympify(md_answer(1, "2"), locals={"s": s})) == 0
    # (3) 표의 상금, X~B(10,1/2) 정확한 기댓값
    g = [2, 1, 5, 7, 17, 31, 65, 127, 257, 511, 1025]
    E3 = sum(sp.Rational(g[i]) * sp.binomial(10, i) / 2**10 for i in range(11))
    ok3 = E3 == sp.sympify(md_answer(1, "3"))
    # (4) 정규근사 N(10,9): P(Z>=(k-10)/3)>=0.23 인 최대 k. 표 값(0.7→0.2580, 1.0→0.3413)과 정확한 정규분포 둘 다로 확인
    Phi0 = lambda z: 0.5 * math.erf(z / math.sqrt(2))     # P(0<=Z<=z)
    kmax = max(k for k in range(1, 101) if 0.5 - Phi0((k - 10) / 3) >= 0.23)
    tbl = {0.7: 0.2580, 1.0: 0.3413}
    table_ok = (0.5 - tbl[0.7] >= 0.23) and (0.5 - tbl[1.0] < 0.23)   # k=12: z=2/3<0.7 → 확률 ≥ 0.242, k=13: z=1 → 0.1587
    ok4 = kmax == int(md_answer(1, "4")) and table_ok and (12 - 10) / 3 < 0.7
    report(1, ok1 and ok2 and ok3 and ok4, f"d={ds} b-check={ok2} E={E3} kmax={kmax}")


def p02():
    tt = sp.symbols("t", real=True)
    # (1) 넓이를 좌표로 직접(외적) 계산하여 격자에서 비교
    P = (sp.cos(tt), sp.sin(tt)); Q = (2 * sp.cos(tt**2 + tt), 2 * sp.sin(tt**2 + tt))
    area = sp.Abs(P[0] * Q[1] - P[1] * Q[0]) / 2
    md1 = sp.sympify(md_answer(2, "1"), locals={"t": tt})
    L = math.sqrt(2 * math.pi)
    grid = [-L + 1e-6 + k * (2 * L - 2e-6) / 1000 for k in range(1001)]
    ok1 = all(abs(float(area.subs(tt, v)) - float(md1.subs(tt, v))) < 1e-9 for v in grid[::10])
    # (2) S 의 미분불가능점: S=0 인 점(후보)에서 좌·우 미분계수를 sympy limit 로 계산
    h = sp.symbols("h", real=True)
    S = sp.Abs(sp.sin(tt**2))
    cands = [0, sp.sqrt(sp.pi), -sp.sqrt(sp.pi)]
    nd = set()
    for c0 in cands:
        expr = (S.subs(tt, c0 + h) - S.subs(tt, c0)) / h
        rp, lm = sp.limit(expr, h, 0, "+"), sp.limit(expr, h, 0, "-")
        if sp.simplify(rp - lm) != 0:
            nd.add(sp.nsimplify(c0))
    ok2 = nd == set(sp.sympify(md_answer(2, "2")))
    # (3) a=sqrt(cot θ)>1, 거리 2a/sqrt(1+a^4) <= 3^(1/4) 인 θ 를 격자 탐색 → 상한
    good = []
    for k in range(1, 20000):
        th = (math.pi / 4) * k / 20000
        a = math.sqrt(1 / math.tan(th))
        if a > 1 and 2 * a / math.sqrt(1 + a**4) <= 3**0.25 + 1e-15:
            good.append(th)
    md3 = sp.sympify(md_answer(2, "3"))
    ok3 = abs(max(good) - float(md3.sup)) < 1e-4 and min(good) < 1e-3 and md3.left_open and not md3.right_open \
        and all(th <= float(md3.sup) + 1e-12 for th in good)
    # (4) 현의 길이 l^2 = 4(√3 - dist^2), a^2 = cot θ
    th = sp.symbols("theta", positive=True)
    a2 = sp.cot(th)
    l2 = 4 * (sp.sqrt(3) - 4 * a2 / (1 + a2**2))
    lim4 = sp.limit((l2 - 4 * sp.sqrt(3)) / th, th, 0, "+")
    ok4 = sp.simplify(lim4 - sp.sympify(md_answer(2, "4"))) == 0
    report(2, ok1 and ok2 and ok3 and ok4, f"S-grid={ok1} nondiff={sorted(nd, key=float)} theta_sup~{max(good):.5f} lim={lim4}")


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
