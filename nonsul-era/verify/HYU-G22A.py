#!/usr/bin/env python3
"""HYU-G22A (한양대 2022학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G22A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import math
    import random
    th = sp.symbols("theta", real=True)
    loc = {"theta": th, "t": t}
    # 좌표: 평면 beta = xy평면, 교선 l = x축, 평면 alpha 는 x축을 축으로 theta 만큼 회전.
    # alpha 위 점 (u, v) -> 공간 (u, v cos th, v sin th) -> beta 위 정사영 (u, v cos th, 0)
    def proj(u, v, thv):
        return (u, v * math.cos(thv))

    def dist(P, Q):
        return math.hypot(P[0] - Q[0], P[1] - Q[1])

    random.seed(1)
    ok1 = ok2 = ok3 = True
    e1 = sp.lambdify(th, sp.sympify(md_answer(1, "1"), locals=loc))
    e2 = sp.lambdify((t, th), sp.sympify(md_answer(1, "2"), locals=loc))
    for _ in range(20):
        thv = random.uniform(0.05, 1.5)
        # (1) BC // l, A 직각, AB=AC=1 : B(-1/√2,0), C(1/√2,0), A(0, 1/√2)
        A, B = proj(0, 1 / math.sqrt(2), thv), proj(-1 / math.sqrt(2), 0, thv)
        ok1 &= abs(dist(A, B) - e1(thv)) < 1e-12
        # (2) PQ=1, l 과 각 tv
        tv = random.uniform(0, math.pi / 2)
        P, Q = proj(0, 0, thv), proj(math.cos(tv), math.sin(tv), thv)
        ok2 &= abs(dist(P, Q) - e2(tv, thv)) < 1e-12
    # (3) cos th = 3/pi, 정삼각형을 임의 회전·평행이동해도 합이 같은지
    thv = math.acos(3 / math.pi)
    vals = []
    for _ in range(20):
        ang, ox, oy = random.uniform(0, 2 * math.pi), random.uniform(-3, 3), random.uniform(0.5, 3)
        pts = [(ox + math.cos(ang + k * 2 * math.pi / 3) / math.sqrt(3),
                oy + math.sin(ang + k * 2 * math.pi / 3) / math.sqrt(3)) for k in range(3)]
        pp = [proj(u, v, thv) for u, v in pts]
        vals.append(sum(dist(pp[i], pp[(i + 1) % 3])**2 for i in range(3)))
    ans3 = float(sp.sympify(md_answer(1, "3")))
    ok3 = max(abs(v - ans3) for v in vals) < 1e-12
    # 기호 확인: 2.의 식으로 t, pi/3-t, pi/3+t 합
    c2 = sp.symbols("c2", positive=True)
    S = sum(sp.sin(a)**2 * c2 + sp.cos(a)**2 for a in (t, sp.pi / 3 - t, sp.pi / 3 + t))
    ok3 &= sp.simplify(sp.expand_trig(S) - (sp.Rational(3, 2) * c2 + sp.Rational(3, 2))) == 0
    report(1, ok1 and ok2 and ok3, f"sum(3)={vals[0]:.12f} vs md {ans3:.12f}")


def p02():
    from fractions import Fraction as Fr
    # (1) 모집단 1,2,3 균등 → 표본평균(54) → -2X̄
    vals = [1, 2, 3]
    m = Fr(sum(vals), 3)
    var = sum((Fr(v) - m)**2 for v in vals) / 3
    mean_y, var_y = -2 * m, 4 * var / 54
    z = (Fr(-11, 3) - mean_y) / sp.sqrt(sp.Rational(var_y.numerator, var_y.denominator))
    table = {sp.Rational(1, 2): 0.1915, 1: 0.3413, sp.Rational(3, 2): 0.4332, 2: 0.4772}
    prob = 0.5 - table[sp.nsimplify(z)]
    a1 = parts(md_answer(2, "1"))
    ok1 = sp.Rational(a1["E"]) == sp.Rational(mean_y.numerator, mean_y.denominator) and \
        sp.Rational(a1["V"]) == sp.Rational(var_y.numerator, var_y.denominator) and abs(float(a1["P"]) - prob) < 1e-12
    # (2) 서로 다른 양의 정수 a^b = b^a 전수 탐색(1..300)
    sols = sorted((a, b) for a in range(1, 301) for b in range(1, 301) if a != b and pow(a, b) == pow(b, a))
    md2 = sorted(tuple(p) for p in sp.sympify(md_answer(2, "2")))
    ok2 = sols == md2
    # (3) 정 n 각형(둘레 1) 넓이: 변 1/n, 넓이 = n s^2 /(4 tan(pi/n))
    N = sp.symbols("N", positive=True)
    fN = N * (1 / N)**2 / (4 * sp.tan(sp.pi / N))
    a3 = parts(md_answer(2, "3"))
    ok3 = sp.simplify(fN.subs(N, 12) - sp.sympify(a3["f12"])) == 0 and \
        sp.simplify(sp.limit(fN, N, sp.oo) - sp.sympify(a3["limit"])) == 0
    report(2, ok1 and ok2 and ok3, f"E={mean_y} V={var_y} z={z} P={prob:.4f} sols={sols} f12={sp.nsimplify(fN.subs(N, 12))}")


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
