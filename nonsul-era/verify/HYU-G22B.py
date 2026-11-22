#!/usr/bin/env python3
"""HYU-G22B (한양대 2022학년도 논술 자연계열 오후1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G22B"
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
    T = sp.symbols("T", positive=True)
    # 반지름 1, 중심각 T: 현 = 코사인법칙, 활꼴 = 부채꼴 - 삼각형
    f = sp.sqrt(2 - 2 * sp.cos(T))
    g = T / 2 - sp.sin(T) / 2
    L1 = sp.limit(g / f, T, 0, "+")
    ok1 = sp.simplify(L1 - sp.sympify(md_answer(1, "1"))) == 0
    # (2) 실제 점으로 평균 계산(큰 n) + 정적분 기호계산
    fl = lambda k, N: math.hypot(1 - math.cos(2 * math.pi * k / N), math.sin(2 * math.pi * k / N))
    I2 = sp.integrate(2 * sp.sin(T / 2), (T, 0, 2 * sp.pi)) / (2 * sp.pi)
    N = 200000
    avg2 = sum(fl(k, N) for k in range(1, N)) / N
    ok2 = sp.simplify(I2 - sp.sympify(md_answer(1, "2"))) == 0 and abs(avg2 - float(I2)) < 1e-4
    # (3) S_1<...<S_{n-1}: 길이 2πk/n 인 호의 활꼴 넓이
    gl = sp.lambdify(T, g)
    avg3 = sum(gl(2 * math.pi * k / N)**2 for k in range(1, N)) / N
    I3 = sp.integrate(g**2, (T, 0, 2 * sp.pi)) / (2 * sp.pi)
    incr = all(gl(2 * math.pi * k / 50) < gl(2 * math.pi * (k + 1) / 50) for k in range(1, 49))
    ok3 = sp.simplify(I3 - sp.sympify(md_answer(1, "3"))) == 0 and abs(avg3 - float(I3)) < 1e-3 and incr
    report(1, ok1 and ok2 and ok3, f"lim={L1} I2={I2} avg2={avg2:.6f} I3={sp.simplify(I3)} avg3={avg3:.6f}")


def p02():
    import itertools
    import math
    p = sp.symbols("p", positive=True)
    # (1) 모든 승패열 전수(최대 5경기)
    E = 0
    for seq in itertools.product("AB", repeat=5):
        a = b = 0
        for i, w in enumerate(seq, 1):
            a += w == "A"; b += w == "B"
            if a == 3 or b == 3:
                break
        # 5자리 전체 확률을 합해도 같으므로 실제 길이 i 를 기댓값에 반영
        E += i * sp.Mul(*[p if w == "A" else 1 - p for w in seq])
    ok1 = sp.expand(E - sp.sympify(md_answer(2, "1"), locals={"p": p})) == 0
    # (2) 기울기 각 φ 의 외접 직사각형: 변 = 2*sqrt(9cos^2+4sin^2) 지지함수
    def side(phi):  # 법선 방향 phi 에 수직인 폭
        return 2 * math.sqrt(9 * math.cos(phi)**2 + 4 * math.sin(phi)**2)
    areas = []
    for i in range(1, 200000):
        phi = i * (math.pi / 2) / 200000
        a, b = side(phi), side(phi + math.pi / 2)
        if abs(a - 2 * math.sqrt(5)) < 1e-4:
            areas.append(a * b)
    ok2 = areas and all(abs(A - float(sp.sympify(md_answer(2, "2")))) < 1e-3 for A in areas)
    # (3) 증명: 항등식 ∫_{-a}^{a}(a-|x|)f'(x)dx = F(a)+F(-a)-2F(0) 확인(임의 f) + 우함수/비우함수 사례
    a = sp.symbols("a", positive=True)
    ok3 = True
    for fx in (x**3 + 2 * x**2 + sp.sin(x), sp.exp(x), sp.cos(x) + x**4, x * sp.exp(-x**2) + 1):
        lhs = sp.integrate((a - x) * sp.diff(fx, x), (x, 0, a)) + sp.integrate((a + x) * sp.diff(fx, x), (x, -a, 0))
        F = sp.integrate(fx, x)
        rhs = F.subs(x, a) + F.subs(x, -a) - 2 * F.subs(x, 0)
        ok3 &= sp.simplify(lhs - rhs) == 0
        zero = sp.simplify(lhs) == 0
        even = sp.simplify(fx - fx.subs(x, -x)) == 0
        ok3 &= zero == even
    ok3 &= md_answer(2, "3") is None
    report(2, ok1 and ok2 and ok3, f"E(p)={sp.expand(E)} areas~{areas[0]:.6f} n={len(areas)}")


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
