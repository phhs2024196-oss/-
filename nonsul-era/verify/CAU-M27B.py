#!/usr/bin/env python3
"""CAU-M27B (2027학년도 중앙대 일반형 모의논술 의·약학부 추가 문제 5) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M27B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p05():
    """[5-1] P를 매개변수로 두고 ∠PFQ=30°를 벡터 각으로 풀어 넓이(신발끈), [5-2] 정사면체 좌표로 cosθ."""
    s = sp.symbols("s", positive=True)            # P=(s^2, 2s)
    P = sp.Matrix([s ** 2, 2 * s]); F = sp.Matrix([1, 0])
    # 접선: 2s·y = 2(x + s^2) → y절편 s
    xx = sp.symbols("xx")
    tangent_y = sp.solve(sp.Eq(2 * s * sp.Symbol("yy"), 2 * (0 + s ** 2)), sp.Symbol("yy"))[0]
    Q = sp.Matrix([0, tangent_y])
    u, v = P - F, Q - F
    cosang = (u.dot(v)) / (u.norm() * v.norm())
    sols = [r for r in sp.solve(sp.Eq(sp.simplify(cosang ** 2), sp.Rational(3, 4)), s) if r.is_positive]
    sols = [r for r in sols if sp.simplify(cosang.subs(s, r) - sp.sqrt(3) / 2) == 0]
    O = sp.Matrix([0, 0])
    poly = [O, F, P, Q]
    def area(pts):
        return sp.Abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2
    areas = [sp.nsimplify(sp.simplify(area([p.subs(s, r) for p in poly]))) for r in sols]
    ok1 = len(areas) == 1 and sp.simplify(areas[0] - 7 * sp.sqrt(3) / 18) == 0
    # 5-2
    A = sp.Matrix([0, 0, sp.sqrt(sp.Rational(2, 3))])
    B = sp.Matrix([1 / sp.sqrt(3), 0, 0]); C = sp.Matrix([-1 / (2 * sp.sqrt(3)), sp.Rational(1, 2), 0]); D = sp.Matrix([-1 / (2 * sp.sqrt(3)), -sp.Rational(1, 2), 0])
    edges = [(A - B).norm(), (A - C).norm(), (A - D).norm(), (B - C).norm(), (B - D).norm(), (C - D).norm()]
    reg = all(sp.simplify(e - 1) == 0 for e in edges)
    Pp = B + (C - B) / 3; Qp = A + 2 * (D - A) / 3
    w1, w2 = Qp - Pp, D - C
    cth = sp.nsimplify(sp.simplify(sp.Abs(w1.dot(w2)) / (w1.norm() * w2.norm())))
    ok2 = reg and sp.simplify(cth - 3 * sp.sqrt(5) / 10) == 0
    report(5, ok1 and ok2, f"[5-1] s={sols}, 넓이={areas} / [5-2] 정사면체 좌표 cosθ={cth}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
