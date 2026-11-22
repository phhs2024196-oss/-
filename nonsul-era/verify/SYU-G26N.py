#!/usr/bin/env python3
"""SYU-G26N (삼육대 2026학년도 논술 자연계열 수학 문제 7~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML 의 answer.value 와 비교.
p<문항번호>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-G26N"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
ITEM_DIR = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def latex_to_sympy(s: str):
    s = s.strip()
    s = re.sub(r"\\[dt]?frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\log_(\d+)\s*(\d+)", r"log(\2,\1)", s)
    s = s.replace("\\pi", "*pi")
    s = re.sub(r"(\d)\s*sqrt", r"\1*sqrt", s)
    s = re.sub(r"^\*", "", s).replace("(*pi", "(pi")
    return sp.sympify(s)


def load_answer(seq: int):
    d = yaml.safe_load((ITEM_DIR / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return latex_to_sympy(str(d["subparts"][0]["answer"]["value"]))


def check(seq, val, msg):
    ans = load_answer(seq)
    report(seq, sp.simplify(sp.nsimplify(val) - ans) == 0 if not isinstance(val, sp.Basic) else sp.simplify(val - ans) == 0,
           f"{msg} = {val}  (yaml {ans})")


def p7():
    th = sp.Rational(3, 5) * sp.pi
    R, gap = 40, 40 - 30                      # 바깥 반지름, 고무날 없는 안쪽 길이
    one = sp.Rational(1, 2) * R**2 * th - sp.Rational(1, 2) * gap**2 * th
    sweeps = sp.Rational(5, 2) * 2            # 1초 1왕복 = 2번 쓸기
    check(7, one * sweeps, "쓸기 1회 넓이×횟수")


def p8():
    a, b, X = sp.symbols("a b X", real=True)
    f = 2**(X + 1) + a
    g = sp.log(-X + 2, 2) + b
    # 점근선: f → y=a (X→-∞), g → x=2 (진수 0)
    m_x = sp.solve(-X + 2, X)[0]
    l_y = sp.limit(f, X, -sp.oo)
    A = (m_x, f.subs(X, m_x))
    Bx = sp.solve(sp.Eq(-X + 2, 2**(l_y - b)), X)[0]
    B = (Bx, l_y)
    w = sp.symbols("w", real=True)            # w = a-b
    d2 = sp.simplify(((A[0] - B[0])**2 + (A[1] - B[1])**2).subs(b, a - w))
    sols = sp.solve(sp.Eq(d2, 100), w)
    sols = [s for s in sols if s.is_real]
    check(8, sp.simplify(sols[0]) if len(sols) == 1 else sp.nan, f"a-b 실수해 {sols}")


def p9():
    a, b, c = 9, 4, 7                         # BC, CA, AB
    cosA = sp.Rational(b**2 + c**2 - a**2, 2 * b * c)
    sinA = sp.sqrt(1 - cosA**2)
    R = a / (2 * sinA)
    S = sp.Rational(1, 2) * b * c * sinA
    r = 2 * S / (a + b + c)
    # 헤론 공식으로 넓이 교차 확인
    s = sp.Rational(a + b + c, 2)
    assert sp.simplify(S - sp.sqrt(s * (s - a) * (s - b) * (s - c))) == 0
    check(9, sp.radsimp(R - r), "R-r")


def p10():
    a1, d = sp.symbols("a1 d", positive=True)
    k = sp.symbols("k", integer=True, positive=True)
    an = lambda m: a1 + (m - 1) * d
    Sn = lambda m: m * (2 * a1 + (m - 1) * d) / 2
    sol = sp.solve([sp.Eq(an(2), 3 * an(1)), sp.Eq(sum(1 / Sn(i) for i in range(1, 4)), sp.Rational(7, 18))], [a1, d], dict=True)
    sol = [s for s in sol if s[a1] > 0 and s[d] >= 0][0]
    val = sum((an(i + 1) / (Sn(i) * Sn(i + 1))).subs(sol) for i in range(1, 10))
    check(10, sp.nsimplify(val), "직접 합")


def p11():
    # f 기함수, g 우함수인 임의 다항식 계수로 일반 확인: 적분값이 6h(7)인지 기호로 검산 후 h(-7) 계산
    c = sp.symbols("c1:6")
    e = sp.symbols("e0:4")
    f = sum(c[i] * x**(2 * i + 1) for i in range(5))
    g = sum(e[i] * x**(2 * i) for i in range(4))
    h = sp.expand(f * g)
    I = sp.integrate((2 * x**5 + x**3 + 3) * sp.diff(h, x), (x, -7, 7))
    assert sp.simplify(I - 6 * h.subs(x, 7)) == 0      # 적분 = 6h(7) (계수 무관)
    H7 = sp.symbols("H7")
    h7 = sp.solve(sp.Eq(6 * H7, 42), H7)[0]
    hm7 = -h7                                           # 기함수
    assert sp.simplify(h.subs(x, -7) + h.subs(x, 7)) == 0
    check(11, hm7, "h(-7)")


def p12():
    found = []
    for m in range(1, 60):
        L = sp.limit((x**(m + 2) - (m + 2) * x + m + 1) / (x**(m + 1) - 1)**2, x, 1)
        if L == sp.Rational(5, 8):
            found.append(m)
    ok_unique = len(found) == 1
    check(12, found[0] if ok_unique else sp.nan, f"전수 탐색 n∈[1,59] 해 {found}")


def p13():
    a, x1, x2 = sp.symbols("a x1 x2", real=True)
    f = x**3 - 5 * x
    g = f + a
    t1 = sp.diff(f, x).subs(x, x1) * (x - x1) + f.subs(x, x1)
    t2 = sp.diff(g, x).subs(x, x2) * (x - x2) + g.subs(x, x2)
    eqs = sp.Poly(sp.expand(t1 - t2), x).all_coeffs()
    cands = set()
    for s in sp.solve(list(eqs) + [sp.Eq((x2 - x1)**2, 25)], [a, x1, x2], dict=True):
        if s[a].is_positive:
            cands.add(s[a])
    check(13, list(cands)[0] if len(cands) == 1 else sp.nan, f"양수 a 후보 {cands}")


def p14():
    a = sp.symbols("a", positive=True)
    T = sp.symbols("T", positive=True)
    xa = [s for s in sp.solve(sp.Eq(x**2 + a * x, T**2), x) if s.subs({a: 1, T: 5}) > 0][0]
    A = sp.Matrix([xa, sp.sqrt(a * xa)])
    B = sp.Matrix([xa, -sp.sqrt(a * xa)])
    C = sp.Matrix([-T, 0])
    u, v = A - C, B - C
    cos_th = (u.dot(v)) / sp.sqrt(u.dot(u) * v.dot(v))
    sin2 = 1 - cos_th**2
    # 수치로 원주각 관계(sinθ=y_A/T) 확인
    num = {a: sp.Rational(7, 3), T: 11}
    assert abs(sp.N(sin2.subs(num) - (a * xa / T**2).subs(num))) < 1e-12
    Lim = sp.limit(T * a * xa / T**2, T, sp.oo)
    aval = sp.solve(sp.Eq(Lim, sp.Rational(3, 10)), a)
    check(14, aval[0] if len(aval) == 1 else sp.nan, f"lim = {Lim}")


def p15():
    # f=(x-α)x^2, g=(x-α)^2 결정 과정을 후보 탐색으로 일부 확인하고 극솟값 조건으로 α 계산
    al = sp.symbols("alpha", negative=True)
    f = (x - al) * x**2
    g = (x - al)**2
    # (나) 수치 확인: α=-2 예에서 ∫_α^x fg ≥0 (x 여러 점)
    F = sp.integrate((f * g).subs(al, -2), (x, -2, t))
    assert all(F.subs(t, v) >= 0 for v in [-6, -4, -3, -2.5, -1, 0, 1, 3])
    h = sp.cancel(f * g / x)
    crit = sp.solve(sp.diff(h, x), x)
    mins = [cp for cp in crit if sp.simplify(sp.diff(h, x, 2).subs(x, cp)).subs(al, -1) > 0]
    xm = mins[0]
    sol = [s for s in sp.solve(sp.Eq(h.subs(x, xm), -432), al) if s.is_real and s < 0]
    check(15, sol[0] if len(sol) == 1 else sp.nan, f"극소점 x={xm}, α 후보 {sol}")


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
