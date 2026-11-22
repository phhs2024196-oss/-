#!/usr/bin/env python3
"""HYU-G25A (한양대 2025학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction
from itertools import product

import sympy as sp

PREFIX = "HYU-G25A"
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


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def in_A(s: str) -> bool:
    """각 자릿수 0/1, 일의 자리를 제외한 자리의 0의 개수 합 <= 1 (자연수이므로 맨 앞자리 1)."""
    return s[0] == "1" and set(s) <= {"0", "1"} and s[:-1].count("0") <= 1


def A_n(k: int):
    """k자리 원소 — k<=12 는 0/1 문자열 전수 조사, 그 이상은 조건을 그대로 따라 생성."""
    if k <= 12:
        return sorted(int("".join(d)) for d in product("01", repeat=k) if in_A("".join(d)))
    out = set()
    for z in [None] + list(range(1, k - 1)):          # 맨 앞(0번)은 1, 가운데 1..k-2 중 0 하나 이하
        mid = ["1"] * k
        if z is not None:
            mid[z] = "0"
        for u in "01":
            mid[-1] = u
            s = "".join(mid)
            assert in_A(s)
            out.add(int(s))
    return sorted(out)


def p01():
    elems = []
    for k in range(1, 13):
        elems += A_n(k)
    elems.sort()
    # (1) 60<r 에서 P(10^5<카드<10^9)=4/7 인 r 전수 탐색(r 은 A 의 앞쪽 원소 개수, 12자리까지면 충분)
    rs = [r for r in range(61, len(elems) + 1)
          if Fraction(sum(1 for v in elems[:r] if 10**5 < v < 10**9), r) == Fraction(4, 7)]
    ok1 = rs == [int(sp.sympify(md_answer(1, "1")))]
    r = rs[0] if rs else 0
    # (2) X = 카드 숫자 전체의 0의 개수 → 분산
    zs = [str(v).count("0") for v in elems[:r]]
    E = Fraction(sum(zs), r)
    V = Fraction(sum(z * z for z in zs), r) - E**2
    ok2 = sp.Rational(V.numerator, V.denominator) == sp.sympify(md_answer(1, "2"))
    # (3) a_n = A_n 원소들에 나타나는 1의 총 개수, n=1..20
    a = [sum(str(v).count("1") for v in A_n(k)) for k in range(1, 21)]
    ok3 = sum(a) == sp.sympify(md_answer(1, "3")) and a[1] == 3
    # 대표 오답: 경우 1 의 r=49 (r>60 위반) 재현
    wrong = sp.solve(sp.Eq((sp.Symbol("R") - 21) / sp.Symbol("R"), sp.Rational(4, 7)))
    report(1, ok1 and ok2 and ok3, f"r={rs} V={V} sum a_k={sum(a)} (오답 r={wrong})")


def p02():
    s = sp.symbols("s", real=True)
    f = 1 / (x**2 + 1)
    fp = sp.diff(f, x)
    # (1) A(t,f(t)) 를 지나는 접선의 접점 s (s≠t)
    eq = sp.together(f.subs(x, t) - (fp.subs(x, s) * (t - s) + f.subs(x, s)))
    num = sp.factor(sp.numer(eq))
    roots = [r for r in sp.solve(num, s) if sp.simplify(r - t) != 0]
    xint = [sp.simplify(sp.solve(fp.subs(x, r) * (x - r) + f.subs(x, r), x)[0]) for r in roots]
    area = sp.Rational(1, 2) * sp.Abs(xint[0] - xint[1]) * f.subs(x, t)
    md1 = sp.sympify(md_answer(2, "1"), locals={"t": t})
    tv = [sp.Rational(-1, 2), sp.Rational(-1, 5), 0, sp.Rational(1, 3), sp.Rational(11, 20)]
    ok1 = len(roots) == 2 and all(abs(sp.N(area.subs(t, v) - md1.subs(t, v), 30)) < 1e-20 for v in tv)
    # 구간 밖(t^2=1/3)에서는 접점 하나가 A 와 겹침을 확인
    ok1 = ok1 and any(sp.simplify(r.subs(t, 1 / sp.sqrt(3)) - 1 / sp.sqrt(3)) == 0 for r in roots)
    # (2) T_n: 선분 위 x좌표 정수 개수를 정수 연산으로 셈 → a_n = S_n - S_{n-1}
    from math import isqrt

    def T(m):
        D = m * m * (4 * m * m - 1)                 # (x-3m)^2 = D 의 근 x = 3m ± sqrt(D)
        lo = 3 * m - isqrt(D)                       # sqrt(D) 는 정수가 아님(아래에서 확인)
        hi = 3 * m + isqrt(D)
        assert isqrt(D) ** 2 != D
        return hi - lo + 1
    S = [None] + [T(m) + 1 for m in range(1, 41)]
    a = [S[1]] + [S[m] - S[m - 1] for m in range(2, 41)]
    md2 = sp.sympify(md_answer(2, "2"), locals={"n": n})
    ok2 = all(md2.subs(n, m) == a[m - 1] for m in range(1, 41))
    # (3) 리만 합 극한: a_k 는 위에서 확인한 식, 인자 (a_k+4)/(16n)
    g = md1
    g2 = sp.diff(g, t, 2)
    k, N = sp.symbols("k N", positive=True, integer=True)
    arg = sp.simplify((md2.subs(n, k) + 4) / (16 * N))       # = k/(2N)
    u = sp.symbols("u", positive=True)
    lim = sp.integrate(g2.subs(t, arg.subs(k, u * N)), (u, 0, 1))   # (1/N)Σ h(k/N) → ∫_0^1 h(u)du
    lim = sp.nsimplify(sp.simplify(lim))
    Nn = 20000
    g2l = sp.lambdify(t, g2, "math")
    riem = sum(g2l(float(arg.subs({k: kk, N: Nn}))) for kk in range(1, Nn + 1)) / Nn
    md3 = sp.sympify(md_answer(2, "3"))
    ok3 = sp.simplify(lim - md3) == 0 and abs(riem - float(md3)) < 1e-3
    report(2, ok1 and ok2 and ok3, f"g(t)≈{sp.simplify(area.subs(t, sp.Rational(1, 3)))}@t=1/3, a_1..3={a[:3]}, lim={lim} (Riemann N={Nn}: {riem:.6f})")


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
