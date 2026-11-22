#!/usr/bin/env python3
"""HIU-G25S (홍익대 2025학년도 수시 논술 자연계열 세종) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction as Fr

import sympy as sp

PREFIX = "HIU-G25S"
WIDTH = 2
RESULTS = []
x, h = sp.symbols("x h", real=True)


def md_answer(no: int, sub: str = "1") -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b = sp.symbols("a b", real=True)
    f = a * x**3 + 7 * x**2 + b * x + a
    num = sp.expand(f.subs(x, 1 - h) - 1)
    # 극한 (num/h) = -4 ⟺ 상수항 0, h 의 계수 -4
    sol = sp.solve([num.coeff(h, 0), num.coeff(h, 1) + 4], [a, b], dict=True)[0]
    lim = sp.limit((num / h).subs(sol), h, 0)
    val = sol[a] + sol[b]
    report(1, lim == -4 and val == sp.sympify(md_answer(1)), f"a={sol[a]} b={sol[b]} lim={lim} a+b={val}")


def p02():
    seq = [Fr(1), Fr(6)]
    while len(seq) < 22:
        seq.append(2 * seq[-1] - seq[-2])          # 2a_{n+1}=a_n+a_{n+2}
    s = sum(1 / (seq[n] * seq[n + 1]) for n in range(20))
    report(2, s == Fr(md_answer(2)), f"a_1..a_3={seq[:3]} sum={s}")


def p03():
    import mpmath as mp
    F = lambda A: mp.log(6 - A) / mp.log(A) + 4 * mp.log(A) / mp.log(6 - A) - 4
    # a>1, b=6-a>1 → 1<a<5. F 는 (t-2)^2/t ≥ 0 꼴이라 부호변화 대신 최소점(=0) 탐색
    grid = [1 + 4 * k / 4000 for k in range(1, 4000)]
    roots = set()
    for i in range(1, len(grid) - 1):
        if F(grid[i]) <= F(grid[i - 1]) and F(grid[i]) <= F(grid[i + 1]) and F(grid[i]) < 1e-3:
            r = mp.findroot(lambda A: mp.diff(F, A), grid[i])   # 중근: F'=0 인 극소점
            if abs(F(r)) < 1e-15:
                roots.add(round(float(r), 9))
    parts = dict(seg.split("=") for seg in md_answer(3).replace(" ", "").split(";"))
    ok = roots == {float(sp.sympify(parts["a"]))} and abs(6 - float(sp.sympify(parts["a"])) - float(sp.sympify(parts["b"]))) < 1e-9
    report(3, ok, f"roots a={roots} b=6-a")


def p04():
    a = sp.symbols("a", real=True)
    left = sp.limit((-x + 2 + 1)**2, x, 1, dir="-")
    right = (a * 1 + 1 + 1)**2                        # x≥1 쪽은 다항식이므로 f(1) 에서 연속
    sols = sp.solve(sp.Eq(left, right), a)
    # x≠1 에서는 다항함수라 연속 → x=1 조건만
    ok = sum(sols) == sp.sympify(md_answer(4)) and len(sols) == 2
    report(4, ok, f"left={left} a={sols} sum={sum(sols)}")


def p05():
    cnt = []
    for k in range(-100, 101):
        p = sp.Poly(x**3 + 3 * x**2 + 3 - (x**2 + 4 * x + k), x)
        if len(set(sp.real_roots(p))) == 3:
            cnt.append(k)
    report(5, len(cnt) == int(md_answer(5)), f"k={cnt[0]}..{cnt[-1]} count={len(cnt)}")


def p06():
    A, B = sp.symbols("A B", real=True)
    As = sp.solveset(sp.sin(A) - sp.sin(A + sp.pi / 2), A, sp.Interval.open(0, sp.pi))
    found = []
    for Av in As:
        Bs = sp.solveset(sp.cos(sp.pi - B) + sp.sin(B), B, sp.Interval.open(0, sp.pi - Av))  # A+C=π-B
        for Bv in Bs:
            Cv = sp.pi - Av - Bv
            R = sp.sqrt(8)                              # πR^2=8π
            area = sp.simplify(2 * R**2 * sp.sin(Av) * sp.sin(Bv) * sp.sin(Cv))
            found.append((Av, Bv, Cv, area))
    ok = len(found) == 1 and found[0][3] == sp.sympify(md_answer(6))
    report(6, ok, f"(A,B,C,area)={found}")


def p07():
    a, b, p, q, s, tt = sp.symbols("a b p q s tt")
    f = p * x**2 + q * x + s                          # 최고차 비교: deg f = 2 (deg f=n → x^{n+1}: c(1-1/(n+1))=[n=2] )
    lhs = (x - 2) * f
    rhs = x**3 - a * x + b + sp.integrate(f.subs(x, tt), (tt, 2, x))
    sol = sp.solve(sp.Poly(sp.expand(lhs - rhs), x).all_coeffs(), [p, q, s, a, b], dict=True)
    vals = []
    for so in sol:
        fp1 = sp.diff(f.subs(so), x).subs(x, 1)
        vals.append(sp.simplify(fp1 + so[a] + so[b]))
    ok = len(set(vals)) == 1 and vals[0] == sp.sympify(md_answer(7)) and all(sp.Symbol("s") not in v.free_symbols for v in vals)
    report(7, ok, f"sol={sol} f'(1)+a+b={vals}")


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
