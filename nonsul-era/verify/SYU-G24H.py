#!/usr/bin/env python3
"""SYU-G24H (2024학년도 삼육대 논술 인문계열 수학 10~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<문항번호>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-G24H"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"
x, t, u, k = sp.symbols("x t u k", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq: int) -> str:
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return str(d["subparts"][0]["answer"]["value"])


def load_pairs(seq: int) -> dict:
    """'d=1/3; k=3/2' 형식."""
    return {kk.strip(): sp.sympify(v) for kk, v in (p.split("=") for p in load_answer(seq).split(";"))}


def load_set(seq: int) -> set:
    s = load_answer(seq).strip().strip("{}")
    return {sp.nsimplify(sp.sympify(e)) for e in s.split(",")}


def p10():
    v = -t**2 + t
    ks = [r for r in sp.solve(sp.integrate(v, (t, 0, k)), k) if r > 0]
    kk = ks[0]
    roots = sorted([r for r in sp.solve(v, t) if 0 < r < kk])
    pts = [0] + roots + [kk]
    d = sum(sp.Abs(sp.integrate(v, (t, a, b))) for a, b in zip(pts, pts[1:]))
    ans = load_pairs(10)
    ok = len(ks) == 1 and sp.simplify(kk - ans["k"]) == 0 and sp.simplify(d - ans["d"]) == 0
    wrong = sp.integrate(v, (t, 0, kk))  # 오답: 거리 = 변위 = 0
    report(10, ok, f"k={kk}, 거리={d} (오답 변위={wrong})")


def p11():
    a, b = sp.symbols("a b")
    g1 = x**2 + x + 1
    g2 = (a * x + b) / (x + 2)
    eqs = [sp.Eq(sp.limit(g1, x, 1, "-"), g2.subs(x, 1)),
           sp.Eq(sp.limit(g2, x, 4, "-"), g1.subs(x, -1))]   # x=4+ ≡ x=-1+ (주기 5)
    sol = sp.solve(eqs, [a, b], dict=True)[0]
    r = 22 % 5
    r = r if r < 4 else r - 5          # 기본 구간 [-1,4)
    val = (g1 if r < 1 else g2).subs(sol).subs(x, r)
    # 연속성 수치 확인: 구간 끝점 주변
    f = lambda z: (g1 if ((z + 1) % 5) - 1 < 1 else g2).subs(sol).subs(x, ((z + 1) % 5) - 1)
    cont = all(abs(float(f(c - sp.Rational(1, 10**6)) - f(c))) < 1e-4 for c in (1, 4, 6, 9, -1))
    ok = cont and sp.simplify(val - sp.sympify(load_answer(11))) == 0
    report(11, ok, f"a,b={sol}, f(22)=f({r})={val}, 연속 수치확인={cont}")


def p12():
    f = sp.sin(3 * x)
    g = f.subs(x, x - sp.pi / 6)
    eq = f**2 - g - sp.Rational(5, 4)
    sols = sp.solveset(eq, x, sp.Interval.Ropen(0, 2 * sp.pi / 3))
    sols = {sp.nsimplify(s) for s in sols}
    # 수치 전수 확인(격자에서 근 근처만 0)
    ok = sols == load_set(12) and all(abs(float(eq.subs(x, s))) < 1e-12 for s in sols)
    wrong = sp.solveset(f**2 - f.subs(x, x + sp.pi / 6) - sp.Rational(5, 4), x, sp.Interval.Ropen(0, 2 * sp.pi / 3))
    report(12, ok, f"해={sorted(sols, key=float)} (부호 반대 평행이동 오답={wrong})")


def p13():
    a1 = sp.symbols("a1")
    an = lambda n: a1 - 4 * (n - 1)
    cand = [s for s in sp.solve(sp.Eq(an(25)**2, an(27)**2), a1)]
    vals = []
    for c in cand:
        a = lambda n: an(n).subs(a1, c)
        S = sum(1 / (sp.sqrt(sp.Abs(a(kk + 1))) + sp.sqrt(sp.Abs(a(kk)))) for kk in range(1, 51))
        vals.append(sp.nsimplify(sp.N(S, 50), [sp.sqrt(2)]))
    wrong = sp.Rational(1, 4) * (sp.sqrt(abs(100)) - sp.sqrt(abs(104 - 4 * 51)))
    ok = len(cand) == 1 and sp.simplify(vals[0] - sp.sympify(load_answer(13))) == 0
    report(13, ok, f"a1={cand}, 합={vals} (절댓값 무시 오답={wrong})")


def p14():
    c0, c1 = sp.symbols("c0 c1")
    f = sp.Rational(5, 8) * x**2 + c1 * x + c0          # (가): 이차, 최고차 5/8 (8f/(x^2+1)→5)
    assert sp.limit(8 * f / (x**2 + 1), x, sp.oo) == 5
    sol = sp.solve([f.subs(x, 1) - 2, sp.diff(f, x).subs(x, 1) - 2], [c0, c1], dict=True)[0]
    f = f.subs(sol)
    assert sp.limit((f - 2) / (x - 1), x, 1) == 2
    m = -1 / sp.diff(f, x).subs(x, 1)
    line = 2 + m * (x - 1)
    b = [r for r in sp.solve(f - line, x) if r != 1][0]
    # 넓이 = 1/2 |(B-A)×(P-A)|, b<k<1 에서 최대
    A = sp.Matrix([1, 2]); B = sp.Matrix([b, f.subs(x, b)]); P = sp.Matrix([k, f.subs(x, k)])
    area = sp.Abs(((B - A)[0] * (P - A)[1] - (B - A)[1] * (P - A)[0])) / 2
    expr = ((B - A)[0] * (P - A)[1] - (B - A)[1] * (P - A)[0]) / 2
    crit = [r for r in sp.solve(sp.diff(expr, k), k) if b < r < 1]
    # 격자 탐색으로 최대 확인
    grid = [b + (1 - b) * sp.Rational(i, 1000) for i in range(1, 1000)]
    kmax = max(grid, key=lambda z: float(area.subs(k, z)))
    ok = len(crit) == 1 and sp.simplify(crit[0] - sp.sympify(load_answer(14))) == 0 and abs(float(kmax - crit[0])) < 0.01
    report(14, ok, f"f={sp.expand(f)}, B_x={b}, 최대 k={crit} (격자 {float(kmax):.3f}, 꼭짓점 오답={sp.solve(sp.diff(f, x), x)})")


def p15():
    def count_a(m):
        f = u**3 - 6 * u**2 + 9 * u + m
        roots = [r for r in sp.Poly(f - 5, u).real_roots() if r >= 0]
        roots = set(roots)
        return sum(1 if r == 0 else 2 for r in roots)

    # 극한식 = f(a^2) 확인(임의 m, a)
    m_, a_ = 3, sp.Rational(3, 2)
    ff = t**3 - 6 * t**2 + 9 * t + m_
    L = sp.limit(sp.integrate(ff, (t, a_**2, x**2)) / (x**2 - a_**2), x, a_)
    lim_ok = sp.simplify(L - ff.subs(t, a_**2)) == 0
    good = [m for m in range(-50, 51) if count_a(m) >= 4]
    total = sum(good)
    wrong = sum(m for m in range(-50, 51) if sum(2 for r in set(sp.Poly(u**3 - 6*u**2 + 9*u + m - 5, u).real_roots()) if r >= 0) >= 4)
    ok = lim_ok and total == sp.sympify(load_answer(15))
    report(15, ok, f"극한=f(a^2) {lim_ok}, m={good}, 합={total} (a^2=0 이중계산 오답={wrong})")


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
