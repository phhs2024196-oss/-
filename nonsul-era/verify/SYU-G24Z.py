#!/usr/bin/env python3
"""SYU-G24Z (2024학년도 삼육대 논술 자연계열2 수학 7~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<문항번호>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-G24Z"
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
    return {kk.strip(): sp.sympify(v) for kk, v in (p.split("=") for p in load_answer(seq).split(";"))}


def load_set(seq: int) -> set:
    s = load_answer(seq).strip().strip("{}")
    return {sp.nsimplify(sp.sympify(e)) for e in s.split(",")}


def p7():
    v = -t**2 + t
    ks = [r for r in sp.solve(sp.integrate(v, (t, 0, k)), k) if r > 0]
    kk = ks[0]
    roots = sorted(r for r in sp.solve(v, t) if 0 < r < kk)
    pts = [0] + roots + [kk]
    d = sum(sp.Abs(sp.integrate(v, (t, a, b))) for a, b in zip(pts, pts[1:]))
    ans = load_pairs(7)
    ok = len(ks) == 1 and sp.simplify(kk - ans["k"]) == 0 and sp.simplify(d - ans["d"]) == 0
    report(7, ok, f"k={kk}, 거리={d}")


def p8():
    # 좌표: 원 중심 원점, A(-5/2,0), B(5/2,0). D 는 원 위, BD=3 (위쪽)
    R = sp.Rational(5, 2)
    A, B = sp.Matrix([-R, 0]), sp.Matrix([R, 0])
    th = sp.symbols("th", real=True)
    P = lambda a: sp.Matrix([R * sp.cos(a), R * sp.sin(a)])  # noqa: E731
    # D: |BD|=3, y>0
    thD = [s for s in sp.solve(sp.Eq((P(th) - B).dot(P(th) - B), 9), th) if sp.sin(s) > 0][0]
    D = P(thD)
    # C: AD 가 ∠CAB 이등분 → AC 방향은 AB 방향을 AD 기준으로 대칭(각 2θ)
    ad = (D - A) / sp.sqrt((D - A).dot(D - A))
    ab = sp.Matrix([1, 0])
    ac = 2 * ad.dot(ab) * ad - ab                               # ab 를 ad 에 대해 반사
    s = sp.symbols("s", positive=True)
    Cs = [A + r * ac for r in sp.solve(sp.Eq((A + s * ac).dot(A + s * ac), R**2), s)]
    lens = [sp.nsimplify(sp.sqrt((c - A).dot(c - A))) for c in Cs if sp.simplify((c - B).norm()) != 0]
    ok = len(lens) == 1 and sp.simplify(lens[0] - sp.sympify(load_answer(8))) == 0
    report(8, ok, f"AD={sp.nsimplify((D - A).norm())}, AC 후보={lens}")


def p9():
    # m 격자 전수: 이동 후 y=log2(x-m+3/4)+2 가 제2사분면(x<0,y>0)을 지나는지
    def passes_q2(m):
        lo = m - sp.Rational(3, 4)          # 정의역 x>lo
        if lo >= 0:
            return False
        # x<0 구간에서 함수는 증가, sup 은 x→0- 값
        return sp.log(-m + sp.Rational(3, 4), 2) + 2 > 0
    ms = [sp.Rational(i, 40) for i in range(-80, 120)]
    good = [m for m in ms if not passes_q2(m)]
    m_min = min(good)
    contiguous = all((m >= m_min) == (m in good) for m in ms)
    ans = sp.sympify(load_answer(9))         # m>=1/2
    bound = ans.rhs if hasattr(ans, "rhs") else None
    ok = contiguous and bound is not None and m_min == bound and isinstance(ans, sp.GreaterThan)
    report(9, ok, f"제2사분면 안 지나는 m 의 최소={m_min}, 그 이상 전부 성립={contiguous}")


def p10():
    a, b = sp.symbols("a b")
    g1 = x**2 + x + 1
    g2 = (a * x + b) / (x + 2)
    eqs = [sp.Eq(g1.subs(x, 1), g2.subs(x, 1)), sp.Eq(g2.subs(x, 4), g1.subs(x, -1))]
    sol = sp.solve(eqs, [a, b], dict=True)[0]
    r = 22 % 5
    r = r if r < 4 else r - 5
    val = (g1 if r < 1 else g2).subs(sol).subs(x, r)
    ok = sp.simplify(val - sp.sympify(load_answer(10))) == 0
    report(10, ok, f"{sol}, f(22)=f({r})={val}")


def p11():
    f = sp.sin(3 * x)
    g = f.subs(x, x - sp.pi / 6)
    sols = sp.solveset(sp.Eq(f**2, g + sp.Rational(5, 4)), x, sp.Interval.Ropen(0, 2 * sp.pi / 3))
    got = {sp.nsimplify(s) for s in sols}
    ok = got == load_set(11)
    report(11, ok, f"해={got}")


def p12():
    a1 = sp.symbols("a1")
    a = lambda m: a1 - 4 * (m - 1)  # noqa: E731
    # 공차 -4 이므로 a25=a27 은 불가능 → |a25|=|a27| ⇔ a25=-a27
    a1v = sp.solve(sp.Eq(a(25), -a(27)), a1)[0]
    s = 0
    for m in range(1, 51):
        p_, q_ = sp.Abs(a(m + 1).subs(a1, a1v)), sp.Abs(a(m).subs(a1, a1v))
        s += (sp.sqrt(p_) - sp.sqrt(q_)) / (p_ - q_)          # 유리화(p_≠q_)
    s = sp.nsimplify(sp.simplify(s))
    num = sum(1 / (abs(float(a(m + 1).subs(a1, a1v))) ** 0.5 + abs(float(a(m).subs(a1, a1v))) ** 0.5) for m in range(1, 51))
    ans = sp.sympify(load_answer(12))
    ok = sp.simplify(s - ans) == 0 and abs(num - float(ans)) < 1e-9
    report(12, ok, f"a1={a1v}, 합={s} (수치 {num:.12f})")


def p13():
    c1, c0 = sp.symbols("c1 c0")
    f = sp.Rational(5, 8) * x**2 + c1 * x + c0
    sol = sp.solve([f.subs(x, 1) - 2, sp.diff(f, x).subs(x, 1) - 2], [c1, c0])
    f = f.subs(sol)
    assert sp.limit(8 * f / (x**2 + 1), x, sp.oo) == 5
    m = -1 / sp.diff(f, x).subs(x, 1)
    bs = [r for r in sp.solve(f - (2 + m * (x - 1)), x) if r != 1]
    b = bs[0]
    A, B, P = sp.Matrix([1, 2]), sp.Matrix([b, f.subs(x, b)]), sp.Matrix([k, f.subs(x, k)])
    expr = ((B - A)[0] * (P - A)[1] - (B - A)[1] * (P - A)[0]) / 2
    crit = [r for r in sp.solve(sp.diff(expr, k), k) if b < r < 1]
    ok = len(crit) == 1 and sp.simplify(crit[0] - sp.sympify(load_answer(13))) == 0
    report(13, ok, f"f={sp.expand(f)}, b={b}, k={crit}")


def p14():
    # 일반 다항식 f(x)=Σ c_i (x-4)^i (3차까지 충분) 로 조건을 걸고 둘째 극한 계산
    c = sp.symbols("c0:4")
    f = sum(ci * (x - 4)**i for i, ci in enumerate(c))
    L1 = sp.limit((f.subs(x, 2 * x + 4) - f.subs(x, -3 * x + 4) - 5 * x * f.subs(x, 4)) / (2 * x), x, 0)
    sol = sp.solve(sp.Eq(L1, 5), c[1])
    L2 = sp.limit((f.subs(x, x + 1) - (x - 2) * f.subs(x, 4) + x - 3) / (x - 3), x, 3)
    val = sp.simplify(L2.subs(c[1], sol[0]))
    ok = val.free_symbols == set() and val == sp.sympify(load_answer(14))
    report(14, ok, f"조건 {sp.Eq(L1, 5)} → 둘째 극한={val}")


def p15():
    def count_a(m):
        f = u**3 - 6 * u**2 + 9 * u + m
        roots = set(r for r in sp.Poly(f - 5, u).real_roots() if r >= 0)
        return sum(1 if r == 0 else 2 for r in roots)

    m_, a_ = 2, sp.Rational(5, 3)
    ff = t**3 - 6 * t**2 + 9 * t + m_
    L = sp.limit(sp.integrate(ff, (t, a_**2, x**2)) / (x**2 - a_**2), x, a_)
    lim_ok = sp.simplify(L - ff.subs(t, a_**2)) == 0
    good = [m for m in range(-50, 51) if count_a(m) >= 4]
    ok = lim_ok and sum(good) == sp.sympify(load_answer(15))
    report(15, ok, f"극한=f(a^2) {lim_ok}, m={good}, 합={sum(good)}")


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
