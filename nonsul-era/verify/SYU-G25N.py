#!/usr/bin/env python3
"""SYU-G25N (삼육대 2025학년도 논술 자연계열 수학 문제 7~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML 의 answer.value 와 비교.
p<문항번호>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import numpy as np
import sympy as sp
import yaml

PREFIX = "SYU-G25N"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
ITEM_DIR = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def latex_to_sympy(s: str):
    s = s.strip()
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"(\d)([A-Za-z(])", r"\1*\2", s)
    return sp.sympify(s, locals={"A": sp.Symbol("A"), "B": sp.Symbol("B"), "a": sp.Symbol("a", positive=True)})


def load_answer(seq: int):
    d = yaml.safe_load((ITEM_DIR / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return latex_to_sympy(str(d["subparts"][0]["answer"]["value"]))


def check(seq, val, msg):
    ans = load_answer(seq)
    val = sp.sympify(val)
    report(seq, sp.simplify(val - ans) == 0, f"{msg} = {val}  (yaml {ans})")


def count_roots(h, lo, hi, pts=200000, exact_pts=()):
    """[lo,hi] 에서 h 의 영점 개수: 격자 부호변화 + 정확한 영점(exact_pts, 접점 포함)."""
    xs = np.linspace(lo, hi, pts)
    xs = xs[~np.any(np.isclose(xs[:, None], np.array(list(exact_pts) or [np.inf])[None, :], atol=1e-7), axis=1)]
    v = h(xs)
    sgn = np.sign(v)
    cnt = 0
    for i in range(len(xs) - 1):
        if sgn[i] * sgn[i + 1] < 0:
            # 사이에 정확한 영점이 있으면 그 점에서 이미 셈
            if not any(xs[i] < e < xs[i + 1] for e in exact_pts):
                cnt += 1
    return cnt + len(exact_pts)


def p7():
    X = sp.symbols("X", real=True)
    f = -sp.log(1 - X, 2)
    g = -sp.log(X, 2)
    # 단조성: f'>0 on [0,1) → 최소 x=0, g'<0 on (0,1] → 최소 x=1
    assert sp.simplify(sp.diff(f, X).subs(X, sp.Rational(1, 2))) > 0 and sp.diff(g, X).subs(X, sp.Rational(1, 2)) < 0
    a, b = 0, 1
    cs = [s for s in sp.solve(sp.Eq(1 - X, X), X) if 0 < s < 1]
    c = cs[0]
    check(7, a + 2 * b + 4 * c, f"a={a}, b={b}, c={c}")


def p8():
    # 일반 4차 다항식 계수로 두 극한 조건을 걸고 f'(1)+2g'(1) 이 하나로 정해지는지 계산
    fa = sp.symbols("f0:5")
    ga = sp.symbols("g0:5")
    h = sp.symbols("h")
    f = sum(fa[i] * x**i for i in range(5))
    g = sum(ga[i] * x**i for i in range(5))
    c1 = sp.solve(sp.Eq((f + g).subs(x, 1), 0), fa[0])[0]          # (분모)→0 이므로 (분자)→0
    f = f.subs(fa[0], c1)
    e1 = sp.limit(((f + g).subs(x, x - 1)) / (x - 2), x, 2)
    e2 = sp.limit((f.subs(x, 1 + h / 2) + g.subs(x, 1 - h / 3)) / h, h, 0)
    sol = sp.solve([sp.Eq(e1, 9), sp.Eq(e2, sp.Rational(1, 3))], [fa[1], ga[1]], dict=True)[0]
    target = sp.simplify((sp.diff(f, x) + 2 * sp.diff(g, x)).subs(x, 1).subs(sol))
    fp = sp.simplify(sp.diff(f, x).subs(x, 1).subs(sol))
    assert not target.free_symbols
    check(8, target, f"f'(1)={fp}")


def p9():
    s = sp.symbols("s", positive=True)       # AB = s
    AB, BC, CA = s, 2 * s, 2 * s / sp.sqrt(2)
    r = sp.sqrt(7)                            # πr^2 = 7π
    p = (AB + BC + CA) / 2
    heron = sp.sqrt(p * (p - AB) * (p - BC) * (p - CA))
    sol = [v for v in sp.solve(sp.Eq(heron, r * p), s) if v.is_positive]
    val = sp.nsimplify(sp.simplify((AB + BC - CA).subs(s, sol[0])))
    check(9, sp.radsimp(sp.simplify(val)), f"AB={sp.simplify(sol[0])}")


def f10(X):
    if X < -1:
        return (-5 * X - 9) / (X + 2)
    if X < 1:
        return 4 * X**2 + 4 * X - 3
    return -sp.log(9 * X - 1, 2)


def g10(k):
    """y=f(x) 와 y=k 의 교점 개수(정확 계산)."""
    k = sp.nsimplify(k)
    cnt = 0
    X = sp.symbols("X", real=True)
    for s in sp.solve(sp.Eq((-5 * X - 9), k * (X + 2)), X):
        if s.is_real and s < -1 and s != -2:
            cnt += 1
    for s in sp.solve(sp.Eq(4 * X**2 + 4 * X - 3, k), X):
        if s.is_real and -1 <= s < 1:
            cnt += 1
    for s in sp.solve(sp.Eq(9 * X - 1, 2**(-k)), X):
        if s.is_real and s >= 1:
            cnt += 1
    return cnt


def p10():
    eps = sp.Rational(1, 1000)
    L1 = g10(-4 - eps)
    L2 = g10(-3 + eps)
    L3 = g10(5 - eps)
    # 근방에서 값이 일정한지(좌·우극한 확인)
    assert g10(-4 - eps / 7) == L1 and g10(-3 + eps / 7) == L2 and g10(5 - eps / 7) == L3
    check(10, L1 + L2 + L3, f"lim g = {L1}, {L2}, {L3}")


def f11_np(xs):
    u = np.mod(xs, 3.0)
    u = np.where(u == 0, 3.0, u)
    return np.where(u <= 1, 3.0**u - 1, -2 * np.log(u) / np.log(3) + 2)


def k11(nn):
    """y=f(x) 와 y=2^{x/n} 의 교점 개수. 2^{x/n}>2 인 x>n 에서는 f≤2 이므로 (0,n] 만 조사.
    정확한 영점: x=n 이 f 의 최대점(3m+1)이면 f(n)=2=2^{n/n}."""
    exact = [float(nn)] if nn % 3 == 1 else []
    h = lambda xs: f11_np(xs) - 2.0 ** (xs / nn)
    return count_roots(h, 1e-9, nn + 0.5, pts=40000 * (nn + 1), exact_pts=exact)


def p11():
    ks = {nn: k11(nn) for nn in range(1, 64)}
    S = {}
    for nn, k in ks.items():
        S[k] = S.get(k, 0) + nn
    # k ≤ 20 인 S_k 는 n ≤ 30 에서 완결(n 이 커지면 k 도 증가) — n≤63 까지 조사해 확인
    assert max(k for nn, k in ks.items() if nn > 31) > 20
    total = sum(S[2 * j] for j in range(1, 11))
    check(11, total, f"S_2,S_4,S_6={S[2]},{S[4]},{S[6]}")


def h12(k):
    X = sp.symbols("X", real=True)
    k = sp.nsimplify(k)
    cnt = 0
    for s in sp.solve(sp.Eq(2 * X**2 + 4 * X - 2, 2 * X + k), X):
        if s.is_real and s < 2:
            cnt += 1
    for s in sp.solve(sp.Eq(X**2 - 10 * X + 30, 2 * X + k), X):
        if s.is_real and s >= 2:
            cnt += 1
    return cnt


def p12():
    # g 가 연속 ⇔ f(a)=2a+k (f 는 연속: f(2-)=f(2)=14) → a 의 개수 = 교점 개수
    X = sp.symbols("X")
    assert (2 * X**2 + 4 * X - 2).subs(X, 2) == (X**2 - 10 * X + 30).subs(X, 2)
    ks = [k for k in range(-60, 61) if h12(k) == 4]
    check(12, len(ks), f"h(k)=4 인 정수 k: {ks[0]}..{ks[-1]}")


def f13(nn):
    """직선 y=2x/n-1 과 y=sin(nπx) 교점 개수. |sin|≤1 이므로 x∈[0,n] 만."""
    exact = []
    # 정확한 교점: sin=0 이고 직선=0 → x=n/2, nπ·n/2 가 π의 정수배 ⇔ n^2/2 정수 ⇔ n 짝수
    if (nn * nn) % 2 == 0:
        exact.append(nn / 2)
    h = lambda xs: np.sin(nn * np.pi * xs) - (2 * xs / nn - 1)
    return count_roots(h, 0.0, float(nn), pts=4000 * nn * nn, exact_pts=exact)


def p13():
    a, b = f13(9), f13(20)
    small = [f13(m) for m in (1, 2, 3)]
    assert small == [1, 3, 9], small            # 공식 해설 풀이 2의 작은 경우와 일치
    check(13, a + b, f"f(9)={a}, f(20)={b}")


def p14():
    a, b = sp.symbols("a b", positive=True)
    A, B = sp.symbols("A B")
    f = x**3 - a**2 * x + 5
    Aval = sp.integrate(f, (x, -b, 0))
    Bval = sp.integrate(-f.subs(x, x - b) + 2, (x, b, a + b))
    target = sp.integrate(f, (x, -a, -b))
    ans = load_answer(14)
    diff = sp.simplify(ans.subs({sp.Symbol("A"): Aval, sp.Symbol("B"): Bval, sp.Symbol("a", positive=True): a}) - target)
    # 극솟값>2 조건 만족 예에서도 수치 확인
    num = {a: 1, b: sp.Rational(1, 2)}
    assert sp.N((5 - 2 * a**3 / (3 * sp.sqrt(3))).subs(num)) > 2
    report(14, diff == 0, f"∫_(-a)^(-b) f - (yaml식 with A,B 대입) = {diff}  (yaml {ans})")


def p15():
    T = sp.symbols("T", nonnegative=True)
    t1 = [s for s in sp.solve(sp.Eq(sp.integrate(3 * t**2, (t, 0, T)), 4 * 16), T) if s.is_real and s > 0][0]
    xP = sp.integrate(3 * t**2, (t, 0, T))
    xQ = sp.integrate(3 * t + 6, (t, 0, T)) + 2   # Q 는 P 보다 호의 길이 2(반 바퀴) 앞에서 출발
    D = sp.expand(xP - xQ)
    # 만남 ⇔ D(T) ≡ 0 (mod 4). D 의 [0,t1] 상의 값 범위에서 4m=D 의 해를 셈
    cnt = 0
    for m in range(-20, 21):
        roots = set(sp.Poly(D - 4 * m, T).real_roots())      # 중근은 한 시각이므로 집합으로 1회
        cnt += sum(1 for r in roots if 0 <= r <= t1)
    check(15, cnt, f"t1={t1}, D(T)={D}")


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
