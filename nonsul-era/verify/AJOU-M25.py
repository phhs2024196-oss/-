#!/usr/bin/env python3
"""AJOU-M25 (아주대 2025학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-M25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def same(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    oks = []
    a = sp.symbols("a", positive=True)
    # (1-1): A 를 x=a 기준으로 나눠 직접 적분 (위쪽 경계: x<a 에서 x^2, x>a 에서 ax; 아래 x^3)
    S = sp.integrate(x**2 - x**3, (x, 0, a)) + sp.integrate(a * x - x**3, (x, a, sp.sqrt(a)))
    S1 = sp.integrate(a * x - x**3, (x, a, sp.sqrt(a)))
    lim = sp.limit(S / S1, a, 0, "+")
    oks.append(lim == sp.sympify(md_answer(1, "1-1")))
    sol = [r for r in sp.solve(sp.Eq(S, sp.Rational(1, 24)), a) if r.is_real and 0 < r < 1]
    oks.append(sol == [sp.sympify(md_answer(1, "1-2"))])
    # (1-2)(1): l1 ⊥ OP (기울기 -1/a), l2 기울기 2a, 예각 φ(a) 최소
    import numpy as np
    m1, m2 = -1 / a, 2 * a
    phi = sp.atan(sp.Abs((m1 - m2) / (1 + m1 * m2)))
    phif = sp.lambdify(a, phi, "numpy")
    grid = np.linspace(1e-3, 1 - 1e-3, 200001)
    amin = grid[int(np.argmin(phif(grid)))]
    oks.append(abs(amin - float(sp.sympify(md_answer(1, "2-1")))) < 1e-4)
    # (1-2)(2) [증명-수치확인]: 0<a<1 격자 1000점에서 점이 원 밖
    out = all(float((v + v**2)**2 + (v + v**2)**6 - (v**2 + v**4)) > 0 for v in np.linspace(1e-3, 1 - 1e-3, 1000))
    oks.append(out)
    # (1-2)(3): T(a) = A ∩ 원(반지름^2 = a^2+a^4) 넓이를 수치적분, T/a^m
    import mpmath as mp
    mp.mp.dps = 40

    def T(av):
        av = mp.mpf(av); R = mp.sqrt(av**2 + av**4)
        def h(xx):
            top = min(xx**2, av * xx); bot = xx**3
            if xx >= R:
                return mp.mpf(0)
            top = min(top, mp.sqrt(R**2 - xx**2))
            return max(mp.mpf(0), top - bot)
        return mp.quad(h, [0, av, R])
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "2-3").replace(" ", "").split(";"))
    mm, L = int(parts["m"]), mp.mpf(str(sp.N(sp.sympify(parts["limit"]), 40)))
    r1, r2 = T("1e-3") / mp.mpf("1e-3")**mm, T("1e-5") / mp.mpf("1e-5")**mm
    lower = T("1e-5") / mp.mpf("1e-5")**(mm - 1)            # m-1 이면 → 0
    upper = T("1e-5") / mp.mpf("1e-5")**(mm + 1)            # m+1 이면 → ∞
    oks.append(abs(r2 - L) < 1e-4 and abs(r2 - L) < abs(r1 - L) and lower < 1e-4 and upper > 1e4)
    report(1, all(oks), f"lim={lim} a(S=1/24)={sol} a*≈{amin:.5f} [증명-수치확인] T/a^{mm}≈{mp.nstr(r2, 10)} {oks}")


def _pairs(seq):
    vals = set(seq); out = set()
    for s_ in vals:
        for t_ in vals:
            if s_ == t_:
                continue
            want, k = (s_, t_, s_, t_), 0
            for v in seq:
                if v == want[k]:
                    k += 1
                    if k == 4:
                        out.add((s_, t_)); break
    return out


def p02():
    oks = []
    import itertools
    f9 = [4, 2, 3, 3, 2, 4, 4, 3, 1]          # 문제 2-1 그림(300dpi 대조)
    P = _pairs(f9)
    md = sp.sympify(md_answer(2, "1-1"))
    oks.append(sp.FiniteSet(*[sp.Tuple(*q) for q in P]) == md)
    # (2-1)(2) [증명-수치확인]: 모범답안의 g 를 n=9..80 에서 직접 확인
    ok = True
    for n_ in range(9, 81):
        g = f9[:6] + [4] * (n_ - 8) + [f9[7], f9[8]]
        ok &= len(g) == n_ and _pairs(g) == P and g[n_ - 3] == 4
    oks.append(ok)
    # (2-2)(1): m=10 전수(첫 값 1로 고정 — 값의 이름 바꾸기 대칭) 불가, m=11 예 존재
    full = {(s_, t_) for s_ in range(1, 5) for t_ in range(1, 5) if s_ != t_}

    def has_all(seq):
        for s_, t_ in full:
            want, k = (s_, t_, s_, t_), 0
            for v in seq:
                if v == want[k]:
                    k += 1
                    if k == 4:
                        break
            if k < 4:
                return False
        return True
    m10 = any(has_all((1,) + rest) for rest in itertools.product(range(1, 5), repeat=9))
    m11 = has_all((1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3))
    oks.append((not m10) and m11 and int(sp.sympify(md_answer(2, "2-1"))) == 11)
    # (2-2)(2) [증명-수치확인]: m≤10 전수(순환 대칭 1→2→3→4→1 로 첫 값 1 고정)에서 반복쌍 집합이 목표와 같은 f 없음
    target = {(1, 2), (2, 3), (3, 4), (4, 1)}
    found = any(_pairs((1,) + rest) == target for mm in range(1, 11) for rest in itertools.product(range(1, 5), repeat=mm - 1))
    oks.append(not found)
    # (2-3): a_n 을 n≤7 전수로 세고, 그 값과 일치하는 닫힌 식으로 a_n/n>2025 의 최소 n
    def a_brute(n_):
        c = 0
        for seq in itertools.product(range(n_), repeat=n_):
            if len(set(seq)) <= 2 and not _pairs(seq):
                c += 1
        return c
    nn = sp.symbols("nn", integer=True, positive=True)
    a_formula = nn + nn * (nn - 1) * (sp.binomial(nn - 1, 1) + sp.binomial(nn - 1, 2))
    brute = [a_brute(k) for k in range(1, 8)]
    oks.append(brute == [int(a_formula.subs(nn, k)) for k in range(1, 8)])
    nmin = next(k for k in range(1, 200) if a_formula.subs(nn, k) / k > 2025)
    oks.append(nmin == int(sp.sympify(md_answer(2, "3"))))
    report(2, all(oks), f"pairs={sorted(P)} m10={m10} [증명-수치확인 2-1(2) n≤80, 2-2(2) m≤10] a_1..7={brute} n={nmin} {oks}")


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
