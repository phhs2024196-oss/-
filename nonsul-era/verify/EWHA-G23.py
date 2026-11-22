#!/usr/bin/env python3
"""EWHA-G23 (이화여대 2023학년도 수시 논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "EWHA-G23"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def f_tent(x: float) -> float:
    if x < 0:
        return 0.0
    n = math.floor(x / 2)
    return 1 - abs(x - 2 * n - 1)


def p01():
    a, n = sp.symbols("a n", positive=True)
    X = sp.symbols("X", real=True)
    loc = {"a": a, "n": n}

    def tri_area(nn):
        # 교점: 왼쪽 변 y=x-2n, 오른쪽 변 y=2n+2-x 와 y=ax, 꼭짓점 (2n+1,1) — 신발끈 공식
        x1 = sp.solve(sp.Eq(a * X, X - 2 * nn), X)[0]
        x2 = sp.solve(sp.Eq(a * X, 2 * nn + 2 - X), X)[0]
        P = [(x1, a * x1), (2 * nn + 1, 1), (x2, a * x2)]
        s = sum(P[i][0] * P[(i + 1) % 3][1] - P[(i + 1) % 3][0] * P[i][1] for i in range(3))
        return sp.simplify(sp.Abs(sp.factor(s)) / 2)

    def signed_area(nn):
        x1 = sp.solve(sp.Eq(a * X, X - 2 * nn), X)[0]
        x2 = sp.solve(sp.Eq(a * X, 2 * nn + 2 - X), X)[0]
        P = [(x1, a * x1), (x2, a * x2), (2 * nn + 1, 1)]   # 반시계(꼭짓점이 직선 위)
        return sp.factor(sum(P[i][0] * P[(i + 1) % 3][1] - P[(i + 1) % 3][0] * P[i][1] for i in range(3)) / 2)

    S0 = tri_area(0)
    ok1 = all(abs(float((S0 - sp.sympify(md_answer(1, "1"), locals=loc)).subs(a, v))) < 1e-12 for v in (0.1, 0.37, 0.8))
    # (2) 범위: 수치로 교점 개수 세기 (n=0..4, a 격자)
    d2 = kv(md_answer(1, "2"))
    up = sp.sympify(d2["a_upper"], locals=loc)
    Sn_md = sp.sympify(d2["S_n"], locals=loc)
    ok2 = True
    import numpy as np
    for nn in range(5):
        for av in np.linspace(0.001, 0.999, 400):
            xs = np.linspace(2 * nn, 2 * nn + 2, 20001)
            g = np.array([f_tent(x) for x in xs]) - av * xs
            sgn = np.sign(g)
            cnt = int(np.sum(sgn[:-1] * sgn[1:] < 0)) + int(np.sum(g == 0))
            two = cnt >= 2
            if two != (av < float(up.subs(n, nn))) and abs(av - float(up.subs(n, nn))) > 3e-3:
                ok2 = False
        Sn = tri_area(nn)
        for av in (0.5 / (2 * nn + 1), 0.9 / (2 * nn + 1)):
            if abs(float(Sn.subs(a, av) - Sn_md.subs({n: nn, a: av}))) > 1e-12:
                ok2 = False
    # (3) n 범위: 직접 조건 [2n,2n+2] 두 점, [2n+2,2n+4] 꼭짓점이 직선 위/아래
    d3 = kv(md_answer(1, "3"))
    lo, hi = sp.sympify(d3["n_lower"], locals=loc), sp.sympify(d3["n_upper"], locals=loc)
    ok3 = True
    for av in (0.003, 0.0171, 0.02, 0.05, 0.11):
        direct = [k for k in range(1, 400) if av * (2 * k + 1) < 1 and av * (2 * k + 3) >= 1]
        frm = [k for k in range(1, 400) if float(lo.subs(a, av)) - 1e-12 <= k < float(hi.subs(a, av))]
        ok3 &= direct == frm
    na_vals = [math.ceil((1 - 3 * av) / (2 * av) - 1e-9) * av for av in (1e-4, 1e-6, 1e-8)]
    lim3 = sp.limit(lo * a, a, 0, "+")
    ok3 &= lim3 == sp.limit(hi * a, a, 0, "+") == sp.sympify(d3["limit"]) and abs(na_vals[-1] - 0.5) < 1e-6
    # (4) 실제 n 과 기하로 구한 S_k 합의 평균 → 극한
    Sgeo = signed_area(n)                      # 기하로 구한 S_n (md 정답 미사용)
    Sk = sp.lambdify((a, n), Sgeo)
    vals = []
    for av in (1e-3, 1e-4, 1e-5):
        nn = math.ceil((1 - 3 * av) / (2 * av) - 1e-12)
        assert av * (2 * nn + 1) < 1 <= av * (2 * nn + 3)
        vals.append(sum(Sk(av, k) for k in range(nn + 1)) / nn)
    kk = sp.symbols("k", integer=True, nonnegative=True)
    closed = sp.simplify(sp.summation(Sgeo.subs(n, kk), (kk, 0, n)) / n)
    u = sp.symbols("u", positive=True)  # u = n*a
    lim4 = sp.limit(sp.limit(sp.simplify(closed.subs(n, u / a)), a, 0, "+").subs(u, sp.Rational(1, 2)), u, sp.Rational(1, 2))
    ans4 = sp.sympify(md_answer(1, "4"))
    ok4 = sp.simplify(lim4 - ans4) == 0 and abs(vals[-1] - float(ans4)) < 1e-4
    report(1, ok1 and ok2 and ok3 and ok4,
           f"S0={sp.simplify(S0)} 범위·S_n 수치일치={ok2} lim na={lim3} lim avg={lim4} (수치 {vals[-1]:.6f})")


def p02():
    import numpy as np

    def count(m, b):
        # |x(x-2)| = m x + b 의 서로 다른 실근 개수 (두 이차방정식, 구간 조건)
        roots = set()
        for sgn in (1, -1):
            # sgn*(x^2-2x) = m x + b
            r = np.roots([sgn, -2 * sgn - m, -b])
            for x in r:
                if abs(x.imag) < 1e-6:
                    x = x.real
                    val = x * (x - 2)
                    if (sgn == 1 and val >= -1e-12) or (sgn == -1 and val <= 1e-12):
                        roots.add(round(x, 4))
        return len(roots)

    # (1) m=1: b 격자 + 후보점에서 세 점이 되는 b
    cands = sorted(set([round(v, 6) for v in np.linspace(-3, 3, 60001)] + [0.0, 0.25, -2.0]))
    good = [b for b in cands if count(1, b) == 3]
    md1 = sorted(float(sp.sympify(v)) for v in md_answer(2, "1").strip("[]").split(","))
    ok1 = sorted(set(round(b, 6) for b in good)) == [round(v, 6) for v in md1]
    # (2) b>0, m>=0 : 세 점 ⇔ 접함 → 판별식 독립 계산
    m, b, x = sp.symbols("m b x", real=True)
    bt = sp.solve(sp.discriminant(-x**2 + 2 * x - m * x - b, x), b)[0]
    d2 = kv(md_answer(2, "2"))
    ok2 = sp.simplify(bt - sp.sympify(d2["b"], locals={"m": m})) == 0 and d2["m_range"] == "[0,2)"
    for mv in np.linspace(0, 3, 61):
        bv = float(bt.subs(m, mv))
        three = bv > 0 and count(mv, bv) == 3
        ok2 &= three == (0 <= mv < 2 - 1e-12)
        # 다른 b>0 에서는 세 점 아님
        ok2 &= all(count(mv, bb) != 3 for bb in (bv * 0.7 + 0.01, bv * 1.3 + 0.01))
    # (3) 0<=m<2 에서 두 넓이 차의 부호 → 0 인 b
    def areas(mv):
        bv = (mv - 2) ** 2 / 4
        p = 1 - mv / 2
        al = ((2 + mv) - math.sqrt((2 + mv) ** 2 + 4 * bv)) / 2
        be = ((2 + mv) + math.sqrt((2 + mv) ** 2 + 4 * bv)) / 2
        from scipy.integrate import quad
        h = lambda t: mv * t + bv - abs(t * (t - 2))
        return quad(h, al, p, points=[0])[0], quad(h, p, be, points=[2])[0], bv
    diffs = [(areas(mv)[1] - areas(mv)[0], areas(mv)[2]) for mv in np.linspace(0, 1.99, 200)]
    zeros = [bv for d, bv in diffs if abs(d) < 1e-9]
    ok3 = all(d > 0 for d, bv in diffs[1:]) and len(zeros) == 1 and abs(zeros[0] - float(sp.sympify(md_answer(2, "3")))) < 1e-12
    report(2, ok1 and ok2 and ok3, f"(1) b={good} (2) b={bt} (3) 등넓이 b={zeros}, m>0 에서 S2-S1>0")


def p03():
    n = sp.symbols("n", integer=True, nonnegative=True)
    x = sp.symbols("x", real=True)
    # 합집합의 세로 단면은 가장 가까운 중심 원이 결정 → 단면 적분
    I1 = sp.integrate(2 * sp.sqrt(1 - x**2), (x, -sp.Rational(1, 2), sp.Rational(1, 2)))
    I2 = sp.integrate(2 * sp.sqrt(1 - x**2), (x, sp.Rational(1, 2), 1))
    S = lambda nn: sp.nsimplify(sp.simplify((2 * nn + 1) * I1 + 2 * I2))
    al = sp.sqrt(3) / 4
    be = sp.pi / 6 - sp.sqrt(3) / 4
    A, B = sp.symbols("A B")

    def coeffs(nn):
        Sn = sp.expand(S(nn))
        eqs = sp.Poly(sp.expand(A * al + B * be - Sn), sp.pi, sp.sqrt(3)).coeffs()
        sol = sp.solve(eqs, [A, B], dict=True)[0]
        return (int(sol[A]), int(sol[B]))

    got = [coeffs(k) for k in (0, 1, 2023)]
    md = [tuple(int(v) for v in md_answer(3, s).strip("()").split(",")) for s in ("1", "2", "3")]
    # 몬테카를로 대신 n=1 수치 교차 확인(겹침 직접 계산)
    import numpy as np
    xs = np.linspace(-2, 2, 400001)
    h = np.max([np.sqrt(np.clip(1 - (xs - c) ** 2, 0, None)) for c in (-1, 0, 1)], axis=0)
    num1 = np.trapezoid(2 * h, xs)
    ok = got == md and abs(num1 - float(S(1))) < 1e-6
    report(3, ok, f"(a,b) n=0,1,2023 → {got}, S1 수치={num1:.8f}")


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
