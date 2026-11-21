#!/usr/bin/env python3
"""KU-G25A (고려대 2025학년도 수시 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import sympy as sp

PREFIX = "KU-G25A"
WIDTH = 2
RESULTS = []
x, y, t = sp.symbols("x y t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _build_f(basis):
    """f(0)=0, f(1)=1 인 다항식 족에서 (라)·(마)(g'(1/2)=0, g(1/2)=61/48)를 만족하도록 계수를 푼다."""
    cs = sp.symbols(f"c0:{len(basis)}")
    f = x**2 + sum(c * b for c, b in zip(cs, basis))
    half = sp.Rational(1, 2)
    gx = sp.integrate(1 - f.subs(x, t), (t, 0, x)) + 3 * sp.integrate(f.subs(x, t), (t, x, 1))
    eqs = [sp.integrate(x * sp.diff(f, x), (x, 0, half)) - sp.Rational(13, 192),
           sp.diff(gx, x).subs(x, half), gx.subs(x, half) - sp.Rational(61, 48)]
    sol = sp.solve(eqs, cs, dict=True)[0]
    return sp.expand(f.subs(sol))


def p01():
    import mpmath as mp
    fams = [[x * (1 - x), x**2 * (1 - x), x**3 * (1 - x)],
            [x * (1 - x), x**3 * (1 - x), x**4 * (1 - x)]]
    vals, okc = [], True
    for basis in fams:
        f = _build_f(basis)
        fp = sp.diff(f, x)
        # 조건 (다) 확인: f'>0, f<=x (격자 2001점)
        grid = [sp.Rational(i, 2000) for i in range(2001)]
        fl, fpl = sp.lambdify(x, f), sp.lambdify(x, fp)
        okc &= all(fpl(float(g)) > 0 for g in grid) and all(fl(float(g)) <= float(g) + 1e-15 for g in grid)
        v1 = f.subs(x, sp.Rational(1, 2))
        r = [rr for rr in sp.Poly(f - sp.Rational(1, 4), x).nroots() if abs(sp.im(rr)) < 1e-12 and 0 <= sp.re(rr) <= 1]
        cut = sp.nsimplify(sp.re(r[0]), [sp.Rational(1, 2)]) if r else sp.Rational(1, 2)
        v2 = sp.integrate(sp.Rational(1, 4) - f, (x, 0, cut)) + sp.integrate(f - sp.Rational(1, 4), (x, cut, 1))
        F = lambda u: abs(fl(fl(u)) - u) * fpl(u)
        v3 = mp.quad(F, [0, 0.5, 1])
        n = 20000
        riem = sum(F(k / n) for k in range(1, n + 1)) / n
        vals.append((sp.nsimplify(v1), sp.nsimplify(v2), v3, riem))
    ok = okc
    for v1, v2, v3, riem in vals:
        ok &= sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
        ok &= sp.simplify(v2 - sp.sympify(md_answer(1, "2"))) == 0
        ok &= abs(v3 - float(sp.sympify(md_answer(1, "3")))) < 1e-9 and abs(riem - v3) < 1e-3
    # 대표 오답: 절댓값을 그냥 벗김 → ∫f - 1/4
    report(1, ok, f"조건(다) 충족={okc} f(1/2)={vals[0][0]} ∫|f-1/4|={vals[0][1]} 급수={float(vals[0][2]):.12f} (두 함수족 일치)")


def p02():
    pi = sp.pi
    random.seed(1)
    # (2-1) [증명-수치확인] 임의 기함수 f, g=f'+C(=g'=f'' 의 일반해) → g(x)=g(-x)
    ok1 = True
    for _ in range(20):
        f = sum(random.randint(-5, 5) * x**(2 * k + 1) for k in range(4))
        C = random.randint(-5, 5)
        g = sp.integrate(sp.diff(f, x, 2), x) + C
        ok1 &= all(abs(float((g - g.subs(x, -x)).subs(x, sp.Rational(i, 500)))) < 1e-12 for i in range(-499, 500))
    # (2-2) 조건을 만족하는 구체적 함수족에서 직접 적분
    res = []
    for k1, k2 in [(1, 0), (2, 3), (-1, 5)]:
        fp = (1 - x**2) * (k1 + k2 * x**2)            # f' 우함수, f'(±1)=0 → f 기함수 가능(f=∫_0^x f')
        f = sp.integrate(fp, (x, 0, x))
        assert sp.simplify(f + f.subs(x, -x)) == 0
        C = sp.symbols("C")
        g = fp + sp.solve(fp.subs(x, 1) + C - pi, C)[0]  # g(1)=π
        a = sp.symbols("a0:5")
        h = sum(a[i] * x**i for i in range(5))
        sol = sp.solve([h.subs(x, 1) - 2 * pi, h.subs(x, -1) - 1,
                        sp.integrate(sp.diff(f, x) * sp.diff(h, x), (x, -1, 1)) - pi], [a[0], a[1], a[3]], dict=True)[0]
        hs = h.subs(sol).subs({a[2]: random.randint(-3, 3), a[4]: random.randint(-3, 3)})
        hp = sp.diff(hs, x)
        val = sp.integrate(g * (hp + hp.subs(x, -x)), (x, 0, 1))
        res.append(sp.simplify(val))
    ok2 = all(sp.simplify(v - sp.sympify(md_answer(2, "2"))) == 0 for v in res)
    report(2, ok1 and ok2, f"[증명-수치확인] (2-1) 기함수 20개 g(x)=g(-x) 성립, (2-2) 함수족 3개 값={res}")


def p03():
    O = sp.Matrix([0, 0]); A = sp.Matrix([2, 0])
    B = 3 * sp.Matrix([sp.cos(sp.pi / 3), sp.sin(sp.pi / 3)])
    M = O + (A - O) / 3; N = A + (B - A) / 3; Bp = O + 2 * (B - O) / 3
    MN = N - M
    a_, b_ = A - O, B - O
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    av, bv = sp.symbols("a b")
    vexpr = sp.sympify(parts["vec"], locals={"a": av, "b": bv})
    cA, cB = sp.Poly(sp.expand(vexpr), av, bv).coeff_monomial(av), sp.Poly(sp.expand(vexpr), av, bv).coeff_monomial(bv)
    ok1 = sp.simplify(cA * a_ + cB * b_ - MN) == sp.zeros(2, 1)
    ok1 &= sp.simplify(MN.norm() - sp.sympify(parts["len"])) == 0
    s = sp.symbols("s", real=True)
    H = M + s * MN
    ss = sp.solve((Bp - H).dot(MN), s)[0]
    MH = sp.simplify((H.subs(s, ss) - M).norm())
    ok2 = sp.simplify(MH - sp.sympify(md_answer(3, "2"))) == 0
    report(3, ok1 and ok2, f"좌표 재계산: |MN|={sp.simplify(MN.norm())} MH={MH} (H 매개변수 s={sp.simplify(ss)})")


def p04():
    def probs(xv, yv):
        m = sp.Min(xv, yv)
        PB1 = PB2 = 0
        for wb in (1, 0):            # 야구 승패
            for wk in (1, 0):        # 농구 승패 (독립)
                p = (xv if wb else 1 - xv) * (yv if wk else 1 - yv)
                if wb or wk:
                    PB1 += p * xv; PB2 += p * yv
                else:
                    PB1 += p * m; PB2 += p * m
        return sp.nsimplify(PB1), sp.nsimplify(PB2)
    expr = sp.sympify(md_answer(4, "1"), locals={"x": x, "y": y})
    random.seed(4)
    ok1 = True
    for _ in range(40):
        xv, yv = sp.Rational(random.randint(0, 100), 100), sp.Rational(random.randint(0, 100), 100)
        ok1 &= sp.simplify(probs(xv, yv)[0] - expr.subs({x: xv, y: yv})) == 0
    # (4-2) x=1/3, y in [1/3,1]
    yy = sp.symbols("yy", real=True)
    xv = sp.Rational(1, 3)
    m = xv  # x<=y
    PA = 1 - (1 - xv) * (1 - yy)
    PB1 = PA * xv + (1 - PA) * m
    PB2 = PA * yy + (1 - PA) * m
    sol = sp.solve_univariate_inequality(sp.expand(PB1 - (1 - PB2)) <= 0, yy, relational=False)
    sol = sol.intersect(sp.Interval(sp.Rational(1, 3), 1))
    md = sp.sympify(md_answer(4, "2"))
    ok2 = sp.simplify(sol.inf - md.inf) == 0 and sp.simplify(sol.sup - md.sup) == 0 and sol.left_open == md.left_open and sol.right_open == md.right_open
    # 격자 직접 확인(전수 열거 확률로)
    for i in range(0, 1001):
        yv = sp.Rational(1, 3) + sp.Rational(2, 3) * sp.Rational(i, 1000)
        p1, p2 = probs(xv, yv)
        ok2 &= ((p1 <= 1 - p2) == bool(md.contains(yv)))
    report(4, ok1 and ok2, f"전수 열거 40점 일치, (4-2) 범위={sol}")


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
