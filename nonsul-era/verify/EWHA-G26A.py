#!/usr/bin/env python3
"""EWHA-G26A (이화여대 2026학년도 수시 논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
증명 소문항은 수치 반례 탐색 [증명-수치확인]."""
import random
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "EWHA-G26A"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1) 일반 삼차식 + 조건 (가)(나)로 계수 결정
    c = sp.symbols("c0:4")
    f = sum(c[i] * x**i for i in range(4))
    g = sp.sin(sp.pi * x / 2)
    eqs = sp.Poly(f.subs(x, -x) + f, x).all_coeffs() + [f.subs(x, 1) - g.subs(x, 1), f.subs(x, 2) - g.subs(x, 2)]
    sol = sp.solve(eqs, c, dict=True)
    assert len(sol) == 1
    f = sp.expand(f.subs(sol[0]))
    ok1 = sp.simplify(f - sp.sympify(md_answer(1, "1"), locals={"x": x})) == 0
    # (2) [증명-수치확인] sin x >= 2x/pi on [0, pi/2]
    ok2 = all(mp.sin(t) - 2 * t / mp.pi >= -1e-15 for t in mp.linspace(0, mp.pi / 2, 2001))
    # (3) h' 의 근이 (0,1)에 정확히 하나: 부호 변화 개수 + h''>0
    h = f - g
    h1 = sp.lambdify(x, sp.diff(h, x), "mpmath")
    h2 = sp.lambdify(x, sp.diff(h, x, 2), "mpmath")
    pts = mp.linspace(0, 1, 4001)
    vals = [h1(t) for t in pts]
    changes = sum(1 for u, v in zip(vals, vals[1:]) if u * v < 0)
    ok3 = changes == 1 and all(h2(t) > 0 for t in pts[1:-1]) and h1(0) < 0 < h1(1)
    cstar = mp.findroot(h1, 0.5)
    # (4) [증명-수치확인] h <= 0 on [0,1]
    hf = sp.lambdify(x, h, "mpmath")
    ok4 = all(hf(t) <= 1e-15 for t in pts)
    report(1, ok1 and ok2 and ok3 and ok4, f"f={f} c≈{mp.nstr(cstar, 10)} [증명-수치확인](2)(3)(4)")


def p02():
    # [증명-수치확인] 무작위 m<n, 서로 다른 b_k 로 (1)(2)(3) 확인 — sympy apart 로 c_k 독립 계산
    random.seed(2026)
    ok = True
    cases = 0
    for n in range(1, 7):
        for m in range(0, n):
            for _ in range(3):
                bs = sorted(random.sample(range(-9, 10), n))
                bs = [sp.Rational(b) for b in bs]
                coeffs = [sp.Rational(random.randint(-9, 9)) for _ in range(m + 1)]
                if coeffs[-1] == 0:
                    coeffs[-1] = 1
                P = sum(coeffs[i] * x**i for i in range(m + 1))
                Q = sp.prod([x - b for b in bs])
                Qp = sp.diff(Q, x)
                # (1)
                ok &= sp.simplify(Qp / Q - sum(1 / (x - b) for b in bs)) == 0
                # (2) apart 의 각 항 분자 c_k
                ap = sp.apart(P / Q, x)
                cs = []
                for b in bs:
                    term = [t for t in sp.Add.make_args(ap) if sp.simplify(sp.denom(sp.together(t)).subs(x, b)) == 0]
                    ck = sp.limit((x - b) * sum(term), x, b) if term else 0
                    cs.append(ck)
                    ok &= Qp.subs(x, b) != 0 and sp.simplify(ck - P.subs(x, b) / Qp.subs(x, b)) == 0
                # (3)
                expect = coeffs[m] if m == n - 1 else 0
                ok &= sp.simplify(sum(cs) - expect) == 0
                cases += 1
    report(2, ok, f"[증명-수치확인] 무작위 {cases}개 (m,n,b,P)에서 (1)(2)(3) 등식 성립")


def p03():
    # [증명-수치확인] 여러 연속함수 f 에 대해 (1)~(4)
    t = sp.symbols("t", real=True)
    ok = True
    tests = [x**2 - 1, sp.exp(x), sp.sin(3 * x) + x, sp.cos(x) * x**3 - 2, sp.Abs(x - sp.Rational(1, 3))]
    for f in tests:
        H = sp.integrate(f.subs(x, t), (t, 0, x)) - sp.integrate(f.subs(x, t), (t, x, 1))
        # (1)/(4) 핵심: H' = 2f (미분가능 → 연속), H = 2F + A
        Hn = sp.lambdify(x, H, "mpmath")
        fn = sp.lambdify(x, f, "mpmath")
        for x0 in (-0.7, 0.1, 0.9, 1.6):
            ok &= abs(mp.diff(Hn, x0) - 2 * fn(x0)) < 1e-8
        # (2)
        lhs = mp.quad(Hn, [0, 1 / 3, 1])
        rhs = mp.quad(lambda s: (1 - 2 * s) * fn(s), [0, 1 / 3, 1])
        ok &= abs(lhs - rhs) < 1e-10
    # (3) 조건 ∫f = 2∫xf 를 만족하도록 f = p(x) + λx 로 맞춘 뒤 c 의 존재 확인
    lam = sp.symbols("lam")
    for p in (sp.exp(x), x**3 - x, sp.sin(5 * x)):
        f = p + lam * x
        L = sp.solve(sp.integrate(f, (x, 0, 1)) - 2 * sp.integrate(x * f, (x, 0, 1)), lam)[0]
        f = f.subs(lam, L)
        fn = sp.lambdify(x, f, "mpmath")
        phi = lambda cc: mp.quad(fn, [0, cc]) - mp.quad(fn, [cc, 1])
        grid = [phi(mp.mpf(i) / 400) for i in range(1, 400)]
        ok &= any(abs(v) < 1e-12 for v in grid) or any(u * v < 0 for u, v in zip(grid, grid[1:]))
    # (4) H ≡ 0 이면 f ≡ 0: 0 이 아닌 f 에 대해 H 가 항등적으로 0 이 아님을 확인
    for f in tests:
        H = sp.integrate(f.subs(x, t), (t, 0, x)) - sp.integrate(f.subs(x, t), (t, x, 1))
        Hn = sp.lambdify(x, H, "mpmath")
        ok &= max(abs(Hn(s)) for s in mp.linspace(-0.99, 1.99, 50)) > 1e-6
    report(3, ok, "[증명-수치확인] H'=2f, ∫H=∫(1-2x)f, 조건 하 c 존재, f≠0 ⇒ H≢0")


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
