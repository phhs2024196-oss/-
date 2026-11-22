#!/usr/bin/env python3
"""INHA-M26 (2026학년도 인하대 자연계열 논술 모의고사, 일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML 정답과 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "INHA-M26"
WIDTH = 2
RESULTS = []
KIT = pathlib.Path(__file__).resolve().parents[1]
x, t, s = sp.symbols("x t s", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int, sub: str) -> str:
    it = yaml.safe_load((KIT / f"items/INHA/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    return next(p for p in it["subparts"] if str(p["id"]) == sub)["answer"]["value"]


def tex(v: str):
    """간단한 LaTeX(\\dfrac, \\sqrt, \\pi, \\ln) → sympy."""
    v = v.replace("\\dfrac", "\\frac")
    v = v.replace("\\pi", "{pi}")
    while "\\frac" in v:
        w = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", v)
        w = re.sub(r"\\frac(\w)(\w)", r"((\1)/(\2))", w)
        w = re.sub(r"\\frac(\w)\{([^{}]*)\}", r"((\1)/(\2))", w)
        w = re.sub(r"\\frac\{([^{}]*)\}(\w)", r"((\1)/(\2))", w)
        if w == v:
            raise ValueError(f"tex 변환 실패: {v}")
        v = w
    v = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", v)
    v = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", v)
    v = v.replace("\\pi", " pi ").replace("\\ln", " log ").replace("{", "(").replace("}", ")")
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    return parse_expr(v, local_dict={"pi": sp.pi, "log": sp.log, "sqrt": sp.sqrt},
                      transformations=standard_transformations + (implicit_multiplication_application, convert_xor))


def p01():
    """(1-1) 교점 x=5 (1-2) h'(t)>0 수치 확인 (1-3) (h^-1)'(5)=46/71."""
    f = x**4 + 17 * x**2 + 3
    fp = sp.diff(f, x)
    # (1-1)
    sol = [r for r in sp.solve(sp.Eq(f, fp.subs(x, 3) * x + 3), x) if r.is_real and r > 3]
    ok1 = len(sol) == 1 and sol[0] == sp.Integer(ans(1, "1"))
    # (1-2) h(t): 근 중 t보다 큰 실근 — 수치로 증가 확인 [증명-수치확인]
    def h(tv):
        poly = sp.Poly(sp.expand((f - 3 - fp.subs(x, tv) * x) / x), x)
        rr = [complex(r) for r in sp.Poly(poly, x).nroots()]
        rr = [r.real for r in rr if abs(r.imag) < 1e-9 and r.real > float(tv) + 1e-9]
        assert len(rr) == 1
        return rr[0]
    ts = [sp.Rational(k, 20) for k in range(1, 201)]
    hs = [h(tv) for tv in ts]
    ok2 = all(b > a for a, b in zip(hs, hs[1:]))
    # (1-3) 독립: h 를 수치미분해 역함수 미분계수 = 1/h'(3)
    e = sp.Rational(1, 10**6)
    hp3 = (h(3 + e) - h(3 - e)) / (2 * float(e))
    # 기호: 관계식 f(s)-f(0)=s f'(u) 에서 u'(s)
    u = sp.Function("u")
    rel = f.subs(x, s) - 3 - s * fp.subs(x, u(s))
    du = sp.solve(sp.diff(rel, s), sp.diff(u(s), s))[0]
    val = sp.nsimplify(du.subs(u(s), 3).subs(s, 5))
    ok3 = sp.simplify(val - tex(ans(1, "3"))) == 0 and abs(1 / hp3 - float(val)) < 1e-6
    report(1, ok1 and ok2 and ok3, f"(1-1) x={sol} (1-2) h 증가 n=200 [증명-수치확인] (1-3) {val}, 1/h'(3)={1/hp3:.6f}")


def p02():
    """(2-1) b (2-2) a,b 최소 (2-3) g'(-5),g'(-2),g'(1)."""
    a, b = sp.symbols("a b", real=True)
    f1 = x + a * sp.sin(sp.pi * x)
    f2 = 2 * x + b * sp.sin(2 * sp.pi * x)
    cond = sp.Eq(sp.diff(f1, x).subs(x, 0), sp.diff(f2, x).subs(x, 0))
    b1 = sp.solve(cond.subs(a, sp.Rational(1, 4)), b)[0]
    ok1 = sp.simplify(b1 - (sp.Rational(1, 8) - 1 / (2 * sp.pi))) == 0
    # (2-2) 증가 조건 |a|,|b|<=1/pi, 직선 a-2b=1/pi 위 a^2+b^2 최소(라그랑주 대신 대입)
    aa = sp.solve(cond, a)[0]
    bb = sp.solve(sp.diff(aa**2 + b**2, b), b)[0]
    a2, b2 = sp.simplify(aa.subs(b, bb)), bb
    inrange = abs(float(a2)) <= 1 / float(sp.pi) and abs(float(b2)) <= 1 / float(sp.pi)
    ok2 = sp.simplify(a2 - 1 / (5 * sp.pi)) == 0 and sp.simplify(b2 + 2 / (5 * sp.pi)) == 0 and inrange
    # 증가성 수치 확인
    F = sp.Piecewise((f2.subs({b: b2}), x < 0), (f1.subs({a: a2}), True))
    Fp = sp.lambdify(x, sp.diff(F, x))
    import numpy as np
    grid = np.linspace(-6, 6, 24001)
    okmono = min(Fp(v) for v in grid) >= -1e-12
    # (2-3) f(x)=c 를 수치로 풀어 g'(c)=1/f'(x)
    Fn = sp.lambdify(x, F)
    from scipy.optimize import brentq
    vals = []
    for c in (-5, -2, 1):
        xr = brentq(lambda v: Fn(v) - c, -10, 10)
        vals.append(sp.nsimplify(1 / Fp(xr), tolerance=1e-10, rational=True))
    ok3 = vals == [sp.Rational(5, 14), sp.Rational(5, 6), sp.Rational(5, 4)]
    report(2, ok1 and ok2 and okmono and ok3, f"(2-1) b={b1} (2-2) a={a2}, b={b2}, 증가 확인 (2-3) {vals}")


def p03():
    """(3-1) ln2 (3-2) 대칭 항등식 (3-3) (π/8)ln2."""
    th = sp.symbols("theta", real=True)
    f = 1 / (sp.cos(th) * (sp.sin(th) + sp.cos(th)))
    i1 = sp.integrate(sp.simplify(sp.sec(th) ** 2 / (1 + sp.tan(th))), (th, 0, sp.pi / 4))
    ok1 = sp.simplify(i1 - sp.log(2)) == 0 and abs(sp.N(sp.Integral(f, (th, 0, sp.pi / 4)), 15) - sp.N(sp.log(2), 15)) < 1e-12
    import random
    random.seed(1)
    ok2 = all(abs(sp.N((f - f.subs(th, sp.pi / 4 - th)).subs(th, random.uniform(0, 0.785)), 20)) < 1e-15 for _ in range(20))
    ok2 = ok2 and sp.simplify(sp.expand_trig(f - f.subs(th, sp.pi / 4 - th))) == 0
    i3 = sp.N(sp.Integral(sp.log(1 + x) / (1 + x**2), (x, 0, 1)), 30)
    ok3 = abs(i3 - sp.N(sp.pi / 8 * sp.log(2), 30)) < 1e-25
    report(3, ok1 and ok2 and ok3, f"(3-1) {i1} (3-2) 무작위 20점·기호 [증명-수치확인] (3-3) {i3:.15f}")


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
