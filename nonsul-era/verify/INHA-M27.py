#!/usr/bin/env python3
"""INHA-M27 (2027학년도 인하대 자연계열 논술 모의고사, 일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 YAML 정답과 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "INHA-M27"
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
    """(1-1) S(t) 접선 삼각형 (1-2) 최솟값 8/9."""
    tp = sp.symbols("tp", positive=True)
    f = x**3 + 1 / x
    tan = sp.diff(f, x).subs(x, tp) * (x - tp) + f.subs(x, tp)
    xA = sp.solve(sp.Eq(tan, 0), x)[0]
    S = sp.simplify(sp.Rational(1, 2) * (tp - xA) * f.subs(x, tp))
    yS = ans(1, "1").split("=", 1)[1]
    ok1 = sp.simplify(S - tex(yS).subs(sp.Symbol("t"), tp)) == 0
    t0 = 3 ** sp.Rational(-1, 4)
    u = sp.symbols("u", positive=True)
    T = sp.simplify(S.subs(tp, u ** sp.Rational(1, 4)))          # u=t^4
    crit = [c for c in sp.solve(sp.numer(sp.together(sp.diff(T, u))), u) if c > sp.Rational(1, 3)]
    mn = sp.simplify(T.subs(u, crit[0]))
    import numpy as np
    Sf = sp.lambdify(tp, S)
    grid = np.linspace(float(t0) + 1e-4, 10, 200001)
    ok2 = len(crit) == 1 and mn == tex(ans(1, "2")) and abs(Sf(grid).min() - float(mn)) < 1e-8
    report(1, ok1 and ok2, f"(1-1) S={S} (1-2) 최소 {mn} at u=t^4={crit}")


def p02():
    """(2-1) 부등식 수치 (2-2) 2/π (2-3) e^(2/π)."""
    import numpy as np
    xs = np.linspace(0, 50, 500001)
    ok1 = bool(np.all(xs - xs**2 <= np.log1p(xs) + 1e-15) and np.all(np.log1p(xs) <= xs + 1e-15))
    k, n = sp.symbols("k n", positive=True, integer=True)
    L1 = sp.integrate(sp.sin(sp.pi * x / 2), (x, 0, 1))
    ok2 = sp.simplify(L1 - tex(ans(2, "2"))) == 0
    def lnP(N):
        kk = np.arange(1, N + 1)
        return np.sum(np.log1p(np.sin(kk * np.pi / (2 * N)) / N))
    vals = [lnP(N) for N in (10**3, 10**5, 10**6)]
    target = float(sp.E ** (2 / sp.pi))
    yv = ans(2, "3")
    ok3 = abs(np.exp(vals[-1]) - target) < 1e-5 and sp.simplify(tex(yv.replace("e^", "E^")) - sp.E ** (2 / sp.pi)) == 0
    report(2, ok1 and ok2 and ok3, f"(2-1) [증명-수치확인] 0≤x≤50 (2-2) {L1} (2-3) P_n(10^6)={np.exp(vals[-1]):.8f} vs e^(2/π)={target:.8f}")


def p03():
    """(3-1) 극값·y절편 (3-2) a=1,-2 (3-3) a=0,-1 — 좌·우 평균변화율 극한."""
    a, h = sp.symbols("a h", real=True)
    f = lambda z: 2 * z**3 + 3 * z**2 - 12 * z + 1
    F = f(x - 3)
    cr = sp.solve(sp.diff(F, x), x)
    ok1 = sorted(cr) == [1, 4] and F.subs(x, 1) == 21 and F.subs(x, 4) == -6 and f(-3) == 10
    def onesided(G):   # 각 조각이 다항식 → 조각의 미분계수 = 평균변화율의 한쪽 극한
        Gl, Gr = G
        return sp.expand(sp.diff(Gl, x).subs(x, 0)), sp.expand(sp.diff(Gr, x).subs(x, 0))
    L, R = onesided((f(-x + a), f(x + a)))
    s2 = sorted(sp.solve(sp.Eq(L, R), a))
    L3, R3 = onesided((f(-x + a) - 16 * x, f(x + a) + 8 * x))
    s3 = sorted(sp.solve(sp.Eq(L3, R3), a))
    ok2 = s2 == [-2, 1]
    ok3 = s3 == [-1, 0]
    report(3, ok1 and ok2 and ok3, f"(3-1) 극값 x={cr} y절편 {f(-3)} (3-2) a={s2} (3-3) a={s3}")


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
