#!/usr/bin/env python3
"""EWHA-G25A (이화여대 2025학년도 수시 논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
PREFIX = "EWHA-G25A"
import re
import sys

import mpmath as mp
import sympy as sp

WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")
x, t = sp.symbols("x t", real=True)


def p01():
    F = sp.Function("f")
    g = F(x) * sp.sin(2 * x)
    # (1) [증명-수치확인] g' = (2f + f' tan2x) cos2x 항등식, cos2x>0 on [0,pi/6]
    ident = sp.simplify(sp.diff(g, x) - (2 * F(x) + sp.diff(F(x), x) * sp.tan(2 * x)) * sp.cos(2 * x)) == 0
    cpos = all(mp.cos(2 * v) > 0 for v in mp.linspace(0, mp.pi / 6, 1001))
    # (2),(3) 조건을 만족하는 서로 다른 f 3개로 수치 확인
    cands = [2 * sp.sqrt(3) + 0 * x,
             2 * sp.sqrt(3) * (1 + (x - sp.pi / 6) / 5),
             2 * sp.sqrt(3) * sp.exp(x - sp.pi / 6)]
    ok2 = ok3 = True
    ans2 = dict(s.split("=") for s in md_answer(1, "2").replace(" ", "").split(";"))
    for fx in cands:
        fp = sp.diff(fx, x)
        cond = all(float((2 * fx + fp * sp.tan(2 * x)).subs(x, v)) > 0 for v in [i * float(sp.pi) / 600 for i in range(101)])
        gl = sp.lambdify(x, fx * sp.sin(2 * x), "mpmath")
        h = lambda y: mp.findroot(lambda u: gl(u) - y, (0, mp.pi / 6), solver="bisect") if 0 < y < 3 else (0 if y == 0 else mp.pi / 6)
        ok2 &= cond and abs(h(0) - float(sp.sympify(ans2["h(0)"]))) < 1e-12 and abs(h(3) - float(sp.sympify(ans2["h(3)"]))) < 1e-12
        Ih = mp.quad(h, [0, 3])
        a = mp.quad(sp.lambdify(x, x * fx * sp.cos(2 * x), "mpmath"), [0, mp.pi / 6])
        b = mp.quad(sp.lambdify(x, x * fp * sp.sin(2 * x), "mpmath"), [0, mp.pi / 6])
        expr = sp.sympify(md_answer(1, "3"))
        ok3 &= abs(Ih - float(expr.subs({sp.Symbol("a"): float(a), sp.Symbol("b"): float(b)}))) < 1e-8
    report(1, ident and cpos and ok2 and ok3, "g' 항등식·cos2x>0 [증명-수치확인], h(0),h(3) 및 ∫h=2a+b 를 f 3종으로 확인")


def p02():
    a = sp.symbols("a", positive=True)
    f0 = sp.Piecewise((a**x - 1, x < 1), (a - a**(x - 1), True))
    I02 = sp.simplify(sp.integrate(a**x - 1, (x, 0, 1)) + sp.integrate(a - a**(x - 1), (x, 1, 2)))
    ok1 = sp.simplify(I02 - sp.sympify(md_answer(2, "1"), locals={"a": a})) == 0
    # (2) 조건 (나)로 직접 계산한 값과 md 식 비교(무작위 a, n, x)
    n = sp.symbols("n", integer=True, positive=True)
    parts = md_answer(2, "2").split(";")
    e1 = sp.sympify(parts[0][:parts[0].index(" (")], locals={"a": a, "n": n, "x": x})
    e2 = sp.sympify(parts[1].strip()[:parts[1].strip().index(" (")], locals={"a": a, "n": n, "x": x})
    import random
    random.seed(1)
    ok2 = True
    for _ in range(40):
        av, nv = random.uniform(1.1, 5), random.randint(1, 6)
        xv = 2 * nv + random.uniform(0, 2)
        direct = av**(-nv) * float(f0.subs({a: av, x: xv - 2 * nv}))
        md = (e1 if xv <= 2 * nv + 1 else e2).subs({a: av, n: nv, x: xv})
        ok2 &= abs(direct - float(md)) < 1e-10
    # (3) S_n 수치(정의에서 직접) → 극한 = a 확인 후 lim=2025 풀기
    def fnum(av, k, xv):                     # 구간 [2k,2k+2] 에서 조건 (나): a^{-k} f(x-2k)
        y = float(xv) - 2 * k
        return av**(-k) * (av**y - 1 if y < 1 else av - av**(y - 1))
    av = 3.0
    S = sum(mp.quad(lambda u, k=k: fnum(av, k, u), [2 * k, 2 * k + 1, 2 * k + 2]) for k in range(0, 40))
    kk = sp.symbols("k", integer=True, positive=True)
    r = sp.symbols("r", positive=True)                     # r = 1/a < 1 (a>1)
    Sinf = sp.simplify(sp.summation(I02 * r**(kk - 1), (kk, 1, sp.oo)).subs(r, 1 / a))
    Sinf = Sinf.args[0][0] if isinstance(Sinf, sp.Piecewise) else Sinf
    sol = sp.solve(sp.Eq(Sinf, 2025), a)
    ok3 = abs(S - av) < 1e-9 and sol == [sp.sympify(md_answer(2, "3"))]
    report(2, ok1 and ok2 and ok3, f"∫0^2 f={I02}, (2) 40점 일치, S_40(a=3)={float(S):.10f}, lim S=a → a={sol}")


def p03():
    f = sp.Function("f")
    # 주어진 식을 미분해 f 를 결정: f + (x^2-x+1)e^x = e^x J,  J' = e^{-x} f'
    # 미분: f' + (x^2+x)e^x = e^x J + f'  ⇒ e^x J = (x^2+x)e^x ⇒ f = e^x J - (x^2-x+1)e^x
    fx = sp.simplify((x**2 + x) * sp.exp(x) - (x**2 - x + 1) * sp.exp(x))
    lhs = fx + (x**2 - x + 1) * sp.exp(x)
    rhs = sp.integrate(sp.exp(x - t) * sp.diff(fx, x).subs(x, t), (t, 0, x))
    ok_eq = sp.simplify(lhs - rhs) == 0
    ok1 = fx.subs(x, 0) == -1
    Fx = sp.simplify(sp.integrate(sp.exp(-t) * fx.subs(x, t), (t, 0, x)))
    roots = set(sp.solve(Fx, x))
    ok2 = roots == set(sp.sympify(md_answer(3, "2")))
    crit = sp.solve(sp.diff(fx, x), x)
    ext = [(c, fx.subs(x, c), sp.diff(fx, x, 2).subs(x, c)) for c in crit]
    ok3 = len(ext) == 1 and ext[0][2] > 0 and sp.simplify(ext[0][1] - sp.sympify(md_answer(3, "3"))) == 0
    report(3, ok_eq and ok1 and ok2 and ok3, f"f={fx}, F={Fx}, 근={roots}, 극소 {ext}")


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
