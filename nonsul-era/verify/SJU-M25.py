#!/usr/bin/env python3
"""SJU-M25 (세종대 2025학년도 모의논술 자연계열) — sympy/수치 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-M25"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): sp.sympify(v.strip()) for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    cs = sp.symbols("c0:5", real=True)
    f = sum(cs[i] * x**i for i in range(5))
    fp = sp.diff(f, x)
    sol = sp.solve([f.subs(x, -1), fp.subs(x, -1) - 1, f.subs(x, 2) - 3, fp.subs(x, 2) - 1], cs[:4], dict=True)[0]
    f = f.subs(sol); k = cs[4]                                           # 남은 자유변수 = 최고차항 계수
    v1 = sp.simplify(sp.diff(f, x).subs(x, sp.Rational(1, 2)))
    ok1 = v1 == sp.sympify(md_answer(1, "1")) and not v1.has(k)
    kcond = sp.solve(f.subs(x, 0) < 0, k)                                # f(0)<0
    # (2) 허용되는 k 몇 개에서 f''>0 인 구간을 수치로
    s = md_answer(1, "2").replace(" ", "")
    lo, hi = [float(sp.sympify(v)) for v in s[1:-1].split(",")] if s.count(",") == 1 else (None, None)
    ok2 = True
    for kv in (-0.3, -1, -5):
        assert f.subs(x, 0).subs(k, kv) < 0
        f2 = sp.lambdify(x, sp.diff(f, x, 2).subs(k, kv))
        xs = np.linspace(-3, 4, 70001)
        pos = xs[f2(xs) > 0]
        ok2 &= abs(pos.min() - lo) < 2e-4 and abs(pos.max() - hi) < 2e-4
    # (3) [증명-수치확인] k<-1/4 여러 값에서 실근 4개, 모두 3 미만
    ok3 = True
    for kv in (-0.2501, -0.3, -1, -3, -50):
        rts = sp.Poly(f.subs(k, kv), x).nroots()
        real = sorted(float(sp.re(r)) for r in rts if abs(sp.im(r)) < 1e-12)
        ok3 &= len(real) == 4 and max(real) < 3
    report(1, ok1 and ok2 and ok3, f"f'(1/2)={v1} k조건 {kcond} 볼록구간~({lo:.4f},{hi:.4f}) 근<3 확인 [증명-수치확인]")


def p02():
    p, q, c, t = sp.symbols("p q c t", real=True)
    f = x**3 + p * x**2 + q * x + c
    tan = sp.diff(f, x).subs(x, t) * (x - t) + f.subs(x, t)
    quo, rem = sp.div(sp.expand(f - tan), sp.expand((x - t)**2), x)
    other = sp.solve(quo, x)[0]                                           # 접선이 다시 만나는 점
    # (가): g(t)=t ⇔ other<=t. 최소 t 가 2 → 경계 other=t 의 해가 2
    tb = sp.solve(sp.Eq(other, t), t)[0]
    ps = sp.solve(sp.Eq(tb, 2), p)[0]
    v1 = sp.diff(f, x, 2).subs({p: ps, x: 1})
    ok1 = v1 == sp.sympify(md_answer(2, "1")) and sp.simplify(sp.diff(other.subs(p, ps), t)) < 0
    g_lt = other.subs(p, ps)
    md2 = sp.sympify(md_answer(2, "2"), locals={"t": t})
    ok2 = sp.simplify(g_lt - md2) == 0 and all(g_lt.subs(t, tv) > tv for tv in (-3, 0, 1.9))
    # (3) h=f∘g 의 t=2 좌우 미분계수 일치 조건으로 q 결정 → h' 연속, ∫_1^2 h'
    fs = f.subs(p, ps)
    hl, hr = fs.subs(x, g_lt), fs.subs(x, t)
    qs = sp.solve(sp.Eq(sp.diff(hl, t).subs(t, 2), sp.diff(hr, t).subs(t, 2)), q)[0]
    dl, dr = sp.diff(hl, t).subs(q, qs), sp.diff(hr, t).subs(q, qs)
    cont = sp.limit(dl, t, 2, "-") == sp.limit(dr, t, 2, "+") == dl.subs(t, 2)
    I = sp.integrate(dl, (t, 1, 2))                                        # t<2 구간의 h'
    ok3 = cont and sp.simplify(I - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3, f"p={ps} f''(1)={v1} g(t)={g_lt} q={qs} ∫h'={I}")


def p03():
    m, a = sp.symbols("m a", real=True)
    g = lambda u: m * (u - a)                                              # f 상수 m 일 때 g
    sols = sp.solve(sp.Poly(sp.expand(g(g(x)) - 4 * x), x).coeffs(), [m, a], dict=True)
    sols = [s for s in sols if s[m] > 0]                                  # f(0)>0
    md3 = dict(seg.split("=", 1) for seg in md_answer(3, "3").replace(" ", "").split(";"))
    fm = sp.sympify(md3["f(x)"], locals={"x": x}); am = sp.sympify(md3["a"])
    tt = sp.symbols("tt", real=True)
    G = lambda u: sp.integrate(fm.subs(x, tt), (tt, am, u))
    ok3 = len(sols) == 1 and sols[0][m] == fm and sols[0][a] == am and sp.simplify(G(G(x)) - 4 * x) == 0
    ok1 = sp.simplify(G(0) - sp.sympify(md_answer(3, "1"))) == 0
    ok2 = sp.simplify(fm.subs(x, 0) - sp.sympify(md_answer(3, "2"))) == 0 and fm.subs(x, 0) > 0
    report(3, ok1 and ok2 and ok3, f"상수해 {sols} g(0)={G(0)} f(0)={fm.subs(x, 0)} g∘g=4x 확인")


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
