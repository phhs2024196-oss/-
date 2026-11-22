#!/usr/bin/env python3
"""HIU-G23S (홍익대 2023학년도 수시 논술 자연계열 세종) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G23S"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b = sp.symbols("a b", positive=True)
    sols = sp.solve([sp.Eq(2 * a + b, 3**2), sp.Eq(a * b, 3**1)], [a, b], dict=True)
    sols = [s for s in sols if s[a] > 0 and s[b] > 0]
    # 원 조건(로그식) 재확인
    ok_log = all(abs(float(sp.log(2 * s[a] + s[b], 3) - 2)) < 1e-12 and
                 abs(float(sp.log(s[a], 3) + sp.log(s[b], 3) - 1)) < 1e-12 for s in sols)
    vals = {sp.nsimplify(sp.simplify(4 * s[a]**2 + s[b]**2)) for s in sols}
    ok = len(sols) == 2 and ok_log and vals == {sp.sympify(md_answer(1, "1"))}
    # 풀이법 3: 2a, b 는 x^2-9x+6=0 의 두 근
    r = sp.solve(x**2 - 9 * x + 6, x)
    ok3 = sp.simplify(r[0]**2 + r[1]**2 - sp.sympify(md_answer(1, "1"))) == 0
    report(1, ok and ok3, f"sols={len(sols)} 4a^2+b^2={vals}")


def p02():
    a = sp.symbols("a", real=True)
    poly = x**3 + 3 * x**2 - 1 + a
    disc = sp.discriminant(poly, x)          # 삼차식: 판별식 > 0 ⇔ 서로 다른 세 실근
    region = sp.solve_univariate_inequality(disc > 0, a, relational=False)
    nums = [sp.Rational(v) for v in re.findall(r"-?\d+", md_answer(2, "1"))]
    md_set = sp.Interval.open(nums[0], nums[1])
    # 전수 표본: 격자 a 에서 실근 개수 직접 계산
    ok_grid = True
    for k in range(-60, 41):
        av = sp.Rational(k, 10)
        distinct = len(set(sp.real_roots(sp.Poly(poly.subs(a, av), x))))
        if (distinct == 3) != (av in md_set):
            ok_grid = False
    report(2, region == md_set and ok_grid, f"disc>0 region={region}")


def p03():
    t = sp.symbols("t", real=True)
    f = sp.Piecewise((-3 * t + 2, t < 0), (3 * t + 2, True))
    g = lambda u: sp.integrate(t * f, (t, -1, u))
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    v1, v2 = g(-2), g(3)
    ok = sp.simplify(v1 - sp.sympify(parts["g(-2)"])) == 0 and sp.simplify(v2 - sp.sympify(parts["g(3)"])) == 0
    # 원 예시 답안의 부호 오기 확인: ∫_{-1}^0 (-3t^2+2t)dt = -2 (앞에 - 를 붙이면 +2)
    piece = sp.integrate(-3 * t**2 + 2 * t, (t, -1, 0))
    report(3, ok and piece == -2, f"g(-2)={v1} g(3)={v2} ∫_-1^0={piece}")


def p04():
    th = sp.symbols("theta", real=True)
    D4 = (2 * sp.sin(th))**2 - 2 * (1 - sp.cos(th))
    sol = sp.solveset(sp.Eq(D4, 0), th, sp.Interval.Ropen(0, 2 * sp.pi))
    md = sp.FiniteSet(*sp.sympify(md_answer(4, "1")))
    # 수치 표본: 격자 3600점에서 D4 부호가 바뀌거나 0 인 곳이 md 집합 근방뿐인지
    import math
    fn = sp.lambdify(th, D4)
    hits = [k * 2 * math.pi / 3600 for k in range(3600) if abs(fn(k * 2 * math.pi / 3600)) < 1e-9]
    ok_num = all(any(abs(h - float(m)) < 1e-6 for m in md) for h in hits) and len(hits) == len(md)
    report(4, sol == md and ok_num, f"solutions={sol}")


def p05():
    a1, d1, b1, d2 = sp.symbols("a1 d1 b1 d2")
    an = lambda n: a1 + (n - 1) * d1
    bn = lambda n: b1 + (n - 1) * d2
    sol = sp.solve([sp.Eq(an(2) + bn(4), 11), sp.Eq(an(5) + bn(13), 20)], [a1, d1], dict=True)[0]
    val = sp.simplify(sol[a1] + b1)                      # b1, d2 자유 → a1+b1 이 자유변수에 무관한지
    ok = val.free_symbols == set() and sp.simplify(val - sp.sympify(md_answer(5, "1"))) == 0
    report(5, ok, f"a1+b1={val}")


def p06():
    a = sp.symbols("a", real=True)
    f = -(3**(-x) + 1)**2 + 3**(-x + 1) + 2 * a
    crit = sp.solve(sp.diff(f, x), x)                     # 임계점
    crit = [c for c in crit if c.is_real and c >= -1]
    # x→∞ 극한, x=-1 끝값, 임계점 값 중 최대·최소
    cand = [f.subs(x, c) for c in crit] + [f.subs(x, -1)]
    lim_inf = sp.limit(f, x, sp.oo)
    fmax = max(cand, key=lambda e: float(e.subs(a, 0)))
    av = sp.solve(sp.Eq(fmax, 1), a)[0]
    vals = [sp.nsimplify(sp.simplify(c.subs(a, av))) for c in cand] + [sp.simplify(lim_inf.subs(a, av))]
    fmin = min(vals, key=float)
    # 최솟값이 실제로 달성되는지(극한값이 아닌지)
    attained = any(sp.simplify(sp.simplify(c.subs(a, av)) - fmin) == 0 for c in cand)
    ok = attained and sp.simplify(fmin - sp.sympify(md_answer(6, "1"))) == 0
    report(6, ok, f"a={av} max=1 min={fmin} lim_x→∞={sp.simplify(lim_inf.subs(a, av))}")


def p07():
    b, c, d = sp.symbols("b c d")
    f = x**3 + b * x**2 + c * x + d                       # (가): 최고차항 계수 1 인 삼차식
    fp = sp.diff(f, x)
    eqs = [sp.Eq(fp.subs(x, 1), 0),                        # (나) 수렴 조건
           sp.Eq(sp.diff(fp, x).subs(x, 1), 6),            # (나) 극한값 = f''(1)
           sp.Eq(f.subs(x, 2), 0)]                         # (다)
    sol = sp.solve(eqs, [b, c, d], dict=True)[0]
    F = sp.expand(f.subs(sol))
    ok_md = sp.expand(F - sp.sympify(md_answer(7, "1"), locals={"x": x})) == 0
    ok_conds = (sp.limit(F / (x**3 - 2 * x**2 + 1), x, sp.oo) == 1 and
                sp.limit(sp.diff(F, x) / (x - 1), x, 1) == 6 and F.subs(x, 2) == 0)
    report(7, ok_md and ok_conds, f"f={F}")


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
