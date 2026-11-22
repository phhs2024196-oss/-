#!/usr/bin/env python3
"""SYU-G26H (삼육대 2026학년도 논술 인문계열 수학 10~15) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산 후 yaml answer.value 와 비교."""
import math
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "SYU-G26H"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "SYU" / "PAST"


def load_answer(seq: int, sub: str = "1"):
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    for s in d["subparts"]:
        if str(s["id"]) == sub:
            return sp.nsimplify(sp.sympify(str(s["answer"]["value"])))
    raise KeyError(sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p10():
    a, b = sp.symbols("a b")
    L = 2 * a * x**3 + 4 * x
    R = 3 * a * x**2 + x + 2 * b
    sol = sp.solve([sp.Eq(L.subs(x, 2), R.subs(x, 2)),
                    sp.Eq(sp.diff(L, x).subs(x, 2), sp.diff(R, x).subs(x, 2))], [a, b], dict=True)[0]
    val = L.subs(sol).subs(x, 1) + R.subs(sol).subs(x, 4)
    # 오답 재현: f(4)를 x<2 식에 대입
    wrong = L.subs(sol).subs(x, 1) + L.subs(sol).subs(x, 4)
    report(10, sp.simplify(val - load_answer(10)) == 0, f"a={sol[a]}, b={sol[b]}, f(1)+f(4)={val} (오답 MC-CAL-4102: {wrong})")


def p11():
    f = sp.Rational(1, 3) * x**3 - x**2 - 3 * x
    def count(k):
        roots = set()
        for rhs in ({k, -k}):
            for r in sp.Poly(f - rhs, x).all_roots():
                if r.is_real:
                    roots.add(sp.nsimplify(r) if r.is_rational else r)
        return len(set(sp.N(r, 30) for r in roots))
    alpha = beta = 0
    table = {}
    for k in range(0, 50):            # k<0 이면 근 없음, k>=10 이면 n=2
        c = count(k)
        table[k] = c
        if c > 2:
            if c % 2:
                alpha += k
            else:
                beta += k
    assert all(table[k] == 2 for k in range(10, 50))
    val = beta - alpha
    report(11, val == load_answer(11), f"n(k)={ {k: table[k] for k in range(0, 11)} }, alpha={alpha}, beta={beta}, beta-alpha={val}")


def p12():
    a = sp.symbols("a")
    y = (x + 2)**2
    tl = sp.diff(y, x).subs(x, a) * (x - a) + y.subs(x, a)
    sols = sorted(sp.solve(sp.Eq(tl.subs(x, -1), -3), a))
    l1, l2 = [sp.expand(tl.subs(a, s)) for s in sols]
    xm = sp.solve(sp.Eq(l1, l2), x)[0]
    area = sp.integrate(y - l1, (x, sols[0], xm)) + sp.integrate(y - l2, (x, xm, sols[1]))
    report(12, sp.simplify(area - load_answer(12)) == 0, f"접점 x={sols}, 접선 {l1}, {l2}, 교점 x={xm}, 넓이={area}")


def p13():
    s = sum(k * (-1)**((k * k + k - 2) // 2) for k in range(1, 100))
    assert all((k * k + k - 2) % 2 == 0 for k in range(1, 100))
    alt = sum(k * (-1)**(k + 1) for k in range(1, 100))   # 오답 MC-ALG-4101 재현
    report(13, s == load_answer(13), f"합={s}, 부호 첫 8항={[(-1)**((k*k+k-2)//2) for k in range(1, 9)]} (교대 오답 {alt})")


def p14():
    # 상한·하한 끼워넣기: 5x-1 < [5x] <= 5x
    lo = sp.limit(sp.sqrt(9 * x**2 + 5 * x - 1) - 3 * x, x, sp.oo)
    hi = sp.limit(sp.sqrt(9 * x**2 + 5 * x) - 3 * x, x, sp.oo)
    # 실제 바닥함수로 큰 x 수치 확인
    import mpmath as mp
    mp.mp.dps = 60
    X = mp.mpf(10)**20 + mp.mpf("0.37")
    num = mp.sqrt(9 * X**2 + mp.floor(5 * X)) - 3 * X
    ans = load_answer(14)
    ok = lo == hi and sp.simplify(lo - ans) == 0 and abs(num - mp.mpf(sp.Rational(ans).p) / sp.Rational(ans).q) < mp.mpf(10)**-15
    report(14, ok, f"하한극한={lo}, 상한극한={hi}, x=1e20 수치={mp.nstr(num, 20)}")


def p15():
    u = sp.symbols("u", positive=True)       # u = x/y > 0
    nn = sp.symbols("nn", positive=True, integer=True)
    T = (2 * u**2 + nn * u + 2) / (2 * u**2 - u + 2)   # y=1 로 동차화 (xy>0 → u>0)
    crit = [c for c in sp.solve(sp.diff(T, u), u) if c.is_positive]
    tmax = sp.simplify(T.subs(u, crit[0]))
    # 경계 u→0+, u→∞ 에서 T→1 < tmax 확인
    assert sp.limit(T, u, 0) == 1 and sp.limit(T, u, sp.oo) == 1
    best = max(k for k in range(1, 200) if sp.log(tmax.subs(nn, k), 3) < 2 and tmax.subs(nn, k) < 9)
    report(15, best == load_answer(15), f"u=x/y 임계점 {crit}, max t={tmax}, f(n)<2 인 최대 n={best}")


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
