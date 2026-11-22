#!/usr/bin/env python3
"""가천대 2024학년도 논술전형 자연D 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-G24D"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int) -> str:
    """YAML 의 answer.value (LaTeX 문자열)."""
    it = yaml.safe_load((ROOT / "items/GCU/PAST" / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def tex(s: str):
    """간단한 LaTeX(\\dfrac, \\sqrt, 정수) → sympy."""
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    for _ in range(5):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\sqrt(\d)", r"sqrt(\1)", s)
        s = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", s)
        s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"((\1)/(\2))", s)
        s = re.sub(r"\\frac(\d)\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("\\pi", "pi").replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)\s*(sqrt|pi|\()", r"\1*\2", s)
    return sp.nsimplify(sp.sympify(s))


def same(val, s) -> bool:
    return sp.simplify(sp.nsimplify(val) - tex(s)) == 0


def parts(s):
    return [tex(seg.split("=")[-1]) for seg in s.split(",\\ ")]


def p07():
    th = sp.symbols("th", real=True)
    T = sp.symbols("T")
    roots = sp.solve(sp.Eq(T - sp.sqrt(3) / T, sp.sqrt(3) - 1), T)
    thetas = [sp.atan(r) + sp.pi for r in roots]  # π<θ<3π/2 이면 θ=atan(T)+π (T>0 일 때만 범위 안)
    ok_th = [v for v in thetas if sp.pi < v < sp.Rational(3, 2) * sp.pi]
    val = sp.nsimplify(sp.simplify(2 * sp.cos(ok_th[0]) - 4 * sp.sin(ok_th[0])), [sp.sqrt(3)])
    report(7, len(ok_th) == 1 and same(val, ans(7)), f"tanθ 후보 {roots}, θ={ok_th}, 값={val}")


def p08():
    A, B = sp.symbols("A B")
    s = sp.solve([sp.Eq(4 * A - 6, 6), sp.Eq(2 * 2 * A + 4 * B - 3, 1)], [A, B], dict=True)[0]
    icpt = (A - 2 * B).subs(s)  # y=f(2)+f'(2)(x-2) 의 y절편
    report(8, same(icpt, ans(8)), f"f(2)={s[A]}, f'(2)={s[B]}, y절편={icpt}")


def p09():
    a = [None, sp.Integer(1)]
    for nn in range(1, 60):
        a.append(a[nn] + nn * sp.sin(nn * sp.pi / 2 + sp.pi))
    ok = all(a[4 * k - 3] == 2 * k - 1 and a[4 * k - 2] == a[4 * k - 1] == -2 * k + 2 and a[4 * k] == 2 * k + 1
             and a[4 * k - 3] + a[4 * k - 2] + a[4 * k - 1] + a[4 * k] == 4 for k in range(1, 14))
    S = sum(a[1:43])
    report(9, ok and same(S, ans(9)), f"블록 규칙(⓵ -2k+2, ⓶ 2k+1, ⓷ 4) k=1..13 확인, Σ_1^42={S}")


def _counts(av, bv, tv):
    f = lambda z: (av * z + 2) / (z - bv)
    gs = set()
    for c in (tv, -tv):
        if c != av:
            gs.add(sp.nsimplify((-c * bv - 2) / (av - c)))
    hs = set()
    for sgn in (1, -1):  # f(x) = sgn*(-t x), x<=0
        for r in sp.Poly(sp.expand((av * x + 2) - sgn * (-tv * x) * (x - bv)), x).nroots():
            if abs(sp.im(r)) < 1e-12:
                r = sp.re(r)
                if r <= 1e-12 and abs(r - bv) > 1e-9 and abs(abs(f(r)) + tv * r) < 1e-8:
                    hs.add(round(float(r), 9))
    return len(gs), len(hs)


def p10():
    a, b = sp.symbols("a b")
    s = sp.solve([sp.Eq(b, -a), sp.Eq((3 * a + 2) / (3 - b) + 1, sp.Rational(2, 3))], [a, b], dict=True)[0]
    av, bv = s[a], s[b]
    ts = [sp.Rational(j, 20) for j in range(1, 80)]
    g = [_counts(av, bv, tv)[0] for tv in ts]
    h = [_counts(av, bv, tv)[1] for tv in ts]
    common = max(set(g), key=g.count)
    jumps = {ts[i] for i in range(len(ts)) if g[i] != common}
    ok = jumps == {bv} and len(set(h)) == 1 and h[0] == 1 and _counts(av, bv, 3)[1] == 1
    h_pos = {_counts(sp.Integer(1), sp.Integer(1), tv)[1] for tv in ts}  # a>0,b>0 예: (나) 위배 확인
    fa = sp.nsimplify(((a * x + 2) / (x - b)).subs(s).subs(x, av))
    report(10, ok and len(h_pos) > 1 and same(fa, ans(10)),
           f"a={av}, b={bv}: g 불연속 t={sorted(jumps)}, h≡{set(h)}; (a,b)=(1,1)이면 h 값 {sorted(h_pos)}로 불연속; f(a)={fa}")


def p11():
    d = sp.symbols("d", positive=True)
    res = []
    for roots in ([4 - d, 4, 4 + d], [4, 4 + d, 4 + 2 * d], [4 - 2 * d, 4 - d, 4]):
        others = [r for r in roots if r != 4]
        f = -(x - 4)**2 * (x - others[0]) * (x - others[1])
        if sp.simplify(sp.diff(f, x, 2).subs(x, 4)) <= 0:
            continue  # x=4 에서 극소가 아님
        crit = [c for c in sp.solve(sp.diff(f, x), x) if c != 4]
        for dv in sp.solve(sp.Eq(f.subs(x, crit[0]), 9), d):
            res.append(sp.simplify(f.subs(d, dv).subs(x, 1)))
    report(11, len(set(res)) == 1 and same(res[0], ans(11)), f"가능한 배열은 4가 가운데뿐, f(1)={set(res)}")


def p13():
    roots = []
    for n_ in (1, 2):
        lo, hi = 2 * n_ - 2, 2 * n_
        for m in range(-10, 30):
            for base in (sp.pi / 3, -sp.pi / 3):
                r = (base + 2 * sp.pi * m) / (n_ * sp.pi)
                if lo <= r < hi:
                    roots.append(sp.nsimplify(r))
    got = parts(ans(13))
    report(13, [min(roots), max(roots)] == got, f"실근 {sorted(set(roots))}, 최소={min(roots)}, 최대={max(roots)}")


def p14():
    av = sp.symbols("a1:12")
    bv = [2 - v for v in av]
    eqs = [av[0] - bv[0],
           sum((k + 1) * av[k + 1] - (k + 2) * av[k] for k in range(10)) - 50,
           sum(av[k + 1]**2 - bv[k + 1]**2 for k in range(9)) - 20]
    eqs = [sp.expand(e) for e in eqs]
    sol = sp.solve(eqs, [av[0], av[10], av[1]], dict=True)[0]
    S = sp.simplify(sum(av).subs(sol))
    report(14, S.free_symbols == set() and same(S, ans(14)), f"a1={sol[av[0]]}, a11={sp.simplify(sol[av[10]])}, Σ_1^11={S}(자유변수 무관)")


def p15():
    c, r = sp.symbols("c r")
    g = c * (9 - x**2)
    f = (x + 1) * (x - r)
    s = sp.solve([sp.Eq(f.subs(x, -2), g.subs(x, -2)), sp.Eq(f.subs(x, sp.Rational(7, 2)), g.subs(x, sp.Rational(7, 2)))], [c, r], dict=True)[0]
    f, g = f.subs(s), g.subs(s)
    shape = s[c] > 0 and s[r] > sp.Rational(7, 2) and g.subs(x, -2) > 0 and g.subs(x, sp.Rational(7, 2)) < 0 and f.subs(x, 0) < 0
    xs = [n_ for n_ in range(-60, 61)
          if f.subs(x, sp.Rational(n_, 3)) > 0 and g.subs(x, sp.Rational(n_, 3)) > 0 and f.subs(x, sp.Rational(n_, 3)) <= g.subs(x, sp.Rational(n_, 3))]
    report(15, shape and same(len(xs), ans(15)), f"그림과 맞는 구체 예 g={sp.factor(g)}, f={sp.expand(f)}: 정수 x={xs}, 개수={len(xs)}")


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
