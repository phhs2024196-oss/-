#!/usr/bin/env python3
"""KNU 2023학년도 논술(AAT) 모의고사 자연계열Ⅰ 수학 (KNU-M23A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-M23A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import yaml, pathlib
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())


def load_answer(seq, i=0):
    return load_item(seq)["subparts"][i]["answer"]["value"]


def p01():
    msgs, ok = [], True
    from math import perm, comb
    s = sum(sp.Rational(perm(10, r), comb(10, 10 - r)) for r in range(5))
    ok &= s == sp.sympify(load_answer(1, 0)); msgs.append(f"1-1(1) {s}")
    ns = [n for n in range(2, 200) if perm(n + 1, 2) + comb(n, n - 2) == 26]
    v = sp.sin(sp.pi / ns[0])
    ok &= len(ns) == 1 and sp.simplify(v - sp.sympify(load_answer(1, 1))) == 0; msgs.append(f"1-1(2) n={ns} sin={v}")
    a = sp.symbols("a", positive=True)
    f = a * (t - 3) * (t + 3)
    g = sp.integrate(f, (t, 0, x)); gp = sp.diff(g, x)
    mins = [r for r in sp.solve(gp, x) if gp.subs(x, r - sp.Rational(1, 10)) < 0 < gp.subs(x, r + sp.Rational(1, 10))]
    ok &= mins == [sp.sympify(load_answer(1, 2))]; msgs.append(f"1-2(1) 극소점={mins}")
    h = sp.expand(sp.integrate(f, (t, x, x + 3)))
    xm = sp.solve(sp.diff(h, x), x)[0]
    av = sp.solve(sp.Eq(h.subs(x, xm), sp.Rational(-33, 4)), a)[0]
    S = sum(h.subs(a, av).subs(x, k) for k in range(1, 11))
    ok &= S == sp.sympify(load_answer(1, 3)); msgs.append(f"1-2(2) a={av} Σh={S}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    m = sp.symbols("m", positive=True)
    good = []
    for a, b, c, d in itertools.product(range(1, 5), repeat=4):
        f = a * x ** 3 + b * x ** 2 + c * x + d
        if sp.diff(f, x).subs(x, 1) != 10:
            continue
        # g=(x-m)(x+1)/f 가 실수 전체에서 연속(미분가능)이려면: f 의 모든 실근이 분자 근으로 약분되어야 함(m>0 이면 f(m)>0 이라 x=m 은 불가)
        q, r = sp.div(sp.Poly(f, x), sp.Poly(x + 1, x))
        if r.is_zero and not [z for z in sp.real_roots(q)]:
            good.append((a, b, c, d))
    vals = sorted({int((a * 8 + b * 4 + c * 2 + d)) for a, b, c, d in good})
    a21 = sorted(int(v) for v in sp.sympify(load_answer(2, 0)))
    ok &= vals == a21; msgs.append(f"2-1 계수={good} f(2)={vals} md={a21}")
    # 2-2: lim f/(x+1)=6 인 경우, g 와 4/g 교점 1개인 m
    sel = [q for q in good if sp.limit((q[0] * x ** 3 + q[1] * x ** 2 + q[2] * x + q[3]) / (x + 1), x, -1) == 6]
    a, b, c, d = sel[0]
    f = a * x ** 3 + b * x ** 2 + c * x + d
    g = sp.cancel((x - m) * (x + 1) / f)
    sols = []
    for mv in sp.solve(sp.discriminant(sp.numer(sp.together(g + 2)), x), m) + sp.solve(sp.discriminant(sp.numer(sp.together(g - 2)), x), m):
        if mv.is_positive:
            gm = g.subs(m, mv)
            cnt = len(set(sp.real_roots(sp.numer(sp.together(gm - 2)))) | set(sp.real_roots(sp.numer(sp.together(gm + 2)))))
            if cnt == 1:
                sols.append(mv)
    a22 = sp.sympify(load_answer(2, 1))
    ok &= len(sel) == 1 and sols == [a22]; msgs.append(f"2-2 경우={sel} m={sols} md={a22}")
    # 2-3: f(0)=3 경우, x=m 에서 연속·미분가능
    sel = [q for q in good if q[3] == 3]
    a, b, c, d = sel[0]
    g = sp.cancel((x - m) * (x + 1) / (a * x ** 3 + b * x ** 2 + c * x + d))
    k = sp.symbols("k")
    left = k - g.subs(x, x - m); right = (x - m) * g
    msol = sp.solve(sp.Eq(sp.diff(left, x).subs(x, m), sp.diff(right, x).subs(x, m)), m)
    ksol = sp.solve(sp.Eq(left.subs(x, m), right.subs(x, m)).subs(m, msol[0]), k)
    a23 = sp.sympify(load_answer(2, 2))
    ok &= len(sel) == 1 and ksol == [a23]; msgs.append(f"2-3 m={msol} k={ksol} md={a23}")
    report(2, ok, "; ".join(msgs))


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    def dist(n, lo, hi):
        sp_ = lambda tt: mp.sqrt((n * mp.cos(tt) ** (n - 1) * mp.sin(tt)) ** 2 + (n * mp.sin(tt) ** (n - 1) * mp.cos(tt)) ** 2)
        return mp.quad(sp_, [lo, (lo + hi) / 2, hi])
    s2, s3 = dist(2, 0, mp.pi / 2), dist(3, 0, mp.pi / 2)
    a31 = sp.sympify(load_answer(3, 0))
    ok &= abs(s2 - mp.mpf(sp.N(a31[0], 30))) < 1e-20 and abs(s3 - mp.mpf(sp.N(a31[1], 30))) < 1e-20
    msgs.append(f"3-1 s2={mp.nstr(s2, 12)} s3={mp.nstr(s3, 12)} md={a31}")
    eq = all(abs(dist(n, 0, mp.pi / 4) - dist(n, mp.pi / 4, mp.pi / 2)) < 1e-20 for n in range(2, 15))
    ok &= eq; msgs.append(f"3-2 a_n=b_n (n=2..14)={eq} [증명-수치확인]")
    vals = [dist(n, 0, mp.pi / 2) for n in (10, 40, 160)]
    a33 = sp.sympify(load_answer(3, 2))
    bound = all(2 - 4 * (mp.sqrt(2) / 2) ** n <= v <= 2 for n, v in zip((10, 40, 160), vals))
    ok &= bound and abs(vals[-1] - a33) < 1e-10
    msgs.append(f"3-3 s_n(n=10,40,160)={[mp.nstr(v, 10) for v in vals]} 끼움성립={bound} → md={a33}")
    report(3, ok, "; ".join(msgs))


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
