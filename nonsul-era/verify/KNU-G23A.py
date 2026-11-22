#!/usr/bin/env python3
"""KNU 2023학년도 수시 논술(AAT) 자연계열Ⅰ 수학 (KNU-G23A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import random
import re
import sys

import sympy as sp

PREFIX = "KNU-G23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


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
    a, b = sp.symbols("a b", real=True)
    msgs, ok = [], True
    # 조건식을 f, g 에 대해 풀고 끝점은 한쪽 극한(연속)
    F, G = sp.symbols("F G")
    f = sp.solve(sp.Eq((F - (1 - x) ** 2) / (x * (1 - x)), a), F)[0]
    g = sp.solve(sp.Eq((G - x ** 2) / (x * (1 - x)), b), G)[0]
    v0 = sp.limit(f - g, x, 0, "+"); v1 = sp.limit(f - g, x, 1, "-")
    a11 = tuple(sp.sympify(load_answer(1, 0)))
    ok &= (v0, v1) == a11
    msgs.append(f"1-1 ({v0},{v1}) md={a11}")
    # 1-2(1): 무작위 (a,b) 에서 (0,1) 안의 근이 유일하고 h(t) 식과 일치 [증명-수치확인]
    random.seed(3)
    hexpr = sp.Piecewise((sp.Rational(1, 2), sp.Eq(t, 0)), ((t - 2 + sp.sqrt(t ** 2 + 4)) / (2 * t), True))
    good = True
    for _ in range(40):
        av, bv = sp.Rational(random.randint(-60, 60), 7), sp.Rational(random.randint(-60, 60), 7)
        if _ == 0: bv = av
        p = sp.expand((f - g).subs({a: av, b: bv}))
        rts = [r for r in sp.solve(p, x) if r.is_real and 0 < r < 1]
        good &= len(rts) == 1 and sp.simplify(rts[0] - hexpr.subs(t, av - bv)) == 0
    ok &= good
    msgs.append(f"1-2(1) 무작위 40쌍 근 유일·식 일치={good} [증명-수치확인]")
    h = (t - 2 + sp.sqrt(t ** 2 + 4)) / (2 * t)
    d0 = sp.limit((h - sp.Rational(1, 2)) / t, t, 0)
    a122 = sp.sympify(load_answer(1, 2))
    ok &= sp.simplify(d0 - a122) == 0
    msgs.append(f"1-2(2) h'(0)={d0} md={a122}")
    I = sp.simplify(sp.integrate(t ** 2 * h, (t, 1, 4)))
    a123 = sp.sympify(load_answer(1, 3))
    ok &= sp.simplify(I - a123) == 0
    msgs.append(f"1-2(3) ∫={sp.nsimplify(I)} md={a123}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    d = sp.symbols("d", positive=True)
    X, Y = sp.symbols("X Y")
    def inter(p1, q1, p2, q2):   # 직선 x/p1+y/q1=1, x/p2+y/q2=1 교점
        s = sp.solve([X / p1 + Y / q1 - 1, X / p2 + Y / q2 - 1], [X, Y], dict=True)[0]
        return s[X], s[Y]
    def setup(nv, dv):
        an, bn, cn = 2 * nv + 8, 5 * nv + 10, nv + 9
        E = inter(an, cn, bn, dv)
        Fp = sp.solve([Y * E[0] - X * E[1], X / an + Y / dv - 1], [X, Y], dict=True)[0]
        F = (Fp[X], Fp[Y])
        gx = sp.solve(sp.Eq(Y, 0).subs(Y, cn + (F[1] - cn) / F[0] * X), X)[0]  # 직선 C_nF_n 의 x절편
        return an, bn, cn, E, F, sp.simplify(gx)
    # 2-1: slope(OE_n) 을 직접 계산해 (k n + l)(n+9)d/(10(n+2)(n+4)(n+9-d)) 꼴의 k,l 결정
    ns = sp.symbols("ns", positive=True)
    an, bn, cn, E, F, gx = setup(ns, d)
    num = sp.factor(sp.simplify(E[1] / E[0] * 10 * (ns + 2) * (ns + 4) * (ns + 9 - d) / ((ns + 9) * d)))
    k, l = sp.Poly(sp.expand(num), ns).all_coeffs()
    a21 = tuple(sp.sympify(load_answer(2, 0)))
    ok &= (k, l) == a21
    msgs.append(f"2-1 (ㄱ,ㄴ)=({k},{l}) md={a21}")
    # 2-2: g_n 을 실제 교점으로 계산(d 무관 확인) 후 극한
    assert sp.simplify(sp.diff(gx, d)) == 0
    L = sp.limit(sp.Abs(gx - an) / gx, ns, sp.oo)
    a22 = sp.sympify(load_answer(2, 1))
    ok &= sp.simplify(L - a22) == 0
    msgs.append(f"2-2 g_n={sp.factor(gx)} 극한={L} md={a22}")
    # 2-3: n=1 넓이(신발끈) 의 [1,9] 최대점
    an, bn, cn, E, F, gx = setup(1, d)
    D, G = (0, d), (gx, 0)
    sarea = sp.simplify(((E[0] - D[0]) * (G[1] - D[1]) - (G[0] - D[0]) * (E[1] - D[1])) / 2)
    area = sp.simplify(sp.sign(sarea.subs(d, 5)) * sarea)          # [1,9] 에서 부호 일정(분모 15-d>0)
    A = sp.lambdify(d, area)
    crit = [r for r in sp.solve(sp.diff(area, d), d) if r.is_real and 1 <= r <= 9]
    cands = crit + [sp.Integer(1), sp.Integer(9)]
    best = max(cands, key=lambda c: float(area.subs(d, c)))
    grid = max((1 + 8 * i / 4000 for i in range(4001)), key=A)
    a23 = sp.sympify(load_answer(2, 2))
    ok &= sp.simplify(best - a23) == 0 and abs(grid - float(a23)) < 1e-2
    msgs.append(f"2-3 최대점={best}(격자 {grid:.4f}) md={a23}")
    report(2, ok, "; ".join(msgs))


def p03():
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    nn = sp.symbols("nn", positive=True, integer=True)
    def fx(nv):
        return (-x ** 2 + nv * (nv + 3) * x + nv) / (nv * (x + 1))
    # 3-1: k=1..(충분히 큰 상한) 전수: f(k) 가 자연수인 가장 큰 k (f 는 x>n 에서 감소, f(k)<1 이면 이후 모두 <1)
    tot = 0; bvals = []
    for nv in range(1, 11):
        f = fx(sp.Integer(nv)); best = None; k = 1
        while True:
            v = f.subs(x, k)
            if k > nv and v < 1:
                break
            if v.is_integer and v >= 1:
                best = k
            k += 1
        tot += best
        crit = [r for r in sp.solve(sp.diff(f, x), x) if r > -1]
        bvals.append(crit[0])
    a31 = sp.sympify(load_answer(3, 0))
    ok &= tot == a31 and bvals == list(range(1, 11))
    msgs.append(f"3-1 Σa_n={tot} b_n=n 확인 md={a31}")
    # 3-2: n=1..30, [b_n,a_n] 격자 1000점에서 f-g>=0 [증명-수치확인]
    good = True
    for nv in range(1, 31):
        f = sp.lambdify(x, fx(nv)); a_, b_ = nv * nv + 2 * nv, nv
        g = lambda z: f(b_) + (f(a_) - f(b_)) / (a_ - b_) * (z - b_)
        good &= all(f(b_ + (a_ - b_) * i / 1000) - g(b_ + (a_ - b_) * i / 1000) >= -1e-9 for i in range(1001))
    ok &= good
    msgs.append(f"3-2 f>=g (n<=30, 격자 1000점)={good} [증명-수치확인]")
    # 3-3: S_n 을 그린 정리로 직접 수치 계산 (P→D 선분, D→E 호, E→P 곡선)
    def S(nv):
        nv = mp.mpf(nv); P = (nv, nv + 1)
        f = lambda z: (-z ** 2 + nv * (nv + 3) * z + nv) / (nv * (z + 1))
        q = mp.mpf(1) * (mp.sqrt(3) + 1) * nv + mp.sqrt(3)
        phiD = mp.atan2(0 - P[1], q - P[0])
        xE = mp.findroot(lambda z: (z - P[0]) ** 2 + (f(z) - P[1]) ** 2 - nv ** 2, P[0] + nv * 0.99)
        phiE = mp.atan2(f(xE) - P[1], xE - P[0])
        arc = nv ** 2 * (phiE - phiD) / 2
        df = lambda z: mp.diff(f, z)
        curve = mp.quad(lambda z: ((z - P[0]) * df(z) - (f(z) - P[1])) / 2, [xE, P[0]])
        return (arc + curve) / nv ** 2, phiD
    vals = [S(10 ** e) for e in (2, 3, 4, 5)]
    a33 = sp.sympify(load_answer(3, 2))
    conv = abs(vals[-1][0] - float(a33)) < 1e-4 and all(abs(v[0] - float(a33)) > abs(w[0] - float(a33)) for v, w in zip(vals, vals[1:]))
    ok &= conv and abs(vals[0][1] + mp.pi / 6) < 1e-20
    msgs.append(f"3-3 S_n/n^2 (n=1e2..1e5)={[mp.nstr(v[0], 8) for v in vals]} → md={a33}≈{float(a33):.8f}")
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
