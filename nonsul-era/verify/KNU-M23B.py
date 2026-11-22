#!/usr/bin/env python3
"""KNU 2023학년도 논술(AAT) 모의고사 자연계열Ⅱ 수학 (KNU-M23B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "KNU-M23B"
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
    r = sp.symbols("r", positive=True)
    X, Y = sp.symbols("X Y", real=True)
    a = lambda n: (2 ** n - 2) * r
    b = lambda n: 2 ** (n - 1) * r
    def P(n):
        sol = sp.solve([(X - a(n)) ** 2 + Y ** 2 - b(n) ** 2, (X - a(n + 1)) ** 2 + Y ** 2 - b(n + 1) ** 2], [X, Y], dict=True)
        s = [q for q in sol if q[Y].is_positive]
        return s[0][X], s[0][Y]
    rv = sp.solve(sp.Eq(P(3)[0], 21), r)
    ok &= rv == [sp.sympify(load_answer(1, 0))]; msgs.append(f"1-1 r={rv}")
    nn = sp.symbols("nn", positive=True, integer=True)
    sol = sp.solve([(X - (2 ** nn - 2) * r) ** 2 + Y ** 2 - (2 ** (nn - 1) * r) ** 2,
                    (X - (2 ** (nn + 1) - 2) * r) ** 2 + Y ** 2 - (2 ** nn * r) ** 2], [X, Y], dict=True)
    s = [q for q in sol if sp.simplify(q[Y].subs({nn: 5, r: 1})) > 0][0]
    L = sp.limit(sp.simplify(s[Y] / s[X]), nn, sp.oo)
    a12 = sp.sympify(load_answer(1, 1))
    ok &= sp.simplify(L - a12) == 0; msgs.append(f"1-2 lim m_n={L}")
    # 1-3: Q_n = 접선의 x절편 (접선 ⟂ 반지름), s_n = 삼각형 넓이(신발끈), A_n = 실제 영역 넓이
    def s_n(n):
        px, py = P(n); qx = sp.solve(sp.Eq((X - px) * (px - a(n)) + (0 - py) * (py - 0), 0), X)[0]
        p2x, p2y = P(n + 1)
        return sp.simplify(sp.Abs((px - qx) * (p2y - 0) - (p2x - qx) * (py - 0)) / 2)
    def A(n):
        s1, s2 = s_n(n), s_n(n + 1)
        c = sp.solve(sp.Eq(s2 * x ** n, s1 * x ** (n - 1)), x)
        c = [v for v in c if v.is_positive][0]
        return sp.simplify(sp.integrate(s2 * x ** n - s1 * x ** n, (x, 0, c)) + sp.integrate(s1 * x ** (n - 1) - s1 * x ** n, (x, c, 1)))
    As = {n: A(n) for n in range(1, 10)}
    good = [n for n in range(1, 9) if sp.simplify(As[n + 1] / As[n]) > 3]
    a13 = sp.sympify(load_answer(1, 2))
    ok &= min(good) == a13 and good == list(range(min(good), 9)); msgs.append(f"1-3 A_(n+1)/A_n>3 인 n(≤8)={good}")
    report(1, ok, "; ".join(msgs))


def p02():
    msgs, ok = [], True
    c = sp.symbols("c", real=True)
    fx = lambda cv, z: cv + sp.sqrt(2 * cv * z - z ** 2)
    # 2-1(1): a_1=c 에서 처음 6항 (정확 계산)
    seqs = {}
    for cv in (sp.Integer(3), sp.Integer(-3)):
        a = [cv]
        for _ in range(6): a.append(sp.simplify(fx(cv, a[-1])))
        seqs[int(cv)] = a
    div = all(len(set(v[2:])) == 2 for v in seqs.values())
    ok &= div and load_answer(2, 0) == "발산"; msgs.append(f"2-1(1) c=±3 수열={seqs} 발산={div}")
    # 2-1(2): 정의역 격자 + 후보(고정점, 고정점의 원상) 에서 수렴 판정 (c=±3, 정확 산술)
    a12 = sp.sympify(load_answer(2, 1), locals={"c": c})
    good = True; found = {}
    for cv in (sp.Integer(3), sp.Integer(-3)):
        z = sp.symbols("z", real=True)
        lo, hi = (0, 2 * cv) if cv > 0 else (2 * cv, 0)
        fixed = [r for r in sp.solve(sp.Eq(fx(cv, z), z), z) if lo <= r <= hi]
        pre = set()
        for fp in fixed:
            pre |= {r for r in sp.solve(sp.Eq(fx(cv, z), fp), z) if lo <= r <= hi}
        cands = set(fixed) | pre
        def conv(a1):
            a = [a1]
            for _ in range(8): a.append(sp.nsimplify(sp.simplify(fx(cv, a[-1]))))
            return sp.simplify(a[-1] - a[-2]) == 0 and sp.simplify(a[-2] - a[-3]) == 0
        grid = [lo + (hi - lo) * sp.Rational(i, 40) for i in range(41)]
        conv_set = {q for q in cands if conv(q)} | {q for q in grid if conv(q)}
        expect = {sp.simplify(v.subs(c, cv)) for v in a12}
        found[int(cv)] = conv_set
        good &= {sp.simplify(q) for q in conv_set} == expect
    ok &= good; msgs.append(f"2-1(2) 수렴 초항(c=±3)={found} md={a12}")
    # 2-2: 후보 f(x)=k x^p 가 조건을 만족하는 (k,p) 결정 후 g 상수, h 최소
    k, pp, xx = sp.symbols("k p xx", positive=True)
    F = k * xx ** pp
    cond = sp.simplify(F * sp.diff(F, xx).subs(xx, 1 / xx) - xx)
    sol = sp.solve([cond.subs(xx, 2), cond.subs(xx, 3)], [k, pp], dict=True)
    gconst = all(sp.simplify(sp.diff(F.subs(s_) * F.subs(s_).subs(xx, 1 / xx), xx)) == 0 for s_ in sol)
    ok &= gconst and sp.sympify(load_answer(2, 2)) == 1
    msgs.append(f"2-2(1) f=kx^p 해 {sol}, g'=0={gconst}")
    tt = sp.symbols("tt", positive=True)
    # t=g(1)=k^2, p=1/t
    hs = []
    for s_ in sol:
        tv = sp.simplify((F * F.subs(xx, 1 / xx)).subs(s_))
        hs.append(sp.simplify(F.subs(s_).subs(xx, sp.E)))
    kk = sp.symbols("kk", positive=True)
    h = sp.sqrt(tt) * sp.exp(1 / tt)
    chk = sp.simplify(cond.subs({k: sp.sqrt(tt), pp: 1 / tt})) == 0
    crit = [r for r in sp.solve(sp.diff(h, tt), tt) if 1 <= r <= 3]
    mn = min([h.subs(tt, v) for v in crit + [1, 3]], key=lambda v: float(v))
    a222 = sp.sympify(load_answer(2, 3))
    ok &= chk and sp.simplify(mn - a222) == 0
    msgs.append(f"2-2(2) f=√t x^(1/t) 조건만족={chk}, 최소={mn} md={a222}")
    report(2, ok, "; ".join(msgs))



def p03():
    import mpmath as mp
    lhs = sp.cos(x - sp.pi/4); rhs = (sp.sin(x)+sp.cos(x))/sp.sqrt(2)
    ok1 = sp.simplify(sp.expand_trig(lhs) - rhs) == 0
    # 점화식 독립 계산
    I = {2: sp.Integer(1)}
    for n in range(4, 11, 2):
        I[n] = sp.Rational(1, n-1) + sp.Rational(n-2, 2*n-2)*I[n-2]
    q1 = mp.quad(lambda u: 1/(mp.sin(u)+mp.cos(u))**10, [0, mp.pi/4, mp.pi/2])
    q2 = mp.quad(lambda u: u/(mp.sin(u)+mp.cos(u))**10, [0, mp.pi/4, mp.pi/2])
    a2 = sp.sympify(load_answer(3, 1)); a3 = sp.sympify(load_answer(3, 2))
    ok2 = sp.simplify(a2 - I[10]) == 0 and abs(float(a2) - float(q1)) < 1e-10
    ok3 = abs(float(a3) - float(q2)) < 1e-10 and sp.simplify(a3 - sp.pi/4*I[10]) == 0
    report(3, ok1 and ok2 and ok3, f"[증명-수치확인] 3-1={ok1}; 3-2 rec={I[10]} quad={float(q1):.12f}; 3-3 quad={float(q2):.12f} (π/4)I10 일치={ok3}")

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
