#!/usr/bin/env python3
"""KNU 2026학년도 논술(AAT) 모의고사 자연계열Ⅱ 수학 (KNU-M26B-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KNU-M26B"
WIDTH = 2
RESULTS = []
x, h = sp.symbols("x h", real=True)


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
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    a, b, E = sp.symbols("a b E", real=True)
    f = x**3 - a * x**2 + a
    g = sp.exp(f - x) + b * (x - 1) ** 2
    # 1-1: (ⅰ) g(1)=1, g'(1)=0
    sol_a = [s_ for s_ in sp.solve(sp.diff(g, x).subs(x, 1), a) if sp.simplify(g.subs({a: s_, x: 1}) - 1) == 0]
    e11 = sp.sympify(load_answer(1, 0))
    ok &= sol_a == [e11]
    msgs.append(f"1-1 a={sol_a} md={e11}")
    # 1-2: 증가함수 + g'(1)=0 ⇒ g''(1)=0 → b; 결과 g 가 실제로 증가하는지 격자 확인
    ga = g.subs(a, sol_a[0])
    bv = sp.solve(sp.diff(ga, x, 2).subs(x, 1), b)
    assert len(bv) == 1
    gg = ga.subs(b, bv[0])
    gp = sp.lambdify(x, sp.diff(gg, x), "mpmath")
    mono = all(gp(mp.mpf(i) / 100) >= 0 for i in range(-500, 301))
    g0 = sp.expand(sp.diff(gg, x).subs(x, 0).subs(sp.E, E))
    p_, q_ = g0.coeff(E, 1), g0.coeff(E, 0)
    e12 = tuple(sp.sympify(load_answer(1, 1)))
    ok &= mono and (p_, q_) == e12
    msgs.append(f"1-2 b={bv[0]} g 증가(격자 801점)={mono} (p,q)={(p_, q_)} md={e12}")
    # 1-3: 역함수 h 를 수치로 구성(h(y)=g^{-1}(y), h'(y)=1/g'(h(y)))해 원래 적분을 직접 수치 계산
    gf = sp.lambdify(x, gg, "mpmath")
    def h(y):
        return mp.findroot(lambda u: gf(u) - y, (mp.mpf(-20), mp.mpf(20)), solver="bisect")
    hp = lambda y: 1 / gp(h(y))
    I = mp.quad(lambda u: gf(u) / (mp.e ** gf(u) * hp(gf(u))), [-1, 0])
    alpha = I - (mp.e**2 - mp.e**3) * mp.e ** (-mp.e)
    md = sp.sympify(load_answer(1, 2))
    closed = sp.integrate(sp.Symbol("t") * sp.exp(-sp.Symbol("t")), (sp.Symbol("t"), gg.subs(x, -1), gg.subs(x, 0))) - (sp.E**2 - sp.E**3) * sp.exp(-sp.E)
    closed = sp.simplify(closed)
    agree_md = abs(alpha - mp.mpf(sp.N(md, 30))) < 1e-15
    agree_cl = abs(alpha - mp.mpf(sp.N(closed, 30))) < 1e-15
    import pathlib
    raw = pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/KNU/PAST/{PREFIX}-01.yaml").read_text()
    flagged = "[공식해설 확인필요]" in raw and "6e^{7}" in raw
    if agree_md:
        msgs.append(f"1-3 α={mp.nstr(alpha, 15)} md={md} 일치")
    else:
        ok &= agree_cl and flagged
        msgs.append(f"1-3 재계산 α={closed}≈{mp.nstr(alpha, 12)} ≠ 공식답 md={md} → [공식해설 확인필요] 기록됨={flagged}")
    report(1, ok, "; ".join(msgs))


def p02():
    """[증명-수치확인] 조건 (ⅰ)~(ⅲ)을 만족하는 F(x)=-(2/π)cos(πx/2) 로 I_n 을 계산해 일반 관계식 확인."""
    import mpmath as mp
    mp.mp.dps = 30
    msgs, ok = [], True
    F = lambda X: -2 / mp.pi * mp.cos(mp.pi * X / 2)
    Fp = lambda X: mp.diff(F, X)
    Fpp = lambda X: mp.diff(F, X, 2)
    # 조건 확인
    conds = abs(F(1)) < 1e-25 and abs(Fp(0)) < 1e-25 and all(-1e-25 <= Fp(mp.mpf(i) / 50) <= 1 + 1e-25 for i in range(51)) \
        and all(abs(Fp(X) ** 2 - Fpp(X) * F(X) - 1) < 1e-20 for X in (mp.mpf(-1.3), mp.mpf(0.2), mp.mpf(0.7), mp.mpf(2.5)))
    I = {n: mp.quad(lambda X: mp.sin(mp.pi * X / 2) ** n, [0, 1]) for n in range(0, 62)}
    # 2-1
    ok21 = all(I[n] ** 2 <= I[n - 1] * I[n] + 1e-25 and I[n - 1] * I[n] <= I[n - 1] ** 2 + 1e-25 for n in range(1, 61))
    # 2-2: md (p,q,r) 로 우변 계산
    p_, q_, r_ = sp.sympify(load_answer(2, 1))
    ok22 = all(abs(I[n + 1] - (q_ * n + r_) / (n + p_) * I[n - 1]) < 1e-20 for n in range(1, 60))
    # 2-3
    a_, b_, c_ = sp.sympify(load_answer(2, 2))
    ok23 = all(abs(I[n - 1] * I[n] - (b_ * n + c_) / (n + a_) * F(0)) < 1e-20 for n in range(1, 61))
    # 2-4: n^α I_n — α=md 에서 수렴(값 √(-F(0))), α=md+0.1 에서 증가
    am = sp.sympify(load_answer(2, 3))
    big = {n: mp.quad(lambda X: mp.sin(mp.pi * X / 2) ** n, [0, 1]) for n in (10**4, 10**6)}
    lim_ok = all(abs(n ** mp.mpf(float(am)) * big[n] - mp.sqrt(-F(0))) < 2e-2 for n in big)
    grow = (10**6) ** (float(am) + 0.1) * big[10**6] > 1.4 * (10**4) ** (float(am) + 0.1) * big[10**4]
    ok &= conds and ok21 and ok22 and ok23 and lim_ok and grow
    msgs.append(f"F 조건={conds}; 2-1 n≤60={ok21}; 2-2 (p,q,r)={p_, q_, r_} 성립={ok22}; 2-3 (a,b,c)={a_, b_, c_} 성립={ok23}; "
                f"2-4 n^{am}·I_n→√(-F(0))={lim_ok}, α>{am} 발산 경향={grow}")
    report(2, ok, "[증명-수치확인] " + "; ".join(msgs))


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
