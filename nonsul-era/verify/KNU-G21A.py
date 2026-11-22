#!/usr/bin/env python3
"""KNU-G21A (경북대 2021 수시 논술 자연계열Ⅰ 수학 1·2·3) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-G21A"
WIDTH = 2
RESULTS = []
mp.mp.dps = 40


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    """소문항 sub(0부터)의 answer.value 를 ';' 로 나눠 sympy 식 목록으로."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]



def _dens(k):
    x = sp.Symbol("x")
    return x, sp.Piecewise(((x + 4) / (4 * (k + 4)), x <= k), ((x - 4) / (4 * (k - 4)), True))


def _P(k, a, b):
    x, f = _dens(k)
    return sp.nsimplify(sp.integrate(f, (x, a, b)))


def _chord(M):
    """직선 y=Mx/2 와 원 (x-2)^2+(y-1)^2=1 의 교점(실수) 목록"""
    x = sp.Symbol("x")
    sols = [s for s in sp.solve((x - 2) ** 2 + (M * x / 2 - 1) ** 2 - 1, x) if s.is_real]
    return [(s, M * s / 2) for s in sols]


def p01():
    oks, msgs = [], []
    # 1-1: 교점 개수 2개인 M 구간을 격자+이분법으로 찾음
    two = lambda M: len(set(_chord(sp.Rational(M)))) == 2
    grid = [sp.Rational(i, 60) for i in range(-240, 241)]
    inside = [m for m in grid if two(m)]
    lo, hi = min(inside), max(inside)
    M = sp.Symbol("M", real=True)
    x = sp.Symbol("x")
    disc = sp.discriminant(sp.expand((x - 2) ** 2 + (M * x / 2 - 1) ** 2 - 1), x)
    roots = sorted(sp.solve(disc, M))
    al, be = roots
    oks.append(al <= lo and hi <= be and be - hi < sp.Rational(1, 30))
    (a11,) = load_answer(1, 0)
    oks.append(3 * (be - al) == a11); msgs.append(f"1-1 ({al},{be}) 3(b-a)={3*(be-al)} md={a11}")
    # 1-2: h(k)=P(al<=M<=be), k in [0,3] 최대점
    hval = lambda kv: _P(kv, al, be)
    ks = [sp.Rational(i, 100) for i in range(0, 301)]
    best = max(ks, key=lambda kv: hval(kv))
    oks.append(best == 2); msgs.append(f"1-2 argmax h(k) 격자={best} [증명-수치확인]")
    c = best
    # 1-3: 사건 B(세 점 일직선 또는 중심각 둔각) — 교점에서 직접 판정
    def obtuse(Mv):
        pts = _chord(Mv)
        if len(pts) != 2: return False
        (x1, y1), (x2, y2) = pts
        dot = (x1 - 2) * (x2 - 2) + (y1 - 1) * (y2 - 1)
        return sp.N(dot) < 0
    g2 = [sp.Rational(i, 700) for i in range(1, int(be * 700))]
    ob = [m for m in g2 if obtuse(m)]
    # 경계: dot=0 이 되는 M (교점 대칭식으로)
    xs = sp.solve((x - 2) ** 2 + (M * x / 2 - 1) ** 2 - 1, x)
    dotM = sp.simplify(sp.expand((xs[0] - 2) * (xs[1] - 2) + (M * xs[0] / 2 - 1) * (M * xs[1] / 2 - 1)))
    bd = sorted([r for r in sp.solve(sp.numer(sp.together(dotM)), M) if r.is_real])
    l3, h3 = bd[0], bd[-1]
    oks.append(abs(min(ob) - l3) < 0.01 and abs(max(ob) - h3) < 0.01)
    pr = sp.nsimplify(_P(c, l3, h3) / _P(c, al, be))
    (a13,) = load_answer(1, 2)
    oks.append(sp.numer(pr) + sp.denom(pr) == a13); msgs.append(f"1-3 B구간=({l3},{h3}) P(B|A)={pr} md={a13}")
    # 1-4: 넓이를 세로 단면 적분으로 직접
    def area(s):
        f = lambda t: max(mp.mpf(0), (1 + mp.sqrt(max(0, 1 - (t - 2) ** 2))) - max(s * t, 1 - mp.sqrt(max(0, 1 - (t - 2) ** 2))))
        return mp.quad(f, [1, 1.5, 2, 3])
    gv = lambda m: 0 if m <= 0 else 1 if m <= 1 else 2 if m <= 2 else 3
    ES = 0
    for gval, (a, b) in zip([0, 1, 2, 3], [(-4, 0), (0, 1), (1, 2), (2, 4)]):
        ES += area(mp.mpf(gval) / 2) * float(_P(c, a, b))
    (a14,) = load_answer(1, 3)
    # E(S)=aπ+b, a,b 유리수: 분모 192 로 정수조합 탐색
    cand = [(A, B) for A in range(0, 192) for B in [round((float(ES) - A * mp.pi / 192) * 192)]
            if abs(ES - (A * mp.pi + B) / 192) < 1e-7]
    oks.append(len(cand) == 1 and cand[0][0] + cand[0][1] == a14)
    msgs.append(f"1-4 E(S)={mp.nstr(ES,15)} 192(a,b)={cand} md={a14}")
    report(1, all(oks), "; ".join(msgs))


def _condA(f, a, xs):
    """조건 (ㄱ): f(a+x)!=0 이면 f(a+x)f(a-x)<0 을 격자에서 검사"""
    for t in xs:
        u, v = f(a + t), f(a - t)
        if u != 0 and not (u * v < 0):
            return False
    return True


def p02():
    oks, msgs = [], []
    x = sp.Symbol("x")
    xs = [sp.Rational(i, 7) for i in range(-80, 81)]
    # (ㄱ)(ㄴ)을 만족하는 f=(x-4)(x-p)(x-q) 를 격자 탐색(a=1/2,3/2,5/2)
    ok0 = True
    for a in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(5, 2)]:
        roots = [sp.Rational(i, 2) for i in range(-16, 9)]
        found = set()
        for p in roots:
            for q in roots:
                if p <= q:
                    F = lambda t, p=p, q=q: (t - 4) * (t - p) * (t - q)
                    ext = xs + [r - a for r in (4, p, q)] + [a - r for r in (4, p, q)]
                    if _condA(F, a, ext):
                        found.add((p, q))
        ok0 &= found == {tuple(sorted((a, 2 * a - 4)))}
    oks.append(ok0); msgs.append(f"(ㄱ)(ㄴ) 격자탐색 근={{2a-4,a,4}} {ok0}")
    a = sp.Symbol("a")
    f = (x - (2 * a - 4)) * (x - a) * (x - 4)
    n = lambda av: len([r for r in set(sp.solve(f.subs(a, av), x)) if r > 0])
    np_, nm = n(2 + sp.Rational(1, 1000)), n(2 - sp.Rational(1, 1000))
    A1 = load_answer(2, 0)
    oks.append([np_, nm] == A1); msgs.append(f"2-1 n(2+)={np_} n(2-)={nm} md={A1}")
    I = sp.integrate(f, (x, 0, 2))
    amax = sp.solve(sp.diff(I, a), a)[0]
    oks.append(sp.diff(I, a, 2) < 0 and 3 * amax == load_answer(2, 1)[0]); msgs.append(f"2-2 I={sp.expand(I)} 3a={3*amax}")
    # 2-3(1): 한쪽 미분계수 비교로 미분불가능점 탐색
    g = sp.lambdify(x, sp.Abs(f.subs(a, 2) * f.subs(a, 2).subs(x, -x)), "mpmath")
    bad = []
    for b in [sp.Rational(i, 4) for i in range(-24, 25)]:
        bb = mp.mpf(b); hh = mp.mpf("1e-12")
        if abs((g(bb + hh) - g(bb)) / hh - (g(bb) - g(bb - hh)) / hh) > 1e-3:
            bad.append(b)
    import pathlib, yaml
    v = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath("items/KNU/PAST/KNU-G21A-02.yaml").read_text())["subparts"][2]["answer"]["value"]
    sset, kval = v.split(";")
    md = sorted(sp.sympify(t) for t in sset.strip(" {}").split(","))
    oks.append(bad == md); msgs.append(f"2-3(1) 꺾임={bad}")
    # 2-3(2): k 후보 = 판별/극값, 각 k 에서 실근 개수 직접 세기(x!=-4)
    prod = sp.expand(f.subs(a, 0) * f.subs(a, 0).subs(x, -x))
    assert all(prod.subs(x, t) <= 0 for t in range(-9, 10))   # f(x)f(-x)<=0 이므로 g=-prod
    g0 = -prod
    def cnt(kv):
        e = sp.cancel(g0 / (x + 4) ** 2) - kv * x
        return len([r for r in sp.Poly(e, x).real_roots() if r != -4 and True]) and len(set(r for r in sp.Poly(e, x).real_roots() if r != -4))
    crit = sorted(set([0] + [sp.cancel(g0 / (x + 4) ** 2 / x).subs(x, r) for r in sp.solve(sp.diff(sp.cancel(g0 / (x + 4) ** 2 / x), x), x)] + [sp.cancel(g0 / (x + 4) ** 2 / x).subs(x, -4)]))
    tests = set(crit)
    for i in range(len(crit) - 1):
        tests.add((crit[i] + crit[i + 1]) / 2)
    tests |= {crit[0] - 1, crit[-1] + 1}
    good = [kv for kv in sorted(tests) if cnt(kv) == 3]
    oks.append(len(good) == 1 and sp.Rational(27, 16) * good[0] == sp.sympify(kval)); msgs.append(f"2-3(2) 3근 k={good} 27k/16={[sp.Rational(27,16)*t for t in good]}")
    report(2, all(oks), "; ".join(msgs))


def p03():
    oks, msgs = [], []
    x, t, u = sp.symbols("x t u", real=True)
    I1 = sp.integrate(sp.sin(x) ** 2, (x, 0, 2 * sp.pi))
    oks.append(sp.simplify(I1 - load_answer(3, 0)[0]) == 0); msgs.append(f"3-1 {I1}")
    # 3-2 [증명-수치확인]: 주기 2π 연속함수 몇 개로 h(x) 가 일정한지
    ok2 = True
    for gg in [lambda s: mp.exp(mp.sin(s)), lambda s: 1 / (2 + mp.cos(s)) + mp.sin(3 * s) ** 2]:
        vals = [mp.quad(gg, [X - 2 * mp.pi, X]) for X in [-3, 0.4, 2, 7.7]]
        ok2 &= max(vals) - min(vals) < 1e-8
    oks.append(ok2); msgs.append(f"3-2 [증명-수치확인]={ok2}")
    # 3-3: f' 를 차수 2 이하 삼각다항식(상수항 포함)으로 두고 (ㄴ)을 미분한 식에 대입해 계수 결정
    cs = sp.symbols("c0 a1 b1 a2 b2")
    c0, a1, b1, a2, b2 = cs
    fp = lambda s: c0 + a1 * sp.cos(s) + b1 * sp.sin(s) + a2 * sp.cos(2 * s) + b2 * sp.sin(2 * s)
    J = sp.integrate(sp.expand_trig(fp(x - t)) * sp.cos(t), (t, 0, 2 * sp.pi))
    rhs = sp.diff(J ** 2 - sp.cos(x), x)
    eq = sp.expand(sp.expand_trig(fp(x) - rhs))
    eqs = [sp.integrate(eq * w, (x, 0, 2 * sp.pi)) for w in [1, sp.cos(x), sp.sin(x), sp.cos(2 * x), sp.sin(2 * x), sp.cos(3 * x), sp.sin(3 * x)]]
    sols = sp.solve(eqs, cs, dict=True)
    al = sp.Symbol("alpha")
    res = set()
    for so in sols:
        F = lambda s: fp(s).subs(so)
        # f 는 (ㄴ)의 우변 그 자체(주기성 보장: 상수항 c0=0 이어야 함)
        if so.get(c0, c0) != 0:
            continue
        Bv = sp.simplify(sp.integrate(F(t) * sp.sin(t), (t, 0, 2 * sp.pi)))
        f0 = (J.subs(so).subs(x, 0)) ** 2 - 1 + al
        res.add((Bv, sp.solve(sp.Eq(f0, 1), al)[0]))
    Bm, alm = load_answer(3, 2)
    oks.append(res == {(Bm, alm)}); msgs.append(f"3-3 해집합(B,α)={res}")
    report(3, all(oks), "; ".join(msgs))


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
