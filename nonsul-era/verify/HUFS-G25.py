#!/usr/bin/env python3
"""한국외대 2025학년도 논술전형(자연계) 1~7 (HUFS-G25-01~07) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "HUFS-G25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_item(seq):
    import pathlib
    import yaml
    return yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/HUFS/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    v = load_item(seq)["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """[증명-수치확인] n=1 기저: 45^2 를 두 자연수 제곱합으로 전수 탐색, 귀납 단계 (a,b)->(45a,45b) 를 n=1..8 반복 확인,
    별해 항등식 (a^2+b^2)(27^2+36^2)=(27a+36b)^2+(36a-27b)^2 기호 확인, n=2,3 은 전수로 존재 재확인."""
    def reps(r):
        R2 = r * r
        out = []
        for u in range(1, r):
            v2 = R2 - u * u
            v = math.isqrt(v2)
            if v >= 1 and v * v == v2:
                out.append((u, v))
        return out
    base = reps(45)
    ok_base = (27, 36) in base
    pa, pb = 27, 36
    chain = True
    for m in range(2, 9):
        pa, pb = 45 * pa, 45 * pb
        chain &= pa > 0 and pb > 0 and pa * pa + pb * pb == (45 ** m) ** 2
    A_, B_ = sp.symbols("A B")
    ident = sp.expand((A_**2 + B_**2) * (27**2 + 36**2) - (27 * A_ + 36 * B_)**2 - (36 * A_ - 27 * B_)**2) == 0
    ident2 = sp.expand((A_**2 + B_**2) * (27**2 + 36**2) - (-27 * A_ + 36 * B_)**2 - (36 * A_ + 27 * B_)**2) == 0
    brute = all(len(reps(45 ** m)) > 0 for m in (2, 3))
    item = load_item(1)
    ok = ok_base and chain and ident and ident2 and brute and item["subparts"][0]["answer"]["kind"] == "proof"
    report(1, ok, f"[증명-수치확인] 45^2 의 자연수 제곱합 표현 {base}, (45a,45b) 사슬 n≤8 {chain}, 별해 항등식 {ident and ident2}, n=2,3 전수 존재 {brute}")


def p02():
    """정수 a=-30..30 전수: 각 조각의 치역(끝값·극한을 sympy 로 계산, 단조성은 도함수 부호로 확인)에서 정수 개수를 세어 32 인 a 를 찾는다."""
    sols = []
    left = 2**x
    tp = sp.Symbol("tp", positive=True)
    lo_l, hi_l = sp.limit(left.subs(x, -tp), tp, sp.oo), left.subs(x, 3)   # (0, 8) 열린 구간, 증가
    assert sp.simplify(sp.diff(left, x)) .subs(x, 0) > 0
    nl = len([m for m in range(-100, 100) if lo_l < m < hi_l])
    A_ = sp.Symbol("A", integer=True)
    gA = sp.Rational(1, 5)**(x + A_) - sp.Rational(1, 5)**(3 + A_) + 8
    assert sp.limit(sp.Rational(1, 5)**x, x, sp.oo) == 0                 # 점근선 = 상수항
    for av in range(-12, 13):
        g = gA.subs(A_, av)
        assert sp.diff(g, x).subs(x, 4) < 0                              # 감소
        top = g.subs(x, 3)                                               # 포함
        L = g.subs(x, 3) - sp.Rational(1, 5)**(3 + av)                   # 점근선(미포함)
        nr = int(top) - int(sp.floor(L))                                 # (L, top] 안의 정수 개수
        if nl + nr == 32:
            sols.append(av)
    ans = load_answer(2)
    report(2, sols == [ans], f"x<3 정수점 {nl}개, 개수 32 인 정수 a = {sols} (a∈[-12,12]), md={ans}")


def p03():
    """S1=∫_{-1}^{1}(2-f), S2=∫_{-1}^{1}(f+2) 를 직접 적분해 4 S1 = 5 S2 를 -1<a<0 에서 풀고, 곡선이 직사각형 안(|f|≤2)에 있는지 격자 확인."""
    f = (x + 1) * (x - a) * (x - 1)
    S1 = sp.integrate(2 - f, (x, -1, 1))
    S2 = sp.integrate(f + 2, (x, -1, 1))
    sols = [s for s in sp.solve(sp.Eq(4 * S1, 5 * S2), a) if -1 < s < 0]
    ans = load_answer(3)
    inside = all(abs(f.subs({a: ans, x: sp.Rational(i, 200)})) <= 2 for i in range(-200, 201))
    report(3, sols == [ans] and inside, f"S1={sp.expand(S1)}, S2={sp.expand(S2)}, 해={sols}, 직사각형 내부 {inside}, md={ans}")


def p04():
    """(다) 를 a=1 로 축척해 (b,c) 격자 [-6,6]² (0.25 간격)에서 (x-2)(x-1)(x²+bx+c)≥0 인 것만 남겨 유일성 확인 →
    f=(x-a)²(x-2a) (a>0 일반화는 x→ax 축척) 의 극솟값을 sympy 로 구해 -500 을 풀어 a 를 얻는다."""
    xs = [sp.Rational(i, 40) for i in range(-200, 321)]
    ok_bc = []
    for i in range(-24, 25):
        for j in range(-24, 25):
            bb, cc = sp.Rational(i, 4), sp.Rational(j, 4)
            if all((xv - 2) * (xv - 1) * (xv**2 + bb * xv + cc) >= 0 for xv in xs[::4]):
                if all((xv - 2) * (xv - 1) * (xv**2 + bb * xv + cc) >= 0 for xv in xs):
                    ok_bc.append((bb, cc))
    A_ = sp.Symbol("A", positive=True)
    f = (x - A_)**2 * (x - 2 * A_)
    crit = sp.solve(sp.diff(f, x), x)
    mins = [c for c in crit if sp.diff(f, x, 2).subs(x, c).is_positive]
    sols = sp.solve(sp.Eq(f.subs(x, mins[0]), -500), A_)
    ans = load_answer(4)
    ok = ok_bc == [(-3, 2)] and sols == [ans]
    report(4, ok, f"a=1 축척에서 조건 (다) 만족 (b,c)={ok_bc}, 극소점 x={mins}, 극솟값={sp.factor(f.subs(x, mins[0]))}, a={sols}, md={ans}")


def p05():
    """정수 n=-20..20 전수: f 의 x=0 연속을 극한으로 확인하고, g 가 x=a 에서 연속일 조건(좌극한=우극한=g(a))을
    각 조각 구간에서 solveset 으로 풀어 a 의 개수가 4 인 n 을 모은다."""
    fL, fR = 3 * x**2 + 12 * x + 8, 2 * x**2 - 8 * x + 8
    cont0 = sp.limit(fL, x, 0, "-") == sp.limit(fR, x, 0, "+") == fR.subs(x, 0)
    good = []
    for nv in range(-20, 21):
        roots = set()
        for piece, dom in ((fL, sp.Interval.open(-sp.oo, 0)), (fR, sp.Interval(0, sp.oo))):
            # x=a 에서: 좌극한 2f(a), 우극한·함숫값 6n-f(a) (f 가 연속이므로 a 근방에서 같은 조각식)
            eq = sp.Eq(sp.limit(2 * piece, x, t, "-"), sp.limit(6 * nv - piece, x, t, "+"))
            roots |= set(sp.solveset(eq.subs(t, x), x, dom))
        if len(roots) == 4:
            good.append(nv)
    ans = load_answer(5)
    report(5, cont0 and sum(good) == ans, f"f 의 x=0 연속 {cont0}, a 가 4개인 정수 n={good}, 합={sum(good)}, md={ans}")


def p06():
    """자연수 a≤12, b≤8, c≤10 전수: (0,2π) 에서 a sin(bx)+c = a² 의 해를 sympy solveset 으로 직접 구해 개수 5 인 (a,b,c) 를 모으고
    abc 의 최댓값·최솟값을 비교. (a≥4 이면 a²-c ≥ 6 > a 라 해 없음, b≥6 이면 해가 0 또는 ≥6 개라 범위 충분)"""
    triples = []
    dom = sp.Interval.open(0, 2 * sp.pi)
    for av in range(1, 13):
        for cv in range(1, 11):
            s = sp.Rational(av * av - cv, av)
            if abs(s) > 1:
                continue
            for bv in range(1, 9):
                sol = sp.solveset(sp.Eq(av * sp.sin(bv * x) + cv, av * av), x, dom)
                if isinstance(sol, sp.FiniteSet) and len(sol) == 5:
                    triples.append((av, bv, cv))
    prods = [p[0] * p[1] * p[2] for p in triples]
    ans = load_answer(6)
    ok = len(triples) > 0 and [max(prods), min(prods)] == list(ans)
    report(6, ok, f"해 5개인 (a,b,c) {len(triples)}쌍 {triples}, abc 최대 {max(prods)}·최소 {min(prods)}, md={ans}")


def p07():
    """P(-2,0) 을 지나는 직선 y=m(x+2) 와 y=|4x²+3x-1|/4 의 교점을 조각별 solveset 으로 정확히 구한다.
    교점 수가 바뀌는 m 후보(두 조각과의 판별식=0, 꺾인 점 통과)와 그 사이·바깥 표본 m 에서 교점 수를 세어
    정확히 3개인 m 이 유일한지 확인하고 x좌표 합을 비교. (x=-2 에서 수직선은 x=-2 한 점뿐이라 제외)"""
    m = sp.Symbol("m", real=True)
    q = (4 * x**2 + 3 * x - 1) / 4
    r1, r2 = sorted(sp.solve(q, x))
    pieces = [(q, sp.Interval(-sp.oo, r1)), (-q, sp.Interval.open(r1, r2)), (q, sp.Interval(r2, sp.oo))]

    def pts(mv):
        s = set()
        for g, dom in pieces:
            s |= set(sp.solveset(sp.Eq(mv * (x + 2), g), x, dom))
        return s
    crit = set()
    for g, _ in pieces:
        disc = sp.discriminant(sp.expand(4 * (g - m * (x + 2))), x)
        crit |= {c for c in sp.solve(disc, m) if c.is_real}
    for rr in (r1, r2):                       # 꺾인 점 (rr, 0) 을 지나는 기울기
        crit.add(sp.Rational(0) / (rr + 2))
    crit = sorted(crit)
    samples = set(crit) | {crit[0] - 1, crit[0] - 10, crit[-1] + 1, crit[-1] + 10}
    samples |= {(u + v) / 2 for u, v in zip(crit, crit[1:])}
    three = {mv: pts(mv) for mv in sorted(samples) if len(pts(mv)) == 3}
    ans = load_answer(7)
    ok = len(three) == 1 and sum(next(iter(three.values()))) == ans
    desc = {str(k): sorted(v) for k, v in three.items()}
    report(7, ok, f"후보 기울기 {crit}, 교점 3개인 기울기와 교점 x {desc}, 합={[sp.nsimplify(sum(v)) for v in three.values()]}, md={ans}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX.split('-', 1)[1]}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
