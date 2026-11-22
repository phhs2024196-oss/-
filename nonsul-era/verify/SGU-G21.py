#!/usr/bin/env python3
"""SGU-G21 (서강대 2021학년도 수시 논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import random
import re
import sys

import sympy as sp

PREFIX = "SGU-G21"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[증명-수치확인] 증명 4개 — 핵심 사실을 격자·이분법으로 확인."""
    x = sp.symbols("x", real=True)
    f = x - sp.sin(x)
    fp = sp.diff(f, x)
    fl = sp.lambdify(x, f, "math")
    N = 2000
    xs = [-math.pi + 2 * math.pi * i / N for i in range(N + 1)]
    # (1-1) f' >= 0, f'=0 은 x=0 뿐, 격자에서 순증가
    zeros = sp.solveset(sp.Eq(fp, 0), x, sp.Interval(-sp.pi, sp.pi))
    ok1 = zeros == sp.FiniteSet(0) and all(fl(a) < fl(b) for a, b in zip(xs, xs[1:]))
    # (1-2) f(-pi)=-pi, f(pi)=pi
    ok2 = f.subs(x, -sp.pi) == -sp.pi and f.subs(x, sp.pi) == sp.pi

    def g(y):  # 이분법 역함수
        lo, hi = -math.pi, math.pi
        for _ in range(200):
            mid = (lo + hi) / 2
            if fl(mid) < y:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    ys = [-math.pi + 2 * math.pi * i / 1000 for i in range(1, 1000) if i != 500]  # y=0 은 부동소수 상쇄로 제외(g(0)=0 은 기호로 자명)
    # (1-3) 홀함수
    ok3 = sp.simplify(f.subs(x, -x) + f) == 0 and all(abs(g(-y) + g(y)) < 1e-9 for y in ys)
    # (1-4) |g|^3 >= 6|y| 와 h''=f, h'(0)=h(0)=0
    t = sp.symbols("t", real=True)
    h = t**3 / 6 - t + sp.sin(t)
    ok4 = sp.simplify(sp.diff(h, t, 2) - f.subs(x, t)) == 0 and sp.diff(h, t).subs(t, 0) == 0 and h.subs(t, 0) == 0
    ok4 = ok4 and all(abs(g(y))**3 >= 6 * abs(y) - 1e-12 for y in ys)
    kinds = [md_answer(1, s) for s in "1234"]
    report(1, ok1 and ok2 and ok3 and ok4 and kinds == [None] * 4,
           f"[증명-수치확인] f'=0 at {zeros}, f(±pi)=±pi, g 홀함수, min(|g|^3-6|y|)={min(abs(g(y))**3 - 6*abs(y) for y in ys):.2e}")


def p02():
    m, n, k = sp.symbols("m n k", integer=True, positive=True)
    loc = {"m": m, "n": n, "k": k}
    # (2-1) 전수 열거
    a1 = sp.sympify(md_answer(2, "1"), locals=loc)
    ok1 = True
    for mv in range(1, 9):
        cnt = sum(1 for xv in range(mv + 1) for yv in range(2 * mv + 1) for zv in range(2 * mv + 1)
                  if 2 * mv - 2 * xv - yv - zv >= 0)
        ok1 &= cnt == a1.subs(m, mv)
    # (2-2)(2-3)(2-4) 전수 열거
    a2 = sp.sympify(md_answer(2, "2"), locals=loc)
    a3 = sp.sympify(md_answer(2, "3"), locals=loc)
    a4 = sp.sympify(md_answer(2, "4"), locals=loc)
    ok2 = ok3 = ok4 = True
    for nv in range(5, 21):
        good = [c for c in itertools.combinations(range(1, nv + 1), 3) if c[1] - c[0] >= 2 and c[2] - c[1] >= 2]
        for kv in range(1, nv - 3):
            ok2 &= sum(1 for c in good if c[0] == kv) == a2.subs({n: nv, k: kv})
        ok3 &= len(good) == a3.subs(n, nv)
        if nv % 2 == 0:
            ok4 &= sp.Rational(sum(1 for c in good if c[0] % 2 == 0), len(good)) == sp.nsimplify(a4.subs(n, nv))
    report(2, ok1 and ok2 and ok3 and ok4, "전수 열거 m=1..8, n=5..20 일치")


def p03():
    r, p, q, al, be, M = sp.symbols("r p q alpha beta M", positive=True)
    loc = {"r": r, "p": p, "q": q, "alpha": al, "beta": be, "M": M}
    # 결합분포: (감염?, 양성?) -> 확률
    J = {(1, 1): r * (1 - p), (1, 0): r * p, (0, 1): (1 - r) * q, (0, 0): (1 - r) * (1 - q)}
    PB = J[1, 1] + J[0, 1]
    ok1 = sp.simplify(PB - sp.sympify(md_answer(3, "1"), locals=loc)) == 0
    ok2 = sp.simplify(J[1, 0] / (J[1, 0] + J[0, 0]) - sp.sympify(md_answer(3, "2"), locals=loc)) == 0
    cost = {(1, 1): 1, (1, 0): 1 + al, (0, 1): 1 + be, (0, 0): 1}
    EX = sum(J[s] * cost[s] for s in J)
    ok3 = sp.simplify(EX - sp.sympify(md_answer(3, "3"), locals=loc)) == 0
    # (3-4) q 소거, 조건 대입 후 최소
    qq = sp.solve(sp.Eq(1 / p + 1 / q, M), q)[0]
    E4 = EX.subs(al, 4 * (1 - r) * be / r).subs(q, qq)
    crit = [s for s in sp.solve(sp.diff(E4, p), p) if sp.simplify(M * s - 1).subs(M, 5) > 0]
    pst = sp.simplify(crit[0]); qst = sp.simplify(qq.subs(p, pst))
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "4").replace(" ", "").split(";"))
    ok4 = len(crit) == 1 and sp.simplify(pst - sp.sympify(parts["p"], locals=loc)) == 0 \
        and sp.simplify(qst - sp.sympify(parts["q"], locals=loc)) == 0
    # 수치: 여러 M 에서 허용구간 1/(M-1)<p<1 격자 최소와 비교
    fE = sp.lambdify((p, M, r, be), E4)
    for Mv in (3.2, 4, 7.5, 20):
        lo = 1 / (Mv - 1)
        grid = [lo + (1 - lo) * i / 20000 for i in range(1, 20000)]
        pbest = min(grid, key=lambda pv: fE(pv, Mv, 0.3, 1.7))
        ok4 &= abs(pbest - float(pst.subs(M, Mv))) < 1e-3
    report(3, ok1 and ok2 and ok3 and ok4, f"E(X)={sp.expand(EX)}, p*={pst}, q*={qst}")


def p04():
    x, n = sp.symbols("x n", positive=True)
    rnd = random.Random(2021)
    # (4-1)(4-2) [증명-수치확인] 무작위 증가함수(양의 계수 다항식+exp 등)로 부등식 확인
    funcs = [x**3, sp.sqrt(x), sp.log(x), -1 / x, sp.exp(x / 3), x + sp.sin(x) / 2, -1 / sp.sqrt(x)]
    ok12 = True
    for fx in funcs:
        fl = sp.lambdify(x, fx, "mpmath")
        import mpmath as mp
        for nv in range(1, 31):
            S = sum(fl(kk) for kk in range(1, nv + 1))
            lhs = fl(1) + mp.quad(fl, [1, nv])
            rhs = mp.quad(fl, [1, nv + 1])
            ok12 &= (lhs <= S + 1e-20) and (S < rhs)
            if nv == 1:
                ok12 &= abs(lhs - S) < 1e-20
    for _ in range(20):  # 4-1: f<g 이면 적분 대소
        a, b = sorted(rnd.uniform(-3, 3) for _ in range(2))
        c = rnd.uniform(0.01, 2)
        ok12 &= sp.integrate(sp.sin(x), (x, a, b)) < sp.integrate(sp.sin(x) + c, (x, a, b))
    # (4-3) p=1..8 에서 sympy 극한
    pp = sp.symbols("p", positive=True, integer=True)
    kk = sp.symbols("k", positive=True, integer=True)
    a3 = sp.sympify(md_answer(4, "3"), locals={"p": pp})
    ok3 = all(sp.limit(sp.summation(kk**pv, (kk, 1, n)) / n**(pv + 1), n, sp.oo) == a3.subs(pp, pv) for pv in range(1, 9))
    # (4-4) 공식 부등식의 양 끝 극한 + 실제 합이 사이에 있음
    low = 2 * (sp.sqrt(1 + 1 / n) - 1 / sp.sqrt(n)); up = 2 - 1 / sp.sqrt(n)
    Lv = sp.limit(low, n, sp.oo); Uv = sp.limit(up, n, sp.oo)
    a4 = sp.sympify(md_answer(4, "4"))
    S = 0.0; inside = True
    for nv in range(1, 20001):
        S += 1 / math.sqrt(nv)
        c = S / math.sqrt(nv)
        inside &= float(low.subs(n, nv)) < c <= float(up.subs(n, nv)) + 1e-12 if nv <= 300 else True
    ok4 = Lv == Uv == a4 and inside and abs(c - float(a4)) < 2 / math.sqrt(20000)
    report(4, ok12 and ok3 and ok4, f"[증명-수치확인] 4-2 부등식 7함수×n≤30, 4-3 p=1..8, 4-4 bounds→{Lv},{Uv}, S_n/√n(n=2e4)={c:.5f}")


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
