#!/usr/bin/env python3
"""SMWU-M23 (숙명여대 2023학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import sympy as sp

PREFIX = "SMWU-M23"
WIDTH = 2
RESULTS = []
x, n, k = sp.symbols("x n k", positive=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1-1) 리만 합 lim (1/n^4) sum k^3
    S = sp.summation((k / n) ** 3 * (1 / n), (k, 1, n))
    v1 = sp.limit(sp.simplify(S), n, sp.oo)
    ok1 = sp.simplify(v1 - sp.integrate(x**3, (x, 0, 1))) == 0 and sp.simplify(v1 - sp.sympify(md_answer(1, "1"))) == 0
    # (1-2) 원식을 그대로 수치 계산(N=12..18) + 적분값 비교
    import mpmath as mp
    mp.mp.dps = 30
    def orig(N):
        M = 2**N
        return mp.fsum(mp.log(((M + j) / mp.mpf(M)) ** (mp.mpf(1) / M)) for j in range(1, M + 1))
    I = sp.integrate(sp.log(x), (x, 1, 2))
    md = sp.sympify(md_answer(1, "2"))
    vals = [orig(N) for N in (12, 14, 16)]
    # 오른쪽 끝점 합 오차 ~ (ln2)/(2·2^N) → 감소 확인 + Richardson 외삽
    extrap = 2 * vals[2] - orig(15)
    ok2 = sp.simplify(I - md) == 0 and abs(extrap - mp.mpf(str(sp.N(md, 30)))) < 1e-8 and \
        abs(vals[0] - float(md)) > abs(vals[2] - float(md))
    report(1, ok1 and ok2, f"int x^3 = {v1}; int ln x = {sp.simplify(I)}; sum(N=16)={float(vals[2]):.8f}")


def p02():
    a, b, c = sp.symbols("a b c", positive=True)
    # (2-1) (1/2 ab sinC)^2 with cosC from law of cosines == Heron
    cosC = (a**2 + b**2 - c**2) / (2 * a * b)
    area2 = sp.Rational(1, 4) * a**2 * b**2 * (1 - cosC**2)
    s = (a + b + c) / 2
    ok1 = sp.simplify(sp.expand(area2 - s * (s - a) * (s - b) * (s - c))) == 0
    # (2-2) 둘레 3 고정, 무작위 삼각형 2만 개 → 넓이 최대 근방이 a=b=c 인지
    random.seed(1)
    H = lambda a_, b_, c_: (lambda s_: max(s_ * (s_ - a_) * (s_ - b_) * (s_ - c_), 0)) ((a_ + b_ + c_) / 2)
    best = None
    for _ in range(20000):
        u, v = sorted((random.random() * 3, random.random() * 3))
        tri = (u, v - u, 3 - v)
        if max(tri) >= 1.5:
            continue
        h = H(*tri)
        if best is None or h > best[0]:
            best = (h, tri)
    eq = H(1, 1, 1)
    ok_rand = best[0] <= eq + 1e-12 and max(abs(t - 1) for t in best[1]) < 0.05
    # 기호: 라그랑주 없이 s-a=X 등 X+Y+Z=s, 곱 최대 → X=Y=Z (AM-GM 등호)
    X, Y = sp.symbols("X Y", positive=True)
    P = X * Y * (sp.Rational(3, 2) - X - Y)
    crit = sp.solve([sp.diff(P, X), sp.diff(P, Y)], [X, Y], dict=True)
    ok_sym = {sp.Rational(1, 2)} == {cr[X] for cr in crit if cr[X] > 0} and all(cr[X] == cr[Y] for cr in crit if cr[X] > 0)
    md = md_answer(2, "2").replace(" ", "")
    ok2 = ok_rand and ok_sym and md == "a=b=c"
    report(2, ok1 and ok2, f"Heron identity={ok1}; best random tri={tuple(round(t, 3) for t in best[1])} area={best[0]**0.5:.5f} vs eq {eq**0.5:.5f}")


def p03():
    # (3-1) md 의 a_n, b_n 을 읽어 n=1..40 홀짝 조건 확인 + a_n 이 등비수열(공비 -1)
    parts = dict(seg.split("=", 1) for seg in md_answer(3, "1").replace(" ", "").split(";"))
    an = sp.sympify(parts["a_n"], locals={"n": n})
    bn = sp.sympify(parts["b_n"], locals={"n": n})
    ok1 = all(an.subs(n, m) == (1 if m % 2 else -1) and bn.subs(n, m) == (0 if m % 2 else 1) for m in range(1, 41)) \
        and all(sp.simplify(an.subs(n, m + 1) / an.subs(n, m)) == an.subs(n, 2) / an.subs(n, 1) for m in range(1, 10))
    # (3-2) [증명-수치확인] sum f_{2n} 기호 계산 = q/(1+q), p 격자 1000점에서 < 1/2
    p = sp.symbols("p", positive=True)
    q = 1 - p
    m = sp.symbols("m", integer=True, positive=True)
    S = sp.summation(p * q ** (2 * m - 1), (m, 1, sp.oo)).subs(p, sp.Rational(3, 10))
    closed = lambda pv: (1 - pv) / (2 - pv)
    ok_closed = sp.simplify(S - closed(sp.Rational(3, 10))) == 0
    # 카드 풀이1 의 최종식 1/2 - p/(2(2-p)) 와 일치
    ok_form = sp.simplify(sp.Rational(1, 2) - p / (2 * (2 - p)) - (1 - p) / (2 - p)) == 0
    grid = [i / 1001 for i in range(1, 1001)]
    ok_grid = all(closed(g) < 0.5 for g in grid)
    # 몬테카를로 직접 모의(p=0.3)
    random.seed(2)
    trials, even = 200000, 0
    for _ in range(trials):
        t = 1
        while random.random() >= 0.3:
            t += 1
        even += (t % 2 == 0)
    ok_mc = abs(even / trials - float(closed(0.3))) < 0.005
    ok2 = ok_closed and ok_form and ok_grid and ok_mc and md_answer(3, "2") is None
    report(3, ok1 and ok2, f"a_n,b_n ok={ok1}; sum f_2n(p=0.3)={S}; MC={even/trials:.4f} [증명-수치확인]")


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
