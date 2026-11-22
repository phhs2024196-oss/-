#!/usr/bin/env python3
"""KNU-M25A (경북대 2025 논술(AAT) 모의고사 자연계열Ⅰ 수학 1·2) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산."""
import itertools
import math
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KNU-M25A"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return [sp.sympify(s.strip()) for s in str(v).split(";")]


def p01():
    oks, msgs = [], []
    # 1-1: 일대일 대응 f:X→X, f(x)≠x, f∘f=id 를 순열 전수로 셈
    cnt = []
    for n in (2, 3, 4):
        N = 2 * n
        cnt.append(sum(1 for p in itertools.permutations(range(N))
                       if all(p[x] != x and p[p[x]] == x for x in range(N))))
    ans = [int(v) for v in load_answer(1, 0)]
    oks.append(cnt == ans); msgs.append(f"a2,a3,a4 전수={cnt}")
    # 1-2: n=1..300 에서 ∫_1^n ln x dx ≤ ln(n!) [증명-수치확인]
    ok2 = all(n * math.log(n) - n + 1 <= math.lgamma(n + 1) + 1e-12 for n in range(1, 301))
    oks.append(ok2); msgs.append(f"1-2 n=1..300 [증명-수치확인]={ok2}")
    # 1-3: a_n 을 짝 짓기 수로(재귀 a_n=(2n-1)a_{n-1}) 독립 계산 → b_n, 그리고 c_n 의 양쪽 경계 비의 극한
    a = 1; ok_b = True
    for n in range(1, 30):
        a = (2 * n - 1) * a
        ok_b &= sp.factorial(2 * n) / (2 ** n * a) == sp.factorial(n)
    lo = lambda n: (mp.loggamma(n + 1) - n * mp.log(3)) / 2 / mp.loggamma(n + 1)
    hi = lambda n: (mp.loggamma(n + 1) + n * mp.log(3)) / 2 / mp.loggamma(n + 1)
    Ns = [mp.mpf(10) ** e for e in (6, 12, 24, 48)]
    gaps = [hi(N) - lo(N) for N in Ns]
    (lim,) = load_answer(1, 2)
    ok3 = ok_b and abs(lo(Ns[-1]) - float(lim)) < 1e-2 and abs(hi(Ns[-1]) - float(lim)) < 1e-2 and all(x > y for x, y in zip(gaps, gaps[1:]))
    oks.append(ok3); msgs.append(f"b_n=n! (n<30)={ok_b}, 경계(N=1e48)=[{mp.nstr(lo(Ns[-1]),6)},{mp.nstr(hi(Ns[-1]),6)}] 폭 감소")
    report(1, all(oks), "; ".join(msgs))


def p02():
    """증명 4개 — 수치 확인 [증명-수치확인]."""
    x = sp.symbols("x")
    f = (sp.exp(x) - 1) / x
    f1 = sp.simplify(sp.diff(f, x)); f2 = sp.simplify(sp.diff(f, x, 2))
    fN = sp.lambdify(x, f, "mpmath"); f1N = sp.lambdify(x, f1, "mpmath"); f2N = sp.lambdify(x, f2, "mpmath")
    F = lambda t: mp.mpf(1) if t == 0 else fN(t)
    F1 = lambda t: mp.mpf(1) / 2 if t == 0 else f1N(t)
    oks, msgs = [], []
    # 2-1 f''>0 on (0,50]
    xs = [mp.mpf(i) / 100 for i in range(1, 5001)]
    ok1 = all(f2N(t) > 0 for t in xs)
    oks.append(ok1); msgs.append(f"2-1 f''>0 (0,50] 5000점={ok1}")
    # 2-2 c 여러 값: h'=c f'-1 의 부호가 한 번(-→+) 바뀌고 그 점 k>0
    ok2 = True
    grid = [mp.mpf(i) / 50 for i in range(-1000, 1001)]
    for c in ("0.1", "0.5", "1", "1.5", "1.9"):
        c = mp.mpf(c)
        sg = [c * F1(t) - 1 for t in grid]
        changes = [grid[i] for i in range(len(sg) - 1) if sg[i] < 0 <= sg[i + 1] or sg[i] >= 0 > sg[i + 1]]
        ok2 &= len(changes) == 1 and changes[0] > 0 and sg[0] < 0 and sg[-1] > 0
        ok2 &= abs(c * F1(0) - 1 - (c - 2) / 2) < 1e-20          # (Ⅱ) h'(0)=(c-2)/2 와 일치
    oks.append(ok2); msgs.append(f"2-2 c 5개 [증명-수치확인]={ok2}")
    # 2-3 g=f-xf' 의 근: [-20,20] 부호변화 1개, (1,2) 안
    G = lambda t: F(t) - t * F1(t)
    sg = [G(t) for t in grid]
    roots = [mp.findroot(G, (grid[i], grid[i + 1]), solver="bisect") for i in range(len(sg) - 1) if sg[i] * sg[i + 1] < 0]
    ok3 = len(roots) == 1 and 1 < roots[0] < 2
    oks.append(ok3); msgs.append(f"2-3 근={[mp.nstr(r,10) for r in roots]}")
    # 2-4 0<b<β 에서 bf(x)=x 의 근 2개, f'(x1)<1/b<f'(x2)
    al = roots[0]; beta = 1 / F1(al)
    ok4 = True
    for frac in ("0.05", "0.3", "0.6", "0.9", "0.999"):
        b = beta * mp.mpf(frac)
        Q = lambda t: b * F(t) - t
        sq = [Q(t) for t in grid]
        rr = [mp.findroot(Q, (grid[i], grid[i + 1]), solver="bisect") for i in range(len(sq) - 1) if sq[i] * sq[i + 1] < 0]
        ok4 &= len(rr) == 2 and F1(rr[0]) < 1 / b < F1(rr[1])
    # b=β 에서는 중근(경계) — 조건 b<β 가 꼭 필요함을 확인
    b = beta; k = mp.findroot(lambda t: b * F1(t) - 1, al)
    ok4 &= abs(b * F(k) - k) < 1e-15
    oks.append(ok4); msgs.append(f"2-4 b 5개 [증명-수치확인]={ok4}, β={mp.nstr(beta,10)}(b=β 중근)")
    report(2, all(oks), "; ".join(msgs))


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
