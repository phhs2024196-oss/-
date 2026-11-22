#!/usr/bin/env python3
"""경북대 2027학년도 논술(AAT) 모의고사 자연계열Ⅰ (KNU-M27A-01~02) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp
import mpmath as mp

PREFIX = "KNU-M27A"
WIDTH = 2
RESULTS = []
x, t, s, c = sp.symbols("x t s c", positive=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return None if v is None else sp.sympify(v)


def p01():
    mp.mp.dps = 30
    p = x * ((x - 1) ** 2 + 1) - sp.exp((x ** 2 - 1) / 2)
    pf = sp.lambdify(x, p, "mpmath")
    # α: √2 보다 큰 근 — 격자로 부호 변화 전부 찾기(유일성 확인)
    grid = [mp.sqrt(2) + mp.mpf(k) / 1000 for k in range(1, 8001)]
    ch = [i for i in range(len(grid) - 1) if pf(grid[i]) * pf(grid[i + 1]) < 0]
    alpha = mp.findroot(pf, (grid[ch[0]], grid[ch[0] + 1]), solver="bisect")
    ok0 = len(ch) == 1
    # (1) p(1), k1 최솟값 = sup{x in (0,α) : p(x) <= 0}
    p1 = sp.simplify(p.subs(x, 1))
    g2 = [alpha * k / 20000 for k in range(1, 20000)]
    nonpos = [v for v in g2 if pf(v) <= mp.mpf("1e-25")]
    k1 = max(nonpos)
    k1_ok = abs(k1 - 1) < 1e-3 and all(pf(v) > 0 for v in g2 if v > 1 + mp.mpf("1e-3"))
    ans1 = load_answer(1, 0)
    ok1 = ok0 and p1 == ans1[0] and k1_ok and ans1[1] == 1
    # (2) x>α 에서 p<0 [증명-수치확인]: (α, 40] 격자 + 지수항 지배
    ok2 = all(pf(alpha + mp.mpf(k) / 200) < 0 for k in range(1, 8000))
    # 1-2(1): h(1)>h(c) 를 h 정의로 직접 계산, c 격자에서 성립 집합 = (1, α)
    def h(X, C):
        g = C ** 2 / 2 * (1 - 1 / X ** 2) - mp.log(X)
        return C ** 2 / X + X + mp.e ** g
    cs = [alpha * k / 4000 for k in range(1, 4000)]
    good = [C for C in cs if h(1, C) > h(C, C) + mp.mpf("1e-25")]
    k2 = min(good)
    ok3 = abs(k2 - 1) < 1e-3 and all(C > 1 for C in good) and load_answer(1, 2) == 1 \
        and all(h(1, C) > h(C, C) for C in cs if C > 1 + mp.mpf("1e-3"))
    # 1-2(2): √2<c<α 에서 h(x)=h(1) 의 (1,c) 근 개수 = 1 [증명-수치확인]
    counts = []
    for j in range(1, 30):
        C = mp.sqrt(2) + (alpha - mp.sqrt(2)) * j / 30
        xs = [1 + (C - 1) * k / 4000 for k in range(1, 4000)]
        vals = [h(X, C) - h(1, C) for X in xs]
        counts.append(sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0))
    ok4 = all(n == 1 for n in counts)
    report(1, ok0 and ok1 and ok2 and ok3 and ok4,
           f"alpha={mp.nstr(alpha,12)} (유일={ok0}), p(1)={p1}, k1min≈{mp.nstr(k1,6)}, (2) x>α p<0={ok2}, "
           f"k2min≈{mp.nstr(k2,6)}, (1,c) 근 개수={set(counts)} [증명-수치확인]")


def p02():
    X = sp.Symbol("X", real=True)
    L = sp.exp(t) + sp.diff(sp.exp(t), t) * (x - t)               # (가) 접선
    # 2-1 항등식
    ok1 = sp.simplify(sp.integrate((s - x) * sp.exp(s), (s, x, t)) - (sp.exp(x) - L)) == 0
    # 2-1 부등식 [증명-수치확인]: 격자
    d = sp.lambdify((x, t), sp.exp(x) - L, "mpmath")
    ok2 = True
    for T in [mp.mpf(k) / 7 for k in range(1, 50)]:
        for j in range(0, 41):
            Xv = T * j / 40
            v = d(Xv, T)
            lo, hi = (T - Xv) ** 2 / 2, mp.e ** T * (T - Xv) ** 2 / 2
            ok2 &= lo - mp.mpf("1e-20") <= v <= hi + mp.mpf("1e-20")
    # 2-2 S(t) 식, 증가
    S = sp.integrate(sp.exp(x) - L, (x, 0, t))
    ok3 = sp.simplify(S - load_answer(2, 1).subs(sp.Symbol("t"), t)) == 0
    dS = sp.simplify(sp.diff(S, t))
    ok3 &= sp.simplify(dS - t ** 2 * sp.exp(t) / 2) == 0
    # 2-3(1): t_n 유일·n t_n^3 <= 6 (n=1..200) [증명-수치확인]
    Sf = sp.lambdify(t, S, "mpmath")
    mp.mp.dps = 40
    ok4 = True
    for n in range(1, 201):
        tn = mp.findroot(lambda u: Sf(u) - mp.mpf(1) / n, (mp.mpf("1e-6"), mp.mpf(10)), solver="bisect")
        ok4 &= n * tn ** 3 <= 6
    # 2-3(2): n^4 T(t_n)^3 = T(t)^3 / S(t)^4 (n = 1/S(t_n)), t→0+ 극한
    Tt = sp.integrate(x * (sp.exp(x) - L), (x, 0, t))
    lim = sp.limit(Tt ** 3 / S ** 4, t, 0, "+")
    ok5 = sp.simplify(lim - load_answer(2, 3)) == 0
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"항등식={ok1}, 2-1 부등식 격자={ok2}, S(t)·S'={ok3}, n t_n^3<=6(n≤200)={ok4}, "
           f"lim T^3/S^4={lim}, md={load_answer(2,3)} [증명-수치확인]")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda q: int(q[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
