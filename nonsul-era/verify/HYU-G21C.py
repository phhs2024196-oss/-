#!/usr/bin/env python3
"""HYU-G21C (한양대 2021학년도 논술 자연계열 오후2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G21C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


BIG = "BIG"          # 2^2020 보다 큰 값(정확한 값은 너무 큼)
LIM = 2**2020


def step(v, odd):
    """ㄴ(짝수 첨자)·ㄷ(홀수 첨자) 한 번 적용. 작은 값은 정확히, 큰 값은 BIG 으로."""
    if v == BIG:
        return BIG    # X>2^2020 이면 두 식 모두 X 보다 큼(아래 p01 에서 확인)
    w = (v - 2022)**2020 + 2018 if odd else (v - 2020)**2021 + 2020
    return BIG if w > LIM else w


def a_of(k):
    v = 2021
    for bit in bin(k)[3:]:                  # 맨 앞 1 다음 비트: 0→ㄴ, 1→ㄷ
        v = step(v, bit == "1")
    return v


def p01():
    # BIG 유지: X>2^2020 이면 X-2020, X-2022 ≥ 2 이므로 (X-2020)^2021 ≥ (X-2020)^1, (X-2022)^2020 ≥ (X-2022)^2.
    # 따라서 하한식 (X-2020)+2020 = X 와 (X-2022)^2+2018 > X 를 확인하면 충분(둘째는 X>2^2020 전 범위에서 증가).
    X = sp.symbols("X", positive=True)
    low2 = (X - 2022)**2 + 2018 - X
    okbig = low2.subs(X, LIM) > 0 and sp.diff(low2, X).subs(X, LIM) > 0 and (LIM - 2020 - 1) ** 2021 > 0
    ok1 = a_of(36) == int(md_answer(1, "1"))
    # (2) k ≤ 2^100: 비트 길이별 상태 수 DP (상태: 2021, 2019, BIG)
    trans = {s: (step(s, False), step(s, True)) for s in (2021, 2019)}
    def count_upto_pow(N):                   # k ≤ 2^N 에서 상태별 개수
        from collections import Counter
        total, layer = Counter(), Counter({2021: 1})
        for _ in range(N):                   # 길이 1..N 비트 (k < 2^N)
            total += layer
            nxt = Counter()
            for s, c in layer.items():
                if s == BIG:
                    nxt[BIG] += 2 * c
                else:
                    for s2 in trans[s]:
                        nxt[s2] += c
            layer = nxt
        total[a_of(2**N)] += 1               # k = 2^N
        return total
    # DP 를 작은 N 에서 전수와 대조
    from collections import Counter
    okdp = all(count_upto_pow(N) == Counter(a_of(k) for k in range(1, 2**N + 1)) for N in range(1, 13))
    c100 = count_upto_pow(100)
    cnt2 = c100[2021] + c100[2019]
    ok2 = cnt2 == int(md_answer(1, "2"))
    # (3) α = 최솟값(2019), c_n 공식 = 전수/DP, S_n 극한
    alpha = min(s for s in (2021, 2019))
    nn = sp.symbols("nn", integer=True, positive=True)
    cpoly = sp.interpolate([(N, count_upto_pow(N)[alpha]) for N in range(3, 8)], nn)
    okc = all(cpoly.subs(nn, N) == count_upto_pow(N)[alpha] for N in range(3, 40))
    tt = sp.symbols("tt", integer=True, positive=True)
    Sn = sp.Sum(2 * (nn - 1) / (2 * cpoly + tt * (nn - 1)), (tt, 1, nn))
    Sval = [float(Sn.subs(nn, N).doit()) for N in (2000,)]
    L = sp.integrate(2 / (1 + x), (x, 0, 1))                 # 구분구적법 극한값
    ok3 = alpha == 2019 and okc and abs(Sval[0] - float(L)) < 1e-3 and \
        sp.simplify(L - sp.sympify(md_answer(1, "3"))) == 0
    report(1, okbig and okdp and ok1 and ok2 and ok3,
           f"a36={a_of(36)} count={cnt2} c_n={sp.factor(cpoly)} S_2000={Sval[0]:.6f} lim={L}")


def p02():
    import mpmath as mp
    mp.mp.dps = 40
    a, b = sp.symbols("a b", real=True)
    u = sp.sqrt(1 + sp.exp(x))
    f = a * u + sp.log(u - b) - sp.log(u + b)
    # f'(x) - sqrt(1+e^x) = 0 이 모든 x 에서 성립: 여러 x 값에서 연립
    eqs = [sp.nsimplify(sp.simplify((sp.diff(f, x) - u).subs(x, sp.log(v)))) for v in (1, 2, 3)]
    sols = [s for s in sp.solve(eqs[:2], [a, b], dict=True)
            if s[a].is_positive and 0 <= s[b] <= 1 and sp.simplify(eqs[2].subs(s)) == 0]
    sol = sols[0]
    ok1 = len(sols) == 1 and sol[a] + sol[b] == sp.sympify(md_answer(2, "1"))
    f0 = f.subs(sol)
    okf = all(abs(float((sp.diff(f0, x) - u).subs(x, v))) < 1e-12 for v in (-3, 0, 2.5, 7))
    # (2) g(k) 수치(고정밀): 곡선 y=e^x 의 길이
    def g(k):
        return mp.quad(lambda s: mp.sqrt(1 + mp.e**(2 * s)), [k, k + mp.e**(-k)])
    gv = [g(k) for k in (10, 20, 30)]
    ok2 = abs(gv[-1] - mp.mpf(sp.sympify(md_answer(2, "2")))) < 1e-9 and abs(gv[1] - 1) < abs(gv[0] - 1)
    # (3) F(2x)/e^x, F(x)=∫_0^x f
    fl = sp.lambdify(x, f0, "mpmath")
    def ratio(X):
        return mp.quad(fl, [0, 1, 10, 2 * X]) / mp.e**X
    r = [ratio(X) for X in (20, 40)]
    ok3 = abs(r[-1] - mp.mpf(sp.sympify(md_answer(2, "3")))) < 1e-9 and abs(r[1] - 4) < abs(r[0] - 4)
    report(2, ok1 and okf and ok2 and ok3,
           f"a,b={sol[a]},{sol[b]} g(30)={mp.nstr(gv[-1], 15)} F(80)/e^40={mp.nstr(r[-1], 15)}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
