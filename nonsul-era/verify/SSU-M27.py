#!/usr/bin/env python3
"""<학교>-<유형> 세트 <n> (<ID 범위>) — sympy 검산. 파일명: verify/<학교>-<유형>-s<n>.py
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M27"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, idx=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return sp.sympify(d["subparts"][idx]["answer"]["value"])


def p01():
    """삼차함수 2(x-k)^2(x-m) 중 (나) 조건을 만족하는 것을 정수 격자 + 기호 해로 독립 탐색."""
    k, m = sp.symbols("k m", real=True)
    f = lambda X: 2 * (X - k) ** 2 * (X - m)
    sols = []
    for cond in sp.solve([f(3), f(5) + 8], [k, m], dict=True):   # (나): f(3)=0, f(5)=-8
        if not (cond[k].is_real and cond[m].is_real):
            continue
        g = f(x).subs(cond)
        if g.subs(x, 10) == 0:
            continue
        L1 = sp.limit(g.subs(x, x + 5) / ((g + 8) * (x + 2)), x, -2)
        L2p = sp.limit(g.subs(x, x + 5) / ((g + 8) * (x + 2)), x, 5, "+")
        if L1.is_finite and not L2p.is_finite:
            sols.append(g.subs(x, 1))
    ans = load_answer(1)
    ok = len(set(sols)) == 1 and sp.simplify(sols[0] - ans) == 0
    report(1, ok, f"f(1) 후보={set(sols)}, md={ans}")


def p02():
    """교점 조건을 수치로 풀어 b 를 구하고 기울기 곱을 직접 계산."""
    import mpmath as mp
    mp.mp.dps = 30
    def alpha_of(b):   # f(x)=b^x 와 g(x)=-log_b x 의 교점 x좌표(증가-감소 → 유일), 이분법
        h = lambda a: b ** a + mp.log(a) / mp.log(b)
        return mp.findroot(h, (mp.mpf("1e-30"), mp.mpf(1)), solver="anderson")
    G = lambda b: alpha_of(b) * (b ** alpha_of(b)) ** 2 - 1
    b0 = mp.findroot(G, (mp.mpf("1.01"), mp.mpf(50)), solver="anderson")
    a0 = alpha_of(b0)
    s1 = b0 ** a0 * mp.log(b0)
    s2 = -1 / (a0 * mp.log(b0))
    ans = load_answer(2)
    ok = b0 > 1 and abs(s1 * s2 - float(ans)) < 1e-20
    report(2, ok, f"alpha={mp.nstr(a0,10)}, b={mp.nstr(b0,10)}, 곱={mp.nstr(s1*s2,15)}, md={ans}")


def p03():
    """사다리꼴 넓이 b_n 을 정의대로 만들고 두 조건을 연립해 r 을 기호 계산."""
    L, R = sp.symbols("L R", positive=True)          # L = ln a1, R = ln r
    k = sp.symbols("k", integer=True, positive=True)
    fa = lambda m: 2 * (L + (m - 1) * R)               # f(a_m) = 2 ln(a1 r^{m-1})
    b = (fa(k) + fa(k + 1)) / 2 * 1                    # 높이 Q_kQ_{k+1}=1 인 사다리꼴
    S = sp.summation(sp.expand(b), (k, 1, 2027))
    sol = sp.solve([sp.Eq(fa(2027), 19), sp.Eq(S, 20270)], [L, R], dict=True)
    ok = len(sol) == 1 and sol[0][L] > 0 and sol[0][R] > 0
    r = sp.exp(sol[0][R])
    ans = load_answer(3)
    ok = ok and sp.simplify(r - ans) == 0
    report(3, ok, f"ln a1={sol[0][L]}, ln r={sol[0][R]}, r={r}, md={ans}")


def p04():
    """양의 정수 a,b (1..60) 전수: 적분을 부호 변화점에서 나누어 정확히 계산 + mpmath 수치 교차확인."""
    import mpmath as mp
    cnt, pairs = 0, []
    for a in range(1, 61):
        for b in range(1, 61):
            g = a * x + b
            r0 = sp.Rational(-b, a)
            if r0 <= -1:
                I = sp.integrate(g, (x, -1, 1))
            else:
                I = sp.integrate(-g, (x, -1, r0)) + sp.integrate(g, (x, r0, 1))
            if I == 10:
                assert abs(mp.quad(lambda X: abs(a * X + b), [-1, float(r0) if r0 > -1 else 0, 1]) - 10) < 1e-12
                pairs.append((a, b))
    ans = load_answer(4)
    ok = len(pairs) == ans
    report(4, ok, f"순서쌍={pairs} ({len(pairs)}개), md={ans}")


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
