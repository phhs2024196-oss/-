#!/usr/bin/env python3
"""CAU-G22C (2022학년도 중앙대 수시 논술 자연계열Ⅲ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G22C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """카드 a(1~4)·주사위 b(1~6) 24쌍 전수: |OA²-OB²| 최소 순서쌍 (답 (1,2),(2,3))."""
    f = lambda X: sp.Integer(2) ** (X - 1) + 2
    d = {}
    for a in range(1, 5):
        for b in range(1, 7):
            yb = sp.sqrt(9 - (b - 3) ** 2)          # y≥0 반원과 x=b 교점
            d[(a, b)] = abs((a ** 2 + f(a) ** 2) - (b ** 2 + yb ** 2))
    mn = min(d.values())
    sols = sorted(k for k, v in d.items() if v == mn)
    report(1, sols == [(1, 2), (2, 3)], f"최소 차 {mn}, 해 {sols}")


def p02():
    """(2-1) 극한 조건에서 f(k)=k, f'(k)=k^{3/2} 를 기호로 얻어 합 + 조건 만족 예시함수로 수치 확인, (2-2) I_n/n 을 n=1..60 에서 직접 계산 (답 44100, 2/3)."""
    n = sp.symbols("n", integer=True, positive=True)
    k = sp.symbols("k", positive=True)
    fk = sp.solve(sp.Eq(sp.Symbol("F") - k, 0), sp.Symbol("F"))[0]      # 분자→0 ⇒ f(k)=k
    fpk = fk * sp.sqrt(fk)                                                 # f'(k)=f(k)√f(k)
    gp = fpk.subs(k, fk) * fpk                                             # g'(n)=f'(f(n))f'(n)
    S = sum(gp.subs(k, i) for i in range(1, 21))
    # 구체적 예(조건 만족 함수)로 수치 확인: f(x)=x+(x^{3/2}-1)sin(2πx)/(2π) ⇒ f(k)=k, f'(k)=k^{3/2}
    X = sp.symbols("X", positive=True)
    fex = X + (X ** sp.Rational(3, 2) - 1) * sp.sin(2 * sp.pi * X) / (2 * sp.pi)
    ok_ex = all(sp.simplify(fex.subs(X, i) - i) == 0 and
                sp.simplify(sp.diff(fex, X).subs(X, i) - i * sp.sqrt(i)) == 0 for i in range(1, 6))
    g = fex.subs(X, fex)
    S2 = sum(sp.N(sp.diff(g, X).subs(X, i), 30) for i in range(1, 21))
    ok1 = S == 44100 and ok_ex and abs(S2 - 44100) < 1e-15
    # (2-2)
    import mpmath as mp
    mp.mp.dps = 25
    h = lambda t: abs(mp.sin(t)) * mp.cos(t) ** 2 + mp.sin(2 * t) ** 5 * mp.cos(t)
    seg = [mp.quad(h, [kk * mp.pi, kk * mp.pi + mp.pi / 2, (kk + 1) * mp.pi]) for kk in range(60)]
    In = [sum(seg[:m]) for m in range(1, 61)]
    J1 = mp.quad(lambda t: mp.sin(2 * t) ** 5 * mp.cos(t), [0, mp.pi / 2, mp.pi])
    ok_J = abs(J1 - mp.mpf(512) / 693) < 1e-18
    ratios = [In[m - 1] / m for m in (59, 60)]
    exact60 = 60 * mp.mpf(2) / 3
    ok2 = ok_J and abs(In[59] - exact60) < 1e-15 and abs(In[58] - (59 * mp.mpf(2) / 3 + J1)) < 1e-15
    report(2, ok1 and ok2, f"(1) 합 {S}, 예시함수 수치합 {mp.nstr(S2, 10)} / (2) I_59/59={mp.nstr(ratios[0], 10)}, I_60/60={mp.nstr(ratios[1], 10)}, ∫_0^π sin^5(2x)cosx={mp.nstr(J1, 12)}")


def p03():
    """(3-1) 대칭점 대신 ℓ∩{y=f^{-1}(x)} 를 수치로 직접 구해 넓이 최대화, (3-2) 원 위 P 를 각으로 훑어 x,y 를 연립으로 풀어 최대·최소 (답 A, 1/3)."""
    import mpmath as mp
    mp.mp.dps = 25
    f = lambda X: X + 1 / (2 * (X + 1) ** 2)
    def finv(Y):                                 # f 는 x≥0 에서 증가(f'>0) ⇒ 이분법
        lo, hi = mp.mpf(0), Y + 1
        for _ in range(120):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < Y else (lo, mid)
        return (lo + hi) / 2
    def area(t):
        ya = f(t)
        c = t + ya                               # ℓ: x+y=c
        lo, hi = mp.mpf(f(0)), c                 # B: finv(X)+X=c, 좌변 증가 ⇒ 이분법
        for _ in range(120):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if finv(mid) + mid < c else (lo, mid)
        xb = (lo + hi) / 2
        yb = c - xb
        return abs(t * yb - ya * xb) / 2
    lo, hi = mp.mpf("0.1"), mp.mpf("1.5")       # 황금분할 탐색(단봉 구간)
    gr = (mp.sqrt(5) - 1) / 2
    for _ in range(80):
        c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        if area(c1) > area(c2):
            hi = c2
        else:
            lo = c1
    tst = (lo + hi) / 2
    tA = (mp.sqrt(5) - 1) / 2
    coarse = max((mp.mpf(i) / 100 for i in range(0, 300)), key=area)
    ok1 = abs(tst - tA) < 1e-8 and abs(f(tA) - (mp.sqrt(5) + 1) / 4) < 1e-20 and abs(coarse - tA) < 0.02
    vals = []
    for i in range(1, 20000):
        th = 2 * mp.pi * i / 20000
        a, b = 2 + mp.cos(th), mp.sin(th)
        PA = (1 - a, -b); PB = (3 - a, -b); PC = (-a, 2 - b)
        if abs(PA[0] * PB[0] + PA[1] * PB[1]) > 1e-15 or abs(b) < 1e-9:
            continue
        det = PA[0] * PB[1] - PA[1] * PB[0]
        xx = (PC[0] * PB[1] - PC[1] * PB[0]) / det
        yy = (PA[0] * PC[1] - PA[1] * PC[0]) / det
        vals.append((xx + yy) / (xx + 3 * yy))
    M, m = max(vals), min(vals)
    ok2 = abs(m * M ** 2 + M * m ** 2 - mp.mpf(1) / 3) < 1e-6 and abs(M - (4 + mp.sqrt(7)) / 6) < 1e-6
    report(3, ok1 and ok2, f"(1) t*={mp.nstr(tst, 12)}, A=({mp.nstr(tst, 8)}, {mp.nstr(f(tst), 8)}) / (2) M≈{mp.nstr(M, 8)}, m≈{mp.nstr(m, 8)}, mM²+Mm²≈{mp.nstr(m*M**2+M*m**2, 10)}")


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
