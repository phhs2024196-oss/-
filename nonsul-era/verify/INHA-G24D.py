#!/usr/bin/env python3
"""INHA-G24D (2024학년도 인하대 논술 자연계열 오전 의예과) — sympy 검산. [문제 1]은 일반 [문제 3](INHA-G24A-03)과 동일해 그쪽에서 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp
from scipy import integrate as si
from scipy import optimize as so

PREFIX = "INHA-G24D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _roots(tv):
    """t=tv 에서 f(x)=0 의 양의 실근(수치)."""
    rs = []
    B = 5 + tv
    dsc = B * B - 16
    if dsc >= 0:
        for s in ((B - np.sqrt(dsc)) / 2, (B + np.sqrt(dsc)) / 2):
            if 0 < s <= 4:
                rs.append(s)
    if tv > -2:
        s = 8 / (2 + tv)
        if s > 4:
            rs.append(s)
    return sorted(set(rs))


def _F(tv, beta):
    """∫_0^β f(x)dx (닫힌 꼴을 직접 적분해 얻은 식)."""
    F4 = lambda b: b**3 / 3 - (5 + tv) * b**2 / 2 + 4 * b
    if beta <= 4:
        return F4(beta)
    return F4(4) - (2 + tv) * (beta**2 - 16) / 2 + 8 * (beta - 4)


def p02():
    """(2-1) c=-2 (2-2) g(0)=11/6 (2-3)(a) p=-1 (b) r=-2/7."""
    # 닫힌 꼴 _F 를 sympy 적분과 대조
    T = sp.symbols("T", real=True)
    fx = sp.Piecewise(((x - 1) * (x - 4) - T * x, x <= 4), (-2 * (x - 4) - T * x, True))
    okF = all(abs(float(sp.integrate(fx.subs(T, tv), (x, 0, bv))) - _F(tv, bv)) < 1e-9
              for tv, bv in [(0.3, 2.5), (-0.5, 5.2), (1.7, 7.0)])
    # (2-1)
    no_root = [tv for tv in np.arange(-12, 3.0001, 0.25) if len(_roots(tv)) == 0]
    c = max(no_root)
    ok1 = abs(c + 2) < 1e-12 and len(_roots(-1.999)) > 0
    # (2-2) t=0: 근 1,4
    vals0 = sorted(sp.integrate((x - 1) * (x - 4), (x, 0, b)) for b in (1, 4))
    others = [min(_F(tv, b) for b in _roots(tv)) for tv in np.arange(-1.999, 8, 0.001) if abs(tv) > 1e-9]
    g0 = max(vals0)
    ok2 = vals0 == [sp.Rational(-8, 3), sp.Rational(11, 6)] and min(others) > float(vals0[0]) and g0 == sp.Rational(11, 6)
    # (2-3)(a) 세 실근 구간의 하한 = 접할 때
    disc = sp.discriminant((x - 1) * (x - 4) - T * x, x)
    pa = min(s for s in sp.solve(disc, T) if s > -2)
    ok3a = pa == -1 and len(_roots(-0.5)) == 3 and len(_roots(-1.1)) == 1
    # (2-3)(b) (-1,0)에서 둘째·셋째 근의 적분값이 0 이 되는 t
    zs = []
    for k in (1, 2):
        Fk = lambda v: _F(v, _roots(v)[k])
        grid = np.linspace(-0.9999, -0.0001, 2000)
        vals = [Fk(v) for v in grid]
        for i in range(len(grid) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]):
                zs.append(so.brentq(Fk, grid[i], grid[i + 1], xtol=1e-14))
    zs.sort()
    bet = sp.symbols("beta", positive=True)
    tt = -2 * (bet - 4) / bet
    Fb = sp.integrate((x - 1) * (x - 4) - tt * x, (x, 0, 4)) + sp.integrate(-2 * (x - 4) - tt * x, (x, 4, bet))
    bsol = [s for s in sp.solve(sp.simplify(Fb), bet) if s > 4]
    r_exact = sp.simplify(tt.subs(bet, bsol[0]))
    ok3b = len(zs) == 2 and abs(zs[-1] - float(r_exact)) < 1e-9 and r_exact == sp.Rational(-2, 7) and bsol[0] == sp.Rational(14, 3)
    report(2, okF and ok1 and ok2 and ok3a and ok3b,
           f"(2-1) c={c} (2-2) 후보 {vals0}, 다른 t 최소 {min(others):.4f} → g(0)={g0} (2-3)(a) p={pa} (b) g=0 되는 t={['%.6f' % z for z in zs]}, r={r_exact}(β={bsol[0]})")


def p03():
    """재배열의 N, Σ|b_i| — 작은 홀수 n 전수조사로 구조 확인 + n=99 예시 재배열 검사."""
    import itertools
    import math

    def stats(a):
        k = len(a)
        b = [a[(i + 1) % k] - a[i] for i in range(k)]
        N = sum(1 for i in range(k) if b[i] * b[(i + 1) % k] > 0)
        return N, sum(abs(v) for v in b)
    ok = True
    msg = []
    for k in (5, 7, 9):
        m = (k - 1) // 2
        Ns, best, cnt = [], -1, 0
        for perm in itertools.permutations(range(1, k + 1)):
            N, S = stats(perm)
            Ns.append(N)
            if N == 1:
                if S > best:
                    best, cnt = S, 1
                elif S == best:
                    cnt += 1
        ok &= min(Ns) >= 1 and 1 in Ns and best == 2 * m * (m + 1) and cnt == 2 * k * math.factorial(m) ** 2
        msg.append(f"n={k}: minN={min(Ns)}, M={best}, 개수={cnt}")
    # (3-1) 부호 교대 명제: 무작위 확인
    rng = np.random.default_rng(1)
    for _ in range(2000):
        kk = int(rng.integers(2, 30))
        s0 = rng.choice([-1, 1])
        xs = [s0 * (-1) ** i * rng.uniform(0.1, 5) for i in range(kk)]
        ok &= (xs[-1] * xs[0] > 0) == (kk % 2 == 1)
    # n=99 예시 재배열 (*) : a1=50, a_{2k}=50+k, a_{2k+1}=k
    a = [0] * 100
    a[1] = 50
    for k in range(1, 50):
        a[2 * k] = 50 + k
        a[2 * k + 1] = k
    perm = a[1:]
    N99, S99 = stats(perm)
    m = 49
    Mf = 2 * (sum(range(51, 100)) - sum(range(1, 50)))
    count = 2 * 99 * math.factorial(49) ** 2
    ok &= sorted(perm) == list(range(1, 100)) and N99 == 1 and S99 == Mf == 4900 and count == 198 * math.factorial(49) ** 2
    report(3, ok, "; ".join(msg) + f"; n=99 예시 N={N99}, Σ|b|={S99}, M=4900, 개수=198·(49!)^2 [증명-수치확인]")


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
