#!/usr/bin/env python3
"""CAU-G21C (2021학년도 중앙대 수시 논술 자연계열Ⅲ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "CAU-G21C"
WIDTH = 2
RESULTS = []
x, t, n, u = sp.symbols("x t n u", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """6^4 전수, 조건(1회째 5 아님·2회째 5) 아래 X 의 조건부 기댓값 = 172/45."""
    tot, acc = 0, sp.Integer(0)
    dist = {}
    for o in itertools.product(range(1, 7), repeat=4):
        if o[0] == 5 or o[1] != 5:
            continue
        X = next((i + 1 for i, v in enumerate(o) if v == 6), 5)
        tot += 1
        acc += X
        dist[X] = dist.get(X, 0) + 1
    E = acc / tot
    pd = {k: sp.Rational(v, tot) for k, v in sorted(dist.items())}
    ok = E == sp.Rational(172, 45) and pd == {1: sp.Rational(1, 5), 3: sp.Rational(2, 15), 4: sp.Rational(1, 9), 5: sp.Rational(5, 9)}
    report(1, ok, f"분포 {pd}, E[X]={E}")


def p02():
    """(1) 두 반지름(접선의 법선) 사이 예각으로 tan^2 θ(t) 수치적분 = 3√2-4, (2) 이등변 내접삼각형 전수 탐색 최댓값 3√3e^-2."""
    def tan2(tt):
        px = tt**2 / 2
        py = mp.sqrt(tt**2 - px**2)
        v1 = (px, py)               # 원점 중심 원의 법선
        v2 = (px - 1, py)           # (1,0) 중심 원의 법선
        cs = abs(v1[0] * v2[0] + v1[1] * v2[1]) / (mp.hypot(*v1) * mp.hypot(*v2))
        return (1 - cs**2) / cs**2
    I = mp.quad(tan2, [mp.sqrt(2), 2])
    ok1 = abs(I - (3 * mp.sqrt(2) - 4)) < 1e-15
    # 꼭지각 A=α (0<α<π), 외접원 반지름 1: BC=2 sin α, AB=AC=2 sin((π-α)/2)
    g = lambda al: (2 * mp.sin(al))**3 * mp.e**(-2 * 2 * mp.sin((mp.pi - al) / 2))
    grid = [mp.pi * i / 20000 for i in range(1, 20000)]
    i0 = max(range(len(grid)), key=lambda i: g(grid[i]))
    am = mp.findroot(lambda al: mp.diff(g, al), grid[i0])
    ok2 = abs(g(am) - 3 * mp.sqrt(3) * mp.e**(-2)) < 1e-20 and abs(2 * mp.sin((mp.pi - am) / 2) - 1) < 1e-20
    report(2, ok1 and ok2, f"(1) 적분≈{mp.nstr(I, 15)}  (2) 최댓값≈{mp.nstr(g(am), 15)} (y=1, 꼭지각 {mp.nstr(am / mp.pi, 10)}π)")


def p03():
    """(1) 교점·넓이 기호적분 π-2/5, (2) b_n 직접 계산(n=0..300)과 n→∞ 증가 확인 → 최솟값 -68."""
    xs = sorted(r for r in sp.solve(sp.Eq(x**4, 2 / (1 + x**2)), x) if r.is_real)
    A = sp.integrate(2 / (1 + x**2) - x**4, (x, xs[0], xs[-1]))
    ok1 = xs == [-1, 1] and sp.simplify(A - (sp.pi - sp.Rational(2, 5))) == 0
    a = lambda k: 3 if k == 0 else 3 - k
    b = [sum(a(n_ - k) * a(k) for k in range(n_ + 1)) for n_ in range(301)]
    mn = min(b)
    ok2 = mn == -68 and b.index(mn) == 11 and all(b[i + 1] > b[i] for i in range(20, 300))
    report(3, ok1 and ok2, f"(1) 교점 {xs}, 넓이={A}  (2) 최솟값 b_{b.index(mn)}={mn}, b_10={b[10]}")

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
