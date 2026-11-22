#!/usr/bin/env python3
"""CAU-G21A (2021학년도 중앙대 수시 논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "CAU-G21A"
WIDTH = 2
RESULTS = []
x, t, n, u = sp.symbols("x t n u", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _flows(on, edges):
    """작동 부품만 남긴 그래프에서 A→B 연결 여부(무방향 전선)."""
    adj = {}
    for a, b, part in edges:
        if part is None or on[part]:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    seen, stack = {"A"}, ["A"]
    while stack:
        v = stack.pop()
        for w in adj.get(v, ()):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return "B" in seen


def p01():
    """부품 16가지 상태 전수 → x=7/16, y=9/16, 이익 증가 기댓값 1650/7(비율 해석)·75(%p 해석)."""
    ga = [("A", "U", "p"), ("U", "B", "s"), ("A", "D", "q"), ("D", "B", "r")]
    gb = ga + [("U", "D", None)]
    xs = ys = sp.Integer(0)
    for bits in itertools.product([0, 1], repeat=4):
        on = dict(zip("pqrs", bits))
        xs += sp.Rational(1, 16) * int(_flows(on, ga))
        ys += sp.Rational(1, 16) * int(_flows(on, gb))
    gain = 10 * ((ys - xs) / xs * 100) - 50
    gain2 = 10 * ((ys - xs) * 100) - 50
    report(1, xs == sp.Rational(7, 16) and ys == sp.Rational(9, 16) and gain == sp.Rational(1650, 7) and gain2 == 75,
           f"x={xs}, y={ys}, 이익 증가 기댓값 {gain}(≈{float(gain):.2f}) / %p 해석 {gain2}")


def p02():
    """(1) 극좌표 넓이 수치적분 f(θ) → f'(θ)=2 의 해 θ=π/3, (2) 최댓값 위치 격자·국소탐색."""
    def area(th):
        a = mp.pi / 4 - th / 2
        return mp.quad(lambda ph: 1 / (2 * mp.sin(ph) * mp.cos(ph)), [a, mp.pi / 2 - a])
    th0 = mp.findroot(lambda th: mp.diff(area, th) - 2, 1.0)
    closed = -mp.log(mp.tan(mp.pi / 4 - mp.mpf(0.7) / 2))
    ok1 = abs(th0 - mp.pi / 3) < 1e-15 and abs(area(mp.mpf(0.7)) - closed) < 1e-20
    fx = lambda X: (mp.sqrt(X + 1) + 1) / (X + 2 * (mp.sqrt(X + 1) + 1) * mp.cos(mp.sqrt(X + 1) - 1))
    R = mp.pi**2 + 2 * mp.pi
    # 분모 양수 확인 + 격자 최대
    grid = [R * k / 20000 for k in range(20001)]
    vals = [fx(X) for X in grid]
    pos = all(X + 2 * (mp.sqrt(X + 1) + 1) * mp.cos(mp.sqrt(X + 1) - 1) > 0 for X in grid)
    kmax = max(range(len(vals)), key=lambda k: vals[k])
    xm = mp.findroot(lambda X: mp.diff(fx, X), grid[kmax])
    target = (5 * mp.pi / 6 + 1)**2 - 1
    ok2 = pos and abs(xm - target) < 1e-12 and fx(xm) > fx(0) and fx(xm) > fx(R)
    report(2, ok1 and ok2, f"(1) θ={mp.nstr(th0, 15)}  (2) 최대점 x≈{mp.nstr(xm, 15)}, (5π/6+1)^2-1≈{mp.nstr(target, 15)}")


def p03():
    """(1) 정적분 수치 = (2π-4)e^(π/2)+4, (2) 음함수 미분으로 Q(2,3)에서 속도 (13/126, 13/18)."""
    F = lambda X: mp.quad(lambda s: mp.sin(s)**2, [0, X])
    val = mp.quad(lambda X: (2 * X - mp.sin(2 * X)) * mp.e**(F(X)) * mp.sin(X)**2, [0, mp.pi])
    ok1 = abs(val - ((2 * mp.pi - 4) * mp.e**(mp.pi / 2) + 4)) < 1e-15
    X, T = sp.symbols("X T", real=True)
    Y = (X + 1) * (X - 1)**2
    G = (X - sp.cos(T))**2 + (Y - sp.sin(T))**2 - 8
    Ts = [s for s in sp.solve(G.subs(X, 2), T) if s.is_real and 0 <= float(s) <= float(sp.pi / 4)]
    T0 = Ts[0]
    ok_sin = sp.simplify(sp.sin(T0) - sp.Rational(5, 13)) == 0
    dxdt = sp.simplify((-sp.diff(G, T) / sp.diff(G, X)).subs({X: 2, T: T0}))
    dydt = sp.simplify(sp.diff(Y, X).subs(X, 2) * dxdt)
    ok2 = ok_sin and len(Ts) == 1 and dxdt == sp.Rational(13, 126) and dydt == sp.Rational(13, 18)
    report(3, ok1 and ok2, f"(1) 값≈{mp.nstr(val, 15)}  (2) sin t={sp.sin(T0)}, 속도=({dxdt}, {dydt})")

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
